import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llkms.utils.aws.s3_client import S3Client
from llkms.utils.langchain.embeddings import OllamaEmbeddings
from llkms.utils.langchain.model_factory import ModelConfig
from llkms.utils.langchain.rag_pipeline import RAGPipeline
from llkms.utils.langchain.vector_store_manager import VectorStoreManager
from llkms.utils.logger import logger


class DocumentProcessor:
    def __init__(self, embedding_config: Optional[Dict[str, str]] = None):
        """
        Initialize DocumentProcessor.

        Args:
            embedding_config (dict, optional): Embedding configuration with keys:
                - provider: "openai" or "ollama"
                - model: Model name (e.g., "text-embedding-3-small" or "nomic-embed-text")
                - api_base: (optional) Custom API base URL for Ollama
        """
        if embedding_config is None:
            embedding_config = {"provider": "openai", "model": "text-embedding-3-small"}

        self.embedding_config = embedding_config
        provider = embedding_config.get("provider", "openai")
        model = embedding_config.get("model")

        if provider == "ollama":
            api_base = embedding_config.get("api_base", "http://localhost:11434")
            self.embeddings = OllamaEmbeddings(model=model or "nomic-embed-text", base_url=api_base)
            self.is_local = True
            logger.info(f"Using Ollama embeddings with model: {model or 'nomic-embed-text'}")
        else:
            self.embeddings = OpenAIEmbeddings(model=model or "text-embedding-3-small")
            self.is_local = False
            logger.info(f"Using OpenAI embeddings with model: {model or 'text-embedding-3-small'}")

        # Reduced chunk size to 1000 to avoid truncation with 512-token limit models (like mxbai-embed-large)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_text(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Convert text content into document chunks.

        Args:
            content (str): The text content to process.
            metadata (dict, optional): Metadata to attach to the documents.

        Returns:
            List[Document]: List of document chunks.
        """
        metadatas = [metadata] if metadata else None
        return self.text_splitter.create_documents([content], metadatas=metadatas)

    def process_pdf(self, file_path: Path) -> List[Document]:
        """
        Process a PDF file into document chunks.

        Args:
            file_path (Path): Path to the PDF file.

        Returns:
            List[Document]: Document chunks extracted from the PDF.
        """
        loader = PyPDFLoader(str(file_path))
        return self.text_splitter.split_documents(loader.load())

    def process_image(self, file_path: Path) -> List[Document]:
        """
        Process an image file into a document.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            List[Document]: Document generated from the image.
        """
        loader = UnstructuredImageLoader(str(file_path))
        return loader.load()

    def process_docx(self, file_path: Path) -> List[Document]:
        """
        Process a DOCX file into document chunks.

        Args:
            file_path (Path): Path to the DOCX file.

        Returns:
            List[Document]: Document chunks from the DOCX.
        """
        try:
            doc = DocxDocument(str(file_path))
            full_text = "\n".join([para.text for para in doc.paragraphs])
            return self.process_text(full_text, metadata={"source": str(file_path)})
        except Exception as e:
            logger.error(f"Error processing DOCX file {file_path}: {str(e)}")
            return []

    def process_html(self, file_path: Path) -> List[Document]:
        """
        Process an HTML file into document chunks.

        Args:
            file_path (Path): Path to the HTML file.

        Returns:
            List[Document]: Document chunks extracted from HTML.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator="\n")
            return self.process_text(text, metadata={"source": str(file_path)})
        except Exception as e:
            logger.error(f"Error processing HTML file {file_path}: {str(e)}")
            return []

    def create_vector_store(self, documents: List[Document]) -> Tuple[FAISS, Dict[str, Any]]:
        """
        Create a FAISS vector store from documents.

        Args:
            documents (List[Document]): List of documents to index.

        Returns:
            Tuple[FAISS, Dict[str, Any]]: The vector store and usage statistics.
        """
        if self.is_local:
            # Local embeddings don't use OpenAI callback
            vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created vector store with {len(documents)} document chunks (local embeddings)")
            return vector_store, {
                "total_tokens": 0,
                "total_cost": 0.0,
                "successful_requests": len(documents),
            }
        else:
            with get_openai_callback() as cb:
                vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created vector store with {len(documents)} document chunks")
                return vector_store, {
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost,
                    "successful_requests": cb.successful_requests,
                }


class DocumentProcessingPipeline:
    def __init__(self, embedding_config: Optional[Dict[str, str]] = None, retriever_k: int = 8):
        """
        Initialize DocumentProcessingPipeline.

        Args:
            embedding_config (dict, optional): Embedding configuration with keys:
                - provider: "openai" or "ollama"
                - model: Model name
                - api_base: (optional) Custom API base URL
            retriever_k (int): Number of documents to retrieve for RAG context.
        """
        load_dotenv(override=True)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.s3_client = S3Client()
        self.doc_processor = DocumentProcessor(embedding_config=embedding_config)
        self.temp_dir = Path("temp")
        self.vector_cache = VectorStoreManager()
        self.retriever_k = retriever_k

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        logger.info("Initialized DocumentProcessingPipeline")

        self.total_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "successful_requests": 0,
        }

    async def process_s3_bucket_async(
        self, bucket: str, prefix: str = "", model_config: ModelConfig = None, reindex: bool = False
    ) -> RAGPipeline:
        """Process files from an S3 bucket asynchronously or load a cached vector store.

        Args:
            bucket (str): S3 bucket name.
            prefix (str, optional): Prefix filter. Defaults to "".
            model_config (ModelConfig, optional): Model configuration. Defaults to None.
            reindex (bool, optional): Force reindexing if True. Defaults to False.

        Returns:
            RAGPipeline: The RAG pipeline initialized with the vector store.
        """
        if self.vector_cache.exists() and not reindex:
            logger.info("Loading local vector store from cache.")
            vector_store = self.vector_cache.load(self.doc_processor.embeddings)
            if vector_store is not None:
                return RAGPipeline(vector_store, model_config=model_config, retriever_k=self.retriever_k)

        logger.info(f"Starting to process bucket {bucket} with prefix '{prefix}'")

        files = self.s3_client.list_files(bucket, prefix)
        tasks = []

        for file_key in files:
            if file_key.endswith("/"):
                logger.debug(f"Skipping directory: {file_key}")
                continue
            tasks.append(self.async_process_file(bucket, file_key))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        documents = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing file: {result}")
            elif result:
                documents.extend(result)

        if not documents:
            logger.error("No documents were successfully processed")
            raise ValueError("No documents were successfully processed")

        logger.info(f"Creating vector store with {len(documents)} documents")

        vector_store, usage = self.doc_processor.create_vector_store(documents)

        self._update_usage(usage)
        self.vector_cache.save(vector_store)

        logger.info("Vector store saved locally.")

        return RAGPipeline(vector_store, model_config=model_config, retriever_k=self.retriever_k)

    async def async_process_file(self, bucket: str, file_key: str) -> List:
        """Asynchronously download and process a single file from S3.

        Args:
            bucket (str): S3 bucket name.
            file_key (str): Key of the file in the bucket.

        Returns:
            List: Documents processed from the file.
        """
        logger.info(f"Processing file asynchronously: {file_key}")
        loop = asyncio.get_running_loop()
        relative_path = Path(file_key)
        local_path = self.temp_dir / relative_path

        await loop.run_in_executor(None, self.s3_client.download_file, bucket, file_key, local_path)

        # Choose processing method based on file extension
        if file_key.lower().endswith(".txt"):
            return await loop.run_in_executor(
                None,
                lambda: self.doc_processor.process_text(
                    local_path.read_text(encoding="utf-8"), metadata={"source": str(local_path)}
                ),
            )
        elif file_key.lower().endswith(".pdf"):
            return await loop.run_in_executor(None, self.doc_processor.process_pdf, local_path)
        elif file_key.lower().endswith((".png", ".jpg", ".jpeg")):
            return await loop.run_in_executor(None, self.doc_processor.process_image, local_path)
        elif file_key.lower().endswith(".docx"):
            return await loop.run_in_executor(None, self.doc_processor.process_docx, local_path)
        elif file_key.lower().endswith((".html", ".htm")):
            return await loop.run_in_executor(None, self.doc_processor.process_html, local_path)
        else:
            logger.warning(f"Skipping unsupported file type: {file_key}")
            return []

    def _update_usage(self, usage: dict):
        """Update the cumulative usage statistics.

        Args:
            usage (dict): Dictionary with usage statistics.
        """
        self.total_usage["total_tokens"] += usage.get("total_tokens", 0)
        self.total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.total_usage["total_cost"] += usage.get("total_cost", 0.0)
        self.total_usage["successful_requests"] += usage.get("successful_requests", 0)

    def cleanup(self):
        """Clean up temporary files used during processing."""
        logger.info("Cleaning up temporary files")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.debug("Temporary directory removed")
