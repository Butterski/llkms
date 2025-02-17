import asyncio
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from PIL import Image
from llkms.utils.aws.s3_client import S3Client
from llkms.utils.langchain.model_factory import ModelConfig
from llkms.utils.langchain.rag_pipeline import RAGPipeline
from llkms.utils.langchain.vector_store_manager import VectorStoreManager
from llkms.utils.logger import logger


class DocumentProcessor:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize DocumentProcessor.

        Args:
            embedding_model (str, optional): The embedding model to use. Defaults to "text-embedding-3-small".
        """
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def process_text(self, content: str) -> List[Document]:
        """
        Convert text content into document chunks.

        Args:
            content (str): The text content to process.

        Returns:
            List[Document]: List of document chunks.
        """
        return self.text_splitter.create_documents([content])

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
            return self.process_text(full_text)
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
            return self.process_text(text)
        except Exception as e:
            logger.error(f"Error processing HTML file {file_path}: {str(e)}")
            return []

    def create_vector_store(self, documents: List[Document], model_provider: str = "deepseek", model: str = "deepseek-chat") -> Tuple[FAISS, Dict[str, Any]]:
        """
        Create a FAISS vector store from documents.

        Args:
            documents (List[Document]): List of documents to index.
            model_provider (str, optional): The provider of the model. Defaults to "deepseek".
            model (str, optional): The model name. Defaults to "deepseek-chat".

        Returns:
            Tuple[FAISS, Dict[str, Any]]: The vector store and usage statistics.
        """
        with get_openai_callback() as cb:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created vector store using {model_provider} {model}")
            return vector_store, {
                "total_tokens": cb.total_tokens,
                "total_cost": cb.total_cost,
                "successful_requests": cb.successful_requests
            }
        
class DocumentProcessingPipeline:
    def __init__(self):
        """Initialize DocumentProcessingPipeline by setting up S3 client, document processor, and temporary directories."""
        load_dotenv(override=True)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.s3_client = S3Client()
        self.doc_processor = DocumentProcessor()
        self.temp_dir = Path("temp")
        self.vector_cache = VectorStoreManager()

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
                return RAGPipeline(vector_store, model_config=model_config)

        logger.info(f"Starting to process bucket {bucket} with prefix '{prefix}'")
        files = self.s3_client.list_files(bucket, prefix)
        tasks = []
        for file_key in files:
            if file_key.endswith("/"):
                logger.debug(f"Skipping directory: {file_key}")
                continue
            tasks.append(self.async_process_file(bucket, file_key))
        # Gather results concurrently
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
        return RAGPipeline(vector_store, model_config=model_config)

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

        # Download file using run_in_executor (since boto3 is synchronous)
        await loop.run_in_executor(None, self.s3_client.download_file, bucket, file_key, local_path)

        # Choose processing method based on file extension
        if file_key.lower().endswith(".txt"):
            return await loop.run_in_executor(None, lambda: self.doc_processor.process_text(local_path.read_text()))
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
