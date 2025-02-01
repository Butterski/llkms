from pathlib import Path
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader, UnstructuredImageLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from PIL import Image
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