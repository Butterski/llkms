"""Local embedding implementations using Ollama."""

from typing import List

import requests
from langchain_core.embeddings import Embeddings

from llkms.utils.logger import logger


class OllamaEmbeddings(Embeddings):
    """
    Local embeddings using Ollama API.

    Ollama provides local embedding models like nomic-embed-text,
    mxbai-embed-large, all-minilm, etc.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """
        Initialize OllamaEmbeddings.

        Args:
            model (str): The Ollama embedding model to use.
                         Recommended: nomic-embed-text, mxbai-embed-large
            base_url (str): Base URL of the Ollama server.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Validate that Ollama server is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]

            # Check if embedding model is available (may have :latest suffix)
            model_base = self.model.split(":")[0]
            available = any(model_base in m for m in models)

            if not available:
                logger.warning(
                    f"Embedding model '{self.model}' not found in Ollama. "
                    f"Available models: {models}. "
                    f"Run 'ollama pull {self.model}' to download it."
                )
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to Ollama at {self.base_url}. " "Make sure Ollama is running.")
        except Exception as e:
            logger.warning(f"Error validating Ollama connection: {e}")

    def _embed(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: Embedding vector.

        Raises:
            Exception: If embedding request fails.
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text}, timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting embedding from Ollama: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        logger.info(f"Embedding {len(texts)} documents with Ollama ({self.model})")
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                logger.debug(f"Embedded {i + 1}/{len(texts)} documents")
            embeddings.append(self._embed(text))
        logger.info(f"Completed embedding {len(texts)} documents")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text (str): Query text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        logger.debug(f"Embedding query with Ollama ({self.model})")
        return self._embed(text)
