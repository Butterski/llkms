import argparse
import asyncio
import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from llkms.utils.aws.s3_client import S3Client
from llkms.utils.interactive_query import run_interactive_query
from llkms.utils.langchain.document_processor import DocumentProcessor
from llkms.utils.langchain.model_factory import ModelConfig, ModelFactory
from llkms.utils.langchain.rag_pipeline import RAGPipeline
from llkms.utils.logger import logger
from llkms.utils.vector_store_manager import VectorStoreManager


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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LLKMS - Language Learning Knowledge Management System")
    parser.add_argument(
        "--model-provider",
        type=str,
        default="deepseek",
        choices=["deepseek", "openai"],
        help="The model provider to use (default: deepseek)",
    )
    parser.add_argument(
        "--model", type=str, default="deepseek-chat", help="The specific model to use (default: deepseek-chat)"
    )
    parser.add_argument("--temperature", type=float, default=1, help="Model temperature (0.0-1.0)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens for model response")
    parser.add_argument("--api-key", type=str, help="Optional API key (otherwise reads from env)")
    parser.add_argument("--api-base", type=str, help="Optional API base URL")
    parser.add_argument(
        "--reindex", action="store_true", help="Force reindexing of the vector store even if a local cache exists"
    )
    return parser.parse_args()


def create_model_config(args) -> ModelConfig:
    """Create model configuration from arguments."""
    return ModelConfig(
        provider=args.model_provider,
        model_name=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


def main():
    """Main entry point for the LLKMS application."""
    args = parse_args()
    logger.info(f"Starting LLKMS application with {args.model_provider} provider and {args.model} model")
    pipeline = DocumentProcessingPipeline()
    model_config = create_model_config(args)

    try:
        rag = asyncio.run(
            pipeline.process_s3_bucket_async(
                bucket="eng-llkms", prefix="lectures", model_config=model_config, reindex=args.reindex
            )
        )
        run_interactive_query(rag, pipeline._update_usage)
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        logger.info("Shutting down LLKMS application")
        pipeline.cleanup()
        usage = pipeline.total_usage
        summary = "\nToken Usage and Cost Summary:\n" + "-" * 40
        summary += f"\nTotal Tokens: {usage['total_tokens']:,}"
        summary += f"\nPrompt Tokens: {usage['prompt_tokens']:,}"
        summary += f"\nCompletion Tokens: {usage['completion_tokens']:,}"
        summary += f"\nSuccessful Requests: {usage['successful_requests']}"
        summary += f"\nTotal Cost (USD): ${usage['total_cost']:.4f}"
        logger.info(summary)


if __name__ == "__main__":
    main()
