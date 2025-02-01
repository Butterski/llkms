import os
import shutil
from pathlib import Path
import argparse
import asyncio
from typing import List

from dotenv import load_dotenv

from utils.aws.s3_client import S3Client
from utils.langchain.document_processor import DocumentProcessor
from utils.langchain.rag_pipeline import RAGPipeline
from utils.vector_store_manager import VectorStoreManager
from utils.logger import logger

class DocumentProcessingPipeline:
    def __init__(self):
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
            "successful_requests": 0
        }

    async def process_s3_bucket_async(self, bucket: str, prefix: str = "", model_provider: str = "deepseek",
                                        model: str = "deepseek-chat", reindex: bool = False) -> RAGPipeline:
        """Asynchronously process files from S3 or load cached vector store, then create RAG pipeline."""
        if self.vector_cache.exists() and not reindex:
            logger.info("Loading local vector store from cache.")
            vector_store = self.vector_cache.load(self.doc_processor.embeddings)
            if vector_store is not None:
                return RAGPipeline(vector_store, model_provider=model_provider, model=model)

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
        return RAGPipeline(vector_store, model_provider=model_provider, model=model)

    async def async_process_file(self, bucket: str, file_key: str) -> List:
        """Asynchronously download and process a single file from S3."""
        logger.info(f"Processing file asynchronously: {file_key}")
        loop = asyncio.get_event_loop()
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
        self.total_usage["total_tokens"] += usage.get("total_tokens", 0)
        self.total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.total_usage["total_cost"] += usage.get("total_cost", 0.0)
        self.total_usage["successful_requests"] += usage.get("successful_requests", 0)

    def cleanup(self):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.debug("Temporary directory removed")

def parse_args():
    parser = argparse.ArgumentParser(description='LLKMS - Language Learning Knowledge Management System')
    parser.add_argument('--model-provider', type=str, default='deepseek',
                        choices=['deepseek', 'openai'],
                        help='The model provider to use (default: deepseek)')
    parser.add_argument('--model', type=str, default='deepseek-chat',
                        help='The specific model to use (default: deepseek-chat)')
    parser.add_argument('--reindex', action="store_true",
                        help='Force reindexing of the vector store even if a local cache exists')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting LLKMS application with {args.model_provider} provider and {args.model} model")
    pipeline = DocumentProcessingPipeline()

    try:
        rag = asyncio.run(pipeline.process_s3_bucket_async(
            bucket="eng-llkms",
            prefix="knowledge",
            model_provider=args.model_provider,
            model=args.model,
            reindex=args.reindex
        ))
        logger.info("Starting interactive query loop")
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == "quit":
                logger.info("User requested to quit")
                break
            try:
                logger.debug(f"Processing question: {question}")
                answer, usage = rag.query(question)
                pipeline._update_usage(usage)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"Error processing question: {e}")
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
        print(summary)
        logger.info(summary)

if __name__ == "__main__":
    main()
