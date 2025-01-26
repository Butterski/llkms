import os
import shutil
from pathlib import Path
from typing import Dict, Any
import argparse

from dotenv import load_dotenv

from utils.aws.s3_client import S3Client
from utils.langchain.document_processor import DocumentProcessor
from utils.langchain.rag_pipeline import RAGPipeline
from utils.logger import logger


class DocumentProcessingPipeline:
    def __init__(self):
        load_dotenv(override=True)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.s3_client = S3Client()
        self.doc_processor = DocumentProcessor()
        self.temp_dir = Path("temp")

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

    def process_s3_bucket(self, bucket: str, prefix: str = "", model_provider: str = "deepseek", model: str = "deepseek-chat") -> RAGPipeline:
        """Process all files in S3 bucket and create RAG pipeline"""
        logger.info(f"Starting to process bucket {bucket} with prefix '{prefix}'")
        files = self.s3_client.list_files(bucket, prefix)
        documents = []

        for file_key in files:
            if file_key.endswith("/"):
                logger.debug(f"Skipping directory: {file_key}")
                continue

            try:
                logger.info(f"Processing file: {file_key}")
                relative_path = Path(file_key)
                local_path = self.temp_dir / relative_path

                self.s3_client.download_file(bucket, file_key, local_path)

                if file_key.lower().endswith(".txt"):
                    logger.debug(f"Processing text file: {file_key}")
                    docs = self.doc_processor.process_text(local_path.read_text())
                elif file_key.lower().endswith(".pdf"):
                    logger.debug(f"Processing PDF file: {file_key}")
                    docs = self.doc_processor.process_pdf(local_path)
                elif file_key.lower().endswith((".png", ".jpg", ".jpeg")):
                    logger.debug(f"Processing image file: {file_key}")
                    docs = self.doc_processor.process_image(local_path)
                else:
                    logger.warning(f"Skipping unsupported file type: {file_key}")
                    continue

                documents.extend(docs)
                logger.debug(f"Successfully processed {file_key}")

            except Exception as e:
                logger.error(f"Error processing file {file_key}: {str(e)}")
                continue

        if not documents:
            logger.error("No documents were successfully processed")
            raise ValueError("No documents were successfully processed")

        logger.info(f"Creating vector store with {len(documents)} documents")
        vector_store, usage = self.doc_processor.create_vector_store(documents)
        self._update_usage(usage)
        return RAGPipeline(vector_store, model_provider=model_provider, model=model)
    
    def _update_usage(self, usage: Dict[str, Any]):
        """Update total usage statistics"""
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
    parser.add_argument(
        '--model-provider', 
        type=str, 
        default='deepseek',
        choices=['deepseek', 'openai'],
        help='The model provider to use (default: deepseek)'
    )
    parser.add_argument(
        '--model', 
        type=str,
        default='deepseek-chat',
        help='The specific model to use (default: deepseek-chat)'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting LLKMS application with {args.model_provider} provider and {args.model} model")
    pipeline = DocumentProcessingPipeline()

    try:
        logger.info("Initializing RAG system")
        rag = pipeline.process_s3_bucket(
            bucket="eng-llkms", 
            prefix="knowledge", 
            model_provider=args.model_provider,
            model=args.model
        )

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
