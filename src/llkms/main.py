import argparse
import asyncio
import os
import yaml
from typing import Dict

from dotenv import load_dotenv

from llkms.utils.interactive_query import run_interactive_query
from llkms.utils.langchain.document_processor import DocumentProcessingPipeline
from llkms.utils.langchain.model_factory import ModelConfig
from llkms.utils.logger import logger


def resolve_env_vars(value: str) -> str:
    """Resolve environment variables with fallback to .env file."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, "")
    return value

def process_env_vars(config: Dict) -> Dict:
    """Recursively process dictionary and resolve environment variables."""
    if isinstance(config, dict):
        return {k: process_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [process_env_vars(v) for v in config]
    else:
        return resolve_env_vars(config)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file and resolve environment variables."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return process_env_vars(config)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LLKMS - Language Learning Knowledge Management System")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument("--reindex", action="store_true", help="Force reindexing of the vector store even if a local cache exists")
    return parser.parse_args()


def create_model_config(config: Dict) -> ModelConfig:
    """Create model configuration from config file."""
    return ModelConfig(
        provider=config['model']['provider'],
        model_name=config['model']['model'],
        api_key=config['model']['api_key'],
        api_base=config['model']['api_base'],
        max_tokens=config['model']['max_tokens'],
        temperature=config['model']['temperature'],
    )


def main():
    """Main entry point for the LLKMS application."""
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    
    if not config['aws']['access_key_id'] or not config['aws']['secret_access_key']:
        raise ValueError("AWS credentials not found in environment or .env file")
    if not config['model']['api_key']:
        raise ValueError("Model API key not found in environment or .env file")

    logger.info(f"Starting LLKMS application with {config['model']['provider']} provider and {config['model']['model']} model")
    pipeline = DocumentProcessingPipeline()
    model_config = create_model_config(config)

    try:
        rag = asyncio.run(
            pipeline.process_s3_bucket_async(
                bucket=config['aws']['bucket'],
                prefix=config['aws']['prefix'],
                model_config=model_config,
                reindex=args.reindex
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
