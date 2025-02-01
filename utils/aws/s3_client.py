import os
import shutil
from pathlib import Path
from typing import List

import boto3
from dotenv import load_dotenv

from ..logger import logger

class S3Client:
    def __init__(self):
        """Initialize the S3 client using AWS credentials from environment variables."""
        load_dotenv()
        self.client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        logger.info("Initialized S3 client")

    def list_files(self, bucket: str, prefix: str = "") -> List[str]:
        """
        List all files in an S3 bucket with a given prefix.

        Args:
            bucket (str): S3 bucket name.
            prefix (str, optional): Prefix filter. Defaults to "".

        Returns:
            List[str]: List of file keys.
        """
        logger.info(f"Listing files in bucket {bucket} with prefix '{prefix}'")
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = [obj["Key"] for obj in response.get("Contents", [])]
        logger.debug(f"Found {len(files)} files")
        return files

    def download_file(self, bucket: str, key: str, local_path: Path):
        """
        Download a single file from S3 to local path.

        Args:
            bucket (str): S3 bucket name.
            key (str): File key in the bucket.
            local_path (Path): Local file path to save the file.

        Raises:
            Exception: Propagates any exception encountered during download.
        """
        try:
            logger.debug(f"Downloading {key} to {local_path}")
            # Create directory structure
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # If file exists, remove it first
            if local_path.exists():
                logger.debug(f"Removing existing file: {local_path}")
                local_path.unlink()

            # Download file
            self.client.download_file(bucket, key, str(local_path))
            logger.debug(f"Successfully downloaded {key}")

        except Exception as e:
            logger.error(f"Error downloading file {key}: {str(e)}")
            raise
