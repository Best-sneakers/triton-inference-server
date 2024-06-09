import json
import logging
from abc import ABC, abstractmethod

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError

from triton_server.core.settings import settings

logger = logging.getLogger("model_handler")


class BaseLoader(ABC):

    @abstractmethod
    def download(self, s3_path, local_path):
        raise NotImplementedError

    @abstractmethod
    def list(self, s3_path):
        raise NotImplementedError


class YandexS3Downloader(BaseLoader):
    def __init__(self):
        self.session_client = self.get_session_client()

    @staticmethod
    def get_session_client():
        s3 = None
        try:
            session = boto3.session.Session()
            s3 = session.client(
                service_name="s3",
                endpoint_url=settings.s3.endpoint_url,
                aws_access_key_id=settings.s3.yandex_access_key,
                aws_secret_access_key=settings.s3.yandex_secret_key,
                config=Config(signature_version="s3v4"),
            )
        except NoCredentialsError:
            logger.error("Credentials not available")
        return s3

    def download(self, s3_path, local_path):

        try:
            print(s3_path)
            self.session_client.download_file(
                settings.s3.bucket, s3_path, local_path
            )
            logger.info(
                f"Successfully downloaded s3://{settings.s3.bucket}/{s3_path}"
                f" to {local_path}"
            )
        except FileNotFoundError:
            logger.error(f"The file was not found: {local_path}")

    def list(self, s3_path):
        files = []

        try:
            paginator = self.session_client.get_paginator("list_objects_v2")
            for result in paginator.paginate(
                Bucket=settings.s3.bucket, Prefix=s3_path
            ):
                if "Contents" in result:
                    for obj in result["Contents"]:
                        files.append(obj["Key"])

        except Exception as e:
            logger.error(f"Error listing objects in folder {s3_path}: {e}")

        return files

    def load_json_from_s3(self, s3_path):
        content = None
        try:
            obj = self.session_client.get_object(
                Bucket=settings.s3.bucket, Key=s3_path
            )
            json_content = obj["Body"].read().decode("utf-8")
            content = json.loads(json_content)

        except NoCredentialsError:
            logger.error("Credentials not available")

        except Exception as e:
            logger.error(
                f"Error loading JSON from s3:"
                f"//{settings.s3.bucket}/{s3_path}: {e}"
            )
        return content
