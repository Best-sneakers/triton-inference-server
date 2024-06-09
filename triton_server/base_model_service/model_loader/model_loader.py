import logging
import os
import re

from triton_server.base_model_service.model_loader.base_model_loader import (
    YandexS3Downloader,
)
from triton_server.core.settings import settings

logger = logging.getLogger("model_handler")


class EfficientNetS3Loader(YandexS3Downloader):
    def __init__(self):
        super().__init__()
        self.model_loss = settings.efficientnet.model_loss
        self.models_path = settings.efficientnet.models_path

    def get_best_model(self, s3_folder_path, local_folder_path):
        files = self.list(s3_folder_path)
        best_file = None
        best_loss = float("inf")

        loss_pattern = re.compile(r"val_loss_(\d+\.\d+)")

        for file_key in files:
            match = loss_pattern.search(file_key)
            if match:
                val_loss = float(match.group(1))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_file = file_key

        if best_file:
            local_file_path = os.path.join(
                local_folder_path, os.path.basename(best_file)
            )
            self.download(best_file, local_file_path)
            logger.info(
                f"Downloaded best model {best_file} to {local_file_path}"
            )

            return local_file_path

        logger.error("No valid model found in the specified S3 folder")
        return best_file
