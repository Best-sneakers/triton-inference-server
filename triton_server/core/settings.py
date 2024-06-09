__all__ = "settings"

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Efficientnet(BaseSettings):
    class Config:
        env_prefix = "S3_"

    models_path: str = "onnx_models/efficientnet"
    model_loss: str = "val_loss"
    load_labels: bool = True


class S3(BaseSettings):
    class Config:
        env_prefix = "S3_"

    bucket: str = "hse-sneakers"
    yandex_access_key: str = ""
    yandex_secret_key: str = ""
    endpoint_url: str = "https://storage.yandexcloud.net"


class ProjectSettings(BaseSettings):
    """Represents Project settings"""

    class Config:
        env_prefix = "SETTINGS_"

    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    project_name: str = "triton_server"
    log_to_file: bool = False
    models_path: str = "/models"


class Settings(BaseSettings):
    project: ProjectSettings = ProjectSettings()
    s3: S3 = S3()
    efficientnet: Efficientnet = Efficientnet()


@lru_cache
def get_settings() -> Settings:
    """Singleton"""
    return Settings()


settings = get_settings()
