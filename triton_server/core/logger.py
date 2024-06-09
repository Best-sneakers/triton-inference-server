__all__ = "LOGGING"
from pathlib import Path

log_dir = Path("/var/logs")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "ecs_logging.StdlibFormatter",
        },
    },
    "handlers": {
        "model_handler": {
            "level": "INFO",
            "formatter": "json",
            "class": "logging.FileHandler",
            "filename": log_dir / "triton_inference.json",
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
        },
        "model_handler": {
            "handlers": ["model_handler", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "efficientnet_handler": {
            "handlers": ["model_handler", "console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": [
            "console",
        ],
    },
}
