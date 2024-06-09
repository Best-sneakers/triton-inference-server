import argparse
import logging
from logging import config as logging_config

import numpy as np
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton

from triton_server.core.logger import LOGGING
from triton_server.core.settings import settings
from triton_server.efficientnet_service.efficientnet import EfficientNetModel


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Batch size of request.",
        required=False,
    )
    return parser.parse_args()


def main():
    args = get_args()

    if settings.project.log_to_file:
        logging_config.dictConfig(LOGGING)

    efficientnet_model = EfficientNetModel()

    logger = logging.getLogger("model_handler")

    with Triton() as triton:
        logger.info("Loading ONNX EfficientNet model.")
        triton.bind(
            model_name="efficientnet",
            infer_func=efficientnet_model.infer,
            inputs=[
                Tensor(name="input_data", dtype=np.uint8, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="label", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(
                max_batch_size=args.max_batch_size,
                batcher=DynamicBatcher(max_queue_delay_microseconds=5000),
            ),
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()
