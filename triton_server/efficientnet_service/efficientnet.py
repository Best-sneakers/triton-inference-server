import io
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image
from pytriton.decorators import batch

from triton_server.base_model_service.base_model import BaseModel
from triton_server.base_model_service.model_loader.model_loader import (
    EfficientNetS3Loader,
)
from triton_server.core.settings import settings

logger = logging.getLogger("efficientnet_handler")


class EfficientNetModel(BaseModel):
    def __init__(self):

        self.ort_session = None
        self.loader = EfficientNetS3Loader()
        self.labels = None
        self.set_up()

    def set_up(self):
        logger.info("Setting up EfficientNetModel")
        model_path = self.loader.get_best_model(
            settings.efficientnet.models_path,
            settings.project.base_dir / "models",
        )
        self.ort_session = ort.InferenceSession(model_path)
        if settings.efficientnet.load_labels:
            self.labels = self.loader.load_json_from_s3(
                settings.efficientnet.models_path + "/labels.json"
            )
            logger.info(self.labels)

    @batch
    def infer(self, input_data: np.ndarray):
        logger.debug(f"Inputs: {input_data[1]}")

        images = []

        for img in input_data:
            img = Image.open(io.BytesIO(img.tobytes()))
            img = img.resize((224, 224))
            img = np.array(img) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            images.append(img)

        images = np.stack(images)
        outputs = self.ort_session.run(None, {"input": images})

        decoded_outputs = []
        for output in outputs[0]:
            logger.debug(1)
            top_index = np.argmax(output)
            label = self.labels["labels"][top_index]
            decoded_outputs.append(label)

        logger.debug(f"Outputs: {decoded_outputs}")
        return {"label": np.char.encode(decoded_outputs, "utf-8")}
