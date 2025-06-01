import onnxruntime as rt
import numpy as np
import pathlib
import huggingface_hub
from typing import Dict


class ModelManager:
    """
    Singleton class that loads and caches ML models for bone detection
    """

    _instances: Dict[str, "ModelManager"] = {}

    @classmethod
    def get_instance(cls, model_type="default"):
        if model_type not in cls._instances:
            cls._instances[model_type] = cls(model_type)
        return cls._instances[model_type]

    def __init__(self, model_type="default"):
        self.model = None
        self.model_type = model_type
        self.img_size = (640, 640)

    def load_model(self):
        """Load the model if it hasn't been loaded already"""
        if self.model is not None:
            return self.model

        model_path = self._get_model_path()

        # load model
        with open(model_path, "rb") as file:
            try:
                import torch

                providers = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": torch.cuda.current_device(),
                            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                        },
                    )
                ]
            except:
                print("Using CPUExecutionProvider")
                providers = ["CPUExecutionProvider"]
            self.model = rt.InferenceSession(file.read(), providers=providers)

        # prime the model on a random image
        self.model.run(
            None,
            {"images": np.random.rand(1, 3, self.img_size[0], self.img_size[1]).astype(np.float32)},
        )

        return self.model

    def _get_model_path(self):
        """Get the appropriate model path based on model_type"""
        if self.model_type == "obb":
            return pathlib.Path(
                huggingface_hub.hf_hub_download(
                    repo_id="gregspangenberg/armcrop",
                    filename="upperarm_yolo11_obb_6.onnx",
                )
            )
        else:  # default
            return pathlib.Path(
                huggingface_hub.hf_hub_download(
                    repo_id="gregspangenberg/armcrop",
                    filename="yolov9c_upperlimb.onnx",
                )
            )
