import onnxruntime as rt
import SimpleITK as sitk
import numpy as np
import pathlib
from typing import Tuple, List, Dict

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor


class BaseCrop:
    """
    Base class for cropping volumes using a machine learning model.
    """

    def __init__(self, img_size=(640, 640)):
        """
        Initialize the BaseCrop class.

        Args:
            img_size: Image size required for the ML model. Defaults to (640, 640).
        """
        self.img_size = img_size
        self.model = None
        self.vol = None
        self.vol_t = None

    def _get_model(self) -> pathlib.Path:
        """
        Get the path to the ML model.

        Returns:
            Path to the ML model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def load_model(self, img_size) -> rt.InferenceSession:
        """
        Load the ML model for inference

        Args:
            img_size: Image size required for the ML model

        Returns:
            model: The ML model for inference
        """
        # load model
        with open(self._get_model(), "rb") as file:
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
            model = rt.InferenceSession(file.read(), providers=providers)

        # prime the model on a random image, to get the slow first inference out of the way
        model.run(
            None,
            {"images": np.random.rand(1, 3, img_size[0], img_size[1]).astype(np.float32)},
        )

        return model

    def load_volume(
        self, volume: pathlib.Path | sitk.Image, img_size=(640, 640)
    ) -> Tuple[sitk.Image, sitk.Image]:
        """
        Load a volume and preprcoess it for inference

        Args:
            volume: Path to the input volume for inference, or the sitk.Image volume
            img_size: Image size required for the ML model. Defaults to (640, 640).

        Returns:
            vol: The original volume
            vol_t: The preprocessed volume
        """
        if isinstance(volume, sitk.Image):
            vol = volume
        else:
            vol = sitk.ReadImage(str(volume))
        vol_t = deepcopy(vol)
        vol_t = sitk.Cast(vol_t, sitk.sitkFloat32)
        vol_t = sitk.Clamp(vol_t, sitk.sitkFloat32, -1024, 3000)
        vol_t = sitk.RescaleIntensity(vol_t, 0, 1)

        new_size = [img_size[0], img_size[1], vol_t.GetDepth()]

        reference_image = sitk.Image(new_size, vol_t.GetPixelIDValue())
        reference_image.SetOrigin(vol_t.GetOrigin())
        reference_image.SetDirection(vol_t.GetDirection())
        reference_image.SetSpacing(
            [sz * spc / nsz for nsz, sz, spc in zip(new_size, vol_t.GetSize(), vol_t.GetSpacing())]
        )

        vol_t = sitk.Resample(vol_t, reference_image)

        return vol, vol_t

    def load(self, volume, img_size) -> Tuple[rt.InferenceSession, sitk.Image, sitk.Image]:
        """
        Load the ML model and the volume for inference

        Args:
            volume: path to the volume for inference, or an sitk.Image object
            img_size: pixel size required for the ML model

        Returns:
            model: The ML model for inference
            vol: The original volume
            vol_t: The preprocessed volume
        """
        with ThreadPoolExecutor() as executor:
            # Apply the tasks asynchronously
            volume_result = executor.submit(self.load_volume, volume, img_size)
            model_result = executor.submit(self.load_model, img_size)

            # Wait for results
            vol, vol_t = volume_result.result()
            model = model_result.result()
        return model, vol, vol_t
