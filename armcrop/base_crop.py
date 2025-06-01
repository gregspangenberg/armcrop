import onnxruntime as rt
import SimpleITK as sitk
import pathlib
from typing import Tuple

from copy import deepcopy
from armcrop.model_manager import ModelManager

# disable logging
rt.set_default_logger_severity(3)


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
        # Use the model manager to get the cached model
        model_type = "obb" if self.__class__.__name__ == "OBBCrop2Bone" else "default"
        model_manager = ModelManager.get_instance(model_type)
        model = model_manager.load_model()

        # Load the volume
        vol, vol_t = self.load_volume(volume, img_size)

        return model, vol, vol_t
