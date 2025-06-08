from armcrop.group import UniqueObjectProcessor
from armcrop.detect import YOLODetector
import abc
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from math import ceil


class CropBase(abc.ABC):
    def __init__(
        self,
        volume: sitk.Image | Path | str,
        detection_confidence: float = 0.5,
        detection_iou: float = 0.5,
    ):
        """
        Initialize the cropper with a volume.

        Args:
            volume (sitk.Image | Path | str): The volume to be cropped, can be a SimpleITK image or a file path.
        """
        if isinstance(volume, (Path, str)):
            self.volume = sitk.ReadImage(str(volume))
        elif isinstance(volume, sitk.Image):
            self.volume = volume
        else:
            raise TypeError("Volume must be a SimpleITK Image or a file path.")

        self._det = YOLODetector()
        self._det.predict(self.volume, detection_confidence, detection_iou)
        self._uop = UniqueObjectProcessor(self._det, self.volume)
        self._lblmap = {
            "clavicle": 0,
            "scapula": 1,
            "humerus": 2,
            "radius-ulna": 3,
            "hand": 4,
        }

    def _check_bone_request(self, requested_bone):
        if requested_bone not in self._lblmap.keys():
            raise ValueError(
                f"Requested bone '{requested_bone}' is not valid. Options are: {list(self._lblmap.keys())}"
            )


class Crop(CropBase):
    def __init__(self, volume, detection_confidence=0.5, detection_iou=0.5):
        super().__init__(volume, detection_confidence, detection_iou)

    def _process(self, box_groups, obb_spacing):
        """Crop the volume based on the detected boxes"""
        return []

    def process(
        self,
        bone: str,
        grouping_iou: float = 0.5,
        grouping_interval: int = 40,
        grouping_min_depth: int = 50,
        obb_spacing: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Process the volume for the requested bone.

        Args:
            bone: The bone to be cropped. Options are: clavicle, scapula, humerus, radius-ulna, hand.
            grouping_iou: IOU threshold for grouping.
            grouping_interval: Interval for grouping.
            grouping_min_depth: Minimum depth in mm for grouping detection boxes together.
            obb_spacing: Spacing for the resampling.

        Returns:
            Aligned images of the oriented bounding boxes.
        """
        self._check_bone_request(bone)
        box_groups = self._uop.objects(
            grouping_iou,
            grouping_interval,
            grouping_min_depth,
        )
        box_groups = box_groups[self._lblmap[bone]]
        return self._process(box_groups, obb_spacing)


class CropOriented(Crop):
    def __init__(self, volume, detection_confidence=0.5, detection_iou=0.5):
        super().__init__(volume, detection_confidence, detection_iou)
        self.interpolator = sitk.sitkBSpline

    def _boolean_volume(self, xyz4):
        """Convert the volume to a boolean mask."""
        # Create boolean image marking box vertices
        # reshape from (n, 4, 3) to (n, 3)
        xyz = xyz4.reshape(-1, 3)
        zyx_bool = np.zeros(np.flip(self.volume.GetSize()))
        zyx_bool[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = 1
        zyx_bool = sitk.GetImageFromArray(zyx_bool)
        zyx_bool.CopyInformation(self.volume)
        zyx_bool = sitk.Cast(zyx_bool, sitk.sitkUInt8)
        return zyx_bool

    def obb_filters(self, box_groups):
        filters = []
        for box_group in box_groups:
            if len(box_group) == 0:
                continue
            # get boolean mask of box vertices
            zyx_bool = self._boolean_volume(box_group)

            obb_filter = sitk.LabelShapeStatisticsImageFilter()
            obb_filter.ComputeOrientedBoundingBoxOn()
            obb_filter.Execute(zyx_bool)

            filters.append(obb_filter)

        return filters

    def _process(self, box_groups, obb_spacing):
        aligned_imgs = []
        # get boolean mask of box vertices
        obb_filters = self.obb_filters(box_groups)
        for obb_filter in obb_filters:
            aligned_img_dir = (
                np.array(obb_filter.GetOrientedBoundingBoxDirection(1))
                .reshape(3, 3)
                .T.flatten()
                .tolist()
            )
            # boot up the resampler
            resampler = sitk.ResampleImageFilter()
            # calculate the size of the aligned image
            aligned_img_size = [
                int(ceil(obb_filter.GetOrientedBoundingBoxSize(1)[i] / obb_spacing[i]))
                for i in range(3)
            ]
            # set origin for the aligned image
            obb_origin = obb_filter.GetOrientedBoundingBoxOrigin(1)

            resampler.SetOutputDirection(aligned_img_dir)
            resampler.SetOutputOrigin(obb_origin)
            resampler.SetOutputSpacing(obb_spacing)
            resampler.SetSize(aligned_img_size)
            resampler.SetInterpolator(self.interpolator)
            resampler.SetDefaultPixelValue(-1023)

            # get the aligned image, and its array
            aligned_img = resampler.Execute(self.volume)

            aligned_imgs.append(aligned_img)
        return aligned_imgs


class Centroids(CropBase):
    def __init__(self, volume):
        super().__init__(volume)

    def _process(self, box_groups):
        """Calculate the centroids of the detected boxes."""
        centroids = []
        for box_group in box_groups:
            if len(box_group) == 0:
                continue
            # Calculate the centroid of each box
            centroid = np.mean(box_group, axis=1)
            centroids.append(centroid)
        return centroids


if __name__ == "__main__":
    volume = sitk.ReadImage("/mnt/slowdata/ct/cadaveric-full-arm/1606011L/1606011L.nrrd")
    cropper = CropOriented(
        volume,
        detection_confidence=0.2,
        detection_iou=0.2,
    )
    output = cropper.process(
        bone="humerus",
        grouping_iou=0.2,
        grouping_interval=50,
        grouping_min_depth=20,
        obb_spacing=(1.0, 1.0, 1.0),
    )
    for i, img in enumerate(output):
        sitk.WriteImage(img, f"aligned_humerus_{i}.nrrd")
