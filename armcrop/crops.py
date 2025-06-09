from armcrop.group import UniqueObjectProcessor
from armcrop.detect import YOLODetector
import abc
import SimpleITK as sitk
from typing import List
import numpy as np
from math import ceil


class CropBase(abc.ABC):
    _detector_cache = {}  # Class-level cache for YOLODetector instances
    _MAX_CACHE_SIZE = 1  # Maximum number of items in the cache

    def __init__(
        self,
        volume: sitk.Image,
        detection_confidence: float = 0.5,
        detection_iou: float = 0.5,
    ):
        """
        Initialize the cropper with a volume.

        Args:
            volume (sitk.Image): The volume to be cropped. Must be a SimpleITK image.
            detection_confidence (float): Confidence threshold for the YOLO detector.
            detection_iou (float): IOU threshold for the YOLO detector.
        """

        self._volume = volume

        # Using id(volume) for the cache key. This assumes that for the "same" volume,
        # the exact same sitk.Image object instance is used.
        # If not, a more robust key based on volume properties would be needed.
        cache_key = (id(self._volume), float(detection_confidence), float(detection_iou))

        if cache_key in CropBase._detector_cache:
            self._det = CropBase._detector_cache[cache_key]
            # print(f"Reusing cached detector for volume ID {id(self.volume)} and params ({detection_confidence}, {detection_iou})") # Optional: for debugging
        else:
            # If the cache is full (i.e., contains _MAX_CACHE_SIZE items)
            # and we are about to add a new, different item, clear the cache first.
            if len(CropBase._detector_cache) >= CropBase._MAX_CACHE_SIZE:
                # print(f"Cache limit ({CropBase._MAX_CACHE_SIZE}) reached. Clearing cache before adding new item.") # Optional: for debugging
                CropBase._detector_cache.clear()

            # print(f"Creating new detector for volume ID {id(self.volume)} and params ({detection_confidence}, {detection_iou})") # Optional: for debugging
            self._det = YOLODetector()
            self._det.predict(self._volume, detection_confidence, detection_iou)
            CropBase._detector_cache[cache_key] = self._det

        self._uop = UniqueObjectProcessor(self._det, self._volume)
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
        self.interpolator = sitk.sitkBSpline

    def _boolean_volume(self, xyz4):
        """Convert the xyz boxes to a boolean mask of all vertices, handles out of bounds errors."""

        # reshape from (n, 4, 3) to (n, 3)
        xyz = xyz4.reshape(-1, 3)

        # clip along x,y,z axes to ensure we don't go out of bounds
        xyz[:, 0] = np.clip(xyz[:, 0], 0, self._volume.GetSize()[0] - 1)
        xyz[:, 1] = np.clip(xyz[:, 1], 0, self._volume.GetSize()[1] - 1)
        xyz[:, 2] = np.clip(xyz[:, 2], 0, self._volume.GetSize()[2] - 1)

        # we are flipping the axes to match the sitk coordinate system
        zyx_bool = np.zeros(np.flip(self._volume.GetSize()))
        zyx_bool[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = 1
        zyx_bool = sitk.GetImageFromArray(zyx_bool)
        zyx_bool.CopyInformation(self._volume)
        zyx_bool = sitk.Cast(zyx_bool, sitk.sitkUInt8)
        return zyx_bool

    def _filters(self, box_groups):
        filters = []
        for box_group in box_groups:
            if len(box_group) == 0:
                continue
            # get boolean mask of box vertices
            zyx_bool = self._boolean_volume(box_group)
            bb_filter = sitk.LabelShapeStatisticsImageFilter()
            bb_filter.Execute(zyx_bool)
            filters.append(bb_filter)

        return filters

    def _process(self, box_groups, spacing):
        cropped_imgs = []
        # get boolean mask of box vertices
        bb_filters = self._filters(box_groups)
        for bb_filter in bb_filters:
            # crop the image using the bounding box
            boundingbox = bb_filter.GetBoundingBox(1)
            roi_filter = sitk.RegionOfInterestImageFilter()
            roi_filter.SetRegionOfInterest(boundingbox)
            crop_img = roi_filter.Execute(self._volume)
            # resample the cropped image to the desired spacing
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(crop_img.GetDirection())
            resampler.SetOutputOrigin(crop_img.GetOrigin())
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(
                [
                    int(ceil(crop_img.GetSize()[i] * crop_img.GetSpacing()[i] / spacing[i]))
                    for i in range(3)
                ]
            )
            resampler.SetInterpolator(self.interpolator)
            resampler.SetDefaultPixelValue(-1023)
            crop_img = resampler.Execute(crop_img)

            cropped_imgs.append(crop_img)
        return cropped_imgs

    def process(
        self,
        bone: str,
        grouping_iou: float = 0.5,
        grouping_interval: int = 40,
        grouping_min_depth: int = 50,
        spacing: tuple = (0.5, 0.5, 0.5),
    ) -> List[sitk.Image]:
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
        return self._process(box_groups, spacing)


class CropOriented(Crop):
    def __init__(self, volume, detection_confidence=0.5, detection_iou=0.5):
        super().__init__(volume, detection_confidence, detection_iou)

    def _filters(self, box_groups):
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

    def _process(self, box_groups, spacing):
        aligned_imgs = []
        # get boolean mask of box vertices
        obb_filters = self._filters(box_groups)
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
                int(ceil(obb_filter.GetOrientedBoundingBoxSize(1)[i] / spacing[i]))
                for i in range(3)
            ]
            # set origin for the aligned image
            obb_origin = obb_filter.GetOrientedBoundingBoxOrigin(1)

            resampler.SetOutputDirection(aligned_img_dir)
            resampler.SetOutputOrigin(obb_origin)
            resampler.SetOutputSpacing(spacing)
            resampler.SetSize(aligned_img_size)
            resampler.SetInterpolator(self.interpolator)
            resampler.SetDefaultPixelValue(-1023)

            # get the aligned image, and its array
            aligned_img = resampler.Execute(self._volume)

            aligned_imgs.append(aligned_img)
        return aligned_imgs


class Centroids(CropBase):
    def __init__(self, volume, detection_confidence=0.5, detection_iou=0.5):
        super().__init__(volume, detection_confidence, detection_iou)

    def _process(self, box_groups) -> List[np.ndarray]:
        """Calculate the centroids of the detected boxes."""
        centroids = []
        for box_group in box_groups:
            if len(box_group) == 0:
                continue
            # Calculate the centroid of each box
            centroid_idx = np.mean(box_group, axis=1).astype(int)
            centroids_obj = []
            for ci in centroid_idx:
                ci_mm = self._volume.TransformIndexToPhysicalPoint(ci.tolist())
                # ci2 = self.volume.TransformPhysicalPointToIndex(ci_mm)
                # print(ci, ci_mm, ci2)
                centroids_obj.append(ci_mm)
            centroids.append(np.array(centroids_obj))
        return centroids

    def process(
        self,
        bone: str,
        grouping_iou: float = 0.5,
        grouping_interval: int = 40,
        grouping_min_depth: int = 50,
    ) -> List[np.ndarray]:
        """
        Process the volume to get centroids of detected boxes.

        Args:
            bone: The bone to be processed. Options are: clavicle, scapula, humerus, radius-ulna, hand.
            grouping_iou: IOU threshold for grouping.
            grouping_interval: Interval for grouping.
            grouping_min_depth: Minimum depth in mm for grouping detection boxes together.

        Returns:
            List of centroids for each detected box.
        """
        self._check_bone_request(bone)
        box_groups = self._uop.objects(
            grouping_iou,
            grouping_interval,
            grouping_min_depth,
        )
        box_groups = box_groups[self._lblmap[bone]]
        return self._process(box_groups)


if __name__ == "__main__":
    volume = sitk.ReadImage("/mnt/slowdata/ct/cadaveric-full-arm/09-12052L/09-12052L.nrrd")
    # Example: Create another reference to the same volume object
    # volume_ref2 = volume
    # If you were to load it again:
    # volume2 = sitk.ReadImage("/mnt/slowdata/ct/cadaveric-full-arm/09-12052L/09-12052L.nrrd")
    # Then id(volume) != id(volume2), and caching based on id() would not work across these two.

    OBB = True
    CENTROID = True  # Run both to test caching
    CROP = True  # Run all three
    OTHER_SETTINGS = False  # Test with different detection parameters
    OTHER_VOLUME = True  # Test with a different volume

    det_conf = 0.2
    det_iou = 0.2

    print(f"Initial cache size: {len(CropBase._detector_cache)}")

    if OBB:
        print("Processing OBB...")
        cropper_obb = CropOriented(
            volume,  # Pass the same volume object
            detection_confidence=det_conf,
            detection_iou=det_iou,
        )
        output_obb = cropper_obb.process(
            bone="scapula",
            grouping_iou=0.2,
            grouping_interval=50,
            grouping_min_depth=20,
            spacing=(1.0, 1.0, 1.0),
        )
        for i, img in enumerate(output_obb):
            sitk.WriteImage(img, f"aligned_{i}.nrrd")
        print(f"Saved {len(output_obb)} OBB images.")
        print(f"Cache size after OBB: {len(CropBase._detector_cache)}")

    if CENTROID:
        print("Processing Centroids...")
        # Using the same volume object and detection parameters
        centroider = Centroids(volume, detection_confidence=det_conf, detection_iou=det_iou)
        output_centroids = centroider.process(
            "humerus",
            grouping_iou=0.2,
            grouping_interval=50,
            grouping_min_depth=20,
        )
        for i, centroid_val in enumerate(output_centroids):
            print(f"Centroid {i}: {centroid_val}")
        print(f"Found {len(output_centroids)} centroids.")
        print(f"Cache size after Centroids: {len(CropBase._detector_cache)}")

    if CROP:
        print("Processing Axis-Aligned Crop...")
        # Using the same volume object and detection parameters
        cropper_axis_aligned = Crop(
            volume,
            detection_confidence=det_conf,
            detection_iou=det_iou,
        )
        output_crop = cropper_axis_aligned.process(
            bone="humerus",
            grouping_iou=0.2,
            grouping_interval=50,
            grouping_min_depth=20,
            spacing=(1.0, 1.0, 1.0),
        )
        for i, img in enumerate(output_crop):
            sitk.WriteImage(img, f"cropped_{i}.nrrd")
        print(f"Saved {len(output_crop)} cropped images.")
        print(f"Cache size after Crop: {len(CropBase._detector_cache)}")

    # Example with different detection parameters - this would create a new cache entry
    if OTHER_SETTINGS:
        print("Processing OBB with different detection parameters...")
        cropper_obb_diff_params = CropOriented(
            volume,  # Same volume object
            detection_confidence=0.7,  # Different confidence
            detection_iou=det_iou,
        )
        # ... process and use cropper_obb_diff_params ...
        print(f"Cache size after OBB with different params: {len(CropBase._detector_cache)}")

        print("Processing Centroids with same different parameters...")
        centroider_diff_params = Centroids(
            volume,  # Same volume object
            detection_confidence=0.7,  # Different confidence
            detection_iou=det_iou,
        )
        output_centroids_diff = centroider_diff_params.process(
            "humerus",
            grouping_iou=0.2,
            grouping_interval=50,
            grouping_min_depth=20,
        )
        for i, centroid_val in enumerate(output_centroids_diff):
            print(f"Centroid {i}: {centroid_val}")
        print(f"Found {len(output_centroids_diff)} centroids with different params.")
        print(f"Cache size after Centroids with different params: {len(CropBase._detector_cache)}")

    if OTHER_VOLUME:
        print("Processing OBB with a different volume...")
        # Load a different volume
        volume2 = sitk.ReadImage("/mnt/slowdata/ct/cadaveric-full-arm/1606011L/1606011L.nrrd")
        cropper_obb_diff_volume = CropOriented(
            volume2,  # Different volume object
            detection_confidence=det_conf,
            detection_iou=det_iou,
        )
        output_obb_diff_volume = cropper_obb_diff_volume.process(
            bone="scapula",
            grouping_iou=0.2,
            grouping_interval=50,
            grouping_min_depth=20,
            spacing=(1.0, 1.0, 1.0),
        )
        for i, img in enumerate(output_obb_diff_volume):
            sitk.WriteImage(img, f"aligned_diff_{i}.nrrd")
        print(f"Saved {len(output_obb_diff_volume)} OBB images for the new volume.")
        print(f"Cache size after OBB with different volume: {len(CropBase._detector_cache)}")
