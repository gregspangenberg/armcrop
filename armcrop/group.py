from armcrop.detect import Detector, YOLODetector
from networkx.utils.union_find import UnionFind
import SimpleITK as sitk
import numpy as np


class GroupsDetector:
    def __init__(self, detector: YOLODetector):
        self.detector = detector

    def objects(self, volume, detection_confidence, detection_iou, grouping_iou, grouping_interval):

        detections = self.detector.predict(volume, detection_confidence, detection_iou)
        self._groups(detections, grouping_iou, grouping_interval, volume.GetSpacing()[2])

    def _groups(self, detections, iou, interval, spacing_z):
        """
        Group detected objects based on their spatial proximity.
        """
        # convert the interval in mm to the number of slices
        interval = int(interval / spacing_z)
        print(f"Interval in slices: {interval}")
        # loop over the classes in the detections
        for class_idx in detections:
            # detections are in the format (z, x, y, w, h, rotation, confidence, class_id)
            # ensure array is ordered by z
            sorted_detections = sorted(detections[class_idx], key=lambda z: z[0])
            zs, xywhrs, _ = np.split(sorted_detections, [1, 6], axis=1)

            idxs = np.arange(len(zs))
            matches = {}
            for i in idxs:
                z = zs[i]
                xywhr = xywhrs[i].reshape(1, -1)

                # find nearby zs within the interval
                zs_nearby = ((zs >= z - interval) & (zs <= z + interval)).flatten()
                xywhr_nearby = xywhrs[zs_nearby]

                # calculate IOU with nearby detections
                ious = np.triu(self.detector._batch_probiou(xywhr, xywhr_nearby), 1)
                pick = (ious > iou).flatten()
                matches[i] = np.where(zs_nearby)[0][pick]

            # find overlapping unions
            ds = UnionFind()
            for gp in matches.values():
                ds.union(*gp)
            ds_sets = sorted([sorted(s) for s in ds.to_sets()], key=len, reverse=True)

            print(f"Class {class_idx} detected groups: {len(ds_sets)}")
            for group in ds_sets:
                print(f"detected: {len(group)}")

            # z_arr is a list of detections at a specific z level
            # sliding window to previous zs and next zs


if __name__ == "__main__":
    # Example usage
    detector = YOLODetector(model_type="yolo11-obb")
    groups_detector = GroupsDetector(detector)
    volume = sitk.ReadImage("/mnt/slowdata/ct/cadaveric-full-arm/1606011L/1606011L.nrrd")
    conf_threshold = 0.5
    iou_threshold = 0.5

    groups_detector.objects(
        volume,
        detection_confidence=conf_threshold,
        detection_iou=iou_threshold,
        grouping_iou=0.2,  # Example IOU threshold for grouping
        grouping_interval=50,  # Example interval
    )
