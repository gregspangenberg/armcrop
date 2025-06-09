from armcrop.detect import Detector, YOLODetector
from networkx.utils.union_find import UnionFind
import SimpleITK as sitk
import numpy as np
from collections import defaultdict


class UniqueObjectProcessor:
    def __init__(self, detector: YOLODetector, volume: sitk.Image):
        self.detector = detector
        self.volume_z_spacing = volume.GetSpacing()[2]

    def objects(
        self,
        grouping_iou,
        grouping_interval,
        grouping_min_depth,
    ):

        detections = self.detector.results
        if detections == {}:
            raise ValueError("No detections found. Please run the detector first.")
        box_groups = self._groups(
            detections, grouping_iou, grouping_interval, self.volume_z_spacing, grouping_min_depth
        )
        return box_groups

    def _groups(self, detections, iou, interval, spacing_z, discard_zlength):
        """
        Group detected objects based on their spatial proximity.
        """
        # convert the interval in mm to the number of slices
        interval = int(interval / spacing_z)
        discard_length = int(discard_zlength / spacing_z)
        # loop over the classes in the detections
        groups_xyz4 = defaultdict(list)
        for class_idx in detections:
            # detections are in the format (z, x, y, w, h, rotation, confidence, class_id)
            # ensure array is ordered by z
            if detections[class_idx] is None:
                continue
            # print(f"Processing class {class_idx} with {len(detections[class_idx])} detections")
            sorted_detections = sorted(detections[class_idx], key=lambda z: z[0])
            zs, xywhrs, _ = np.split(sorted_detections, [1, 6], axis=1)

            idxs = np.arange(zs.shape[0])
            matches = {}
            xyz4s = []
            for i in idxs:
                # get the z, xywhr for the current detection
                z = zs[i]
                xywhr = xywhrs[i].reshape(1, -1)
                # convert to xyz4 format
                xyxyxyxy = self._xywhr2xyxyxyxy(xywhr, round=True, pad=0)
                # add in z coordinate
                xyzxyzxyzxyz = np.concatenate([xyxyxyxy, np.full((1, 4, 1), z)], axis=-1)
                xyz4s.append(xyzxyzxyzxyz)

                # find nearby zs within the interval
                z_nearby_mask = ((zs >= z - interval) & (zs <= z + interval)).flatten()
                nearby_indices = np.where(z_nearby_mask)[
                    0
                ]  # Get original indices where mask is True
                xywhr_nearby = xywhrs[z_nearby_mask]

                # calculate IOU with nearby detections
                ious = np.triu(self.detector._batch_probiou(xywhr, xywhr_nearby), 1)
                pick = (ious > iou).flatten()

                # Store the original indices that match
                if len(pick) > 0 and np.any(pick):
                    matches[i] = nearby_indices[pick]
                else:
                    matches[i] = np.array([], dtype=int)

            # Concatenate all xyz4s
            xyz4s = np.concatenate(xyz4s, axis=0)

            # find overlapping unions
            ds = UnionFind()
            # Initialize all nodes in the UnionFind
            for i in idxs:
                ds[i]  # This ensures all indices are present in the UnionFind

            # Now create the unions based on matches
            for i, match_indices in matches.items():
                for j in match_indices:
                    if i != j:  # Don't union a node with itself
                        ds.union(i, j)

            # Get the sets from the UnionFind
            ds_sets = sorted([sorted(s) for s in ds.to_sets() if len(s) > 0], key=len, reverse=True)

            for group in ds_sets:
                if len(group) < discard_length:
                    continue
                groups_xyz4[class_idx].append(xyz4s[group, :].astype(int))

        return groups_xyz4

    def _xywhr2xyxyxyxy(self, x, round=True, pad=0):
        """
        Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4].
        Where xy1 is the top-left corner, xy2 is the top-right corner, xy3 is the bottom-right corner, and xy4 is the bottom-left corner.

        Args:
            x (numpy.ndarray): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

        Returns:
            (numpy.ndarray): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
        """

        def round2pminf_add_remainder(x, r):
            return np.copysign(np.ceil(np.abs(x) + r), x)

        def regularize_rboxes(rboxes):
            """
            Regularize rotated boxes in range [0, pi/2].

            Args:
                rboxes (numpy.ndarray): Input boxes of shape(N, 5) in xywhr format.

            Returns:
                (numpy.ndarray): The regularized boxes.
            """
            x, y, w, h, t = np.split(rboxes, 5, axis=-1)
            w_ = np.where(w > h, w, h)
            h_ = np.where(w > h, h, w)
            t = np.where(w > h, t, t + np.pi / 2) % np.pi
            return np.concatenate([x, y, w_, h_, t], axis=-1)  # regularized boxes

        # Regularize the input boxes first
        rboxes = regularize_rboxes(x)

        ctr = rboxes[..., :2]
        w, h, angle = (rboxes[..., i : i + 1] for i in range(2, 5))

        # add padding to the width and height
        w += pad
        h += pad

        cos_value = np.cos(angle)
        sin_value = np.sin(angle)

        vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
        vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)

        # round to the nearest pixel
        if round:
            rem = np.remainder(ctr, np.floor(ctr))
            ctr = np.floor(ctr)
            pt1 = ctr + round2pminf_add_remainder(vec1 + vec2, rem)
            pt2 = ctr + round2pminf_add_remainder(vec1 - vec2, rem)
            pt3 = ctr - round2pminf_add_remainder(vec1 - vec2, rem)
            pt4 = ctr - round2pminf_add_remainder(vec1 + vec2, rem)
        else:
            pt1 = ctr + vec1 + vec2
            pt2 = ctr + vec1 - vec2
            pt3 = ctr - vec1 - vec2
            pt4 = ctr - vec1 + vec2

        return np.stack([pt1, pt2, pt3, pt4], axis=-2)


if __name__ == "__main__":
    # load a volume
    volume = sitk.ReadImage("/mnt/slowdata/ct/cadaveric-full-arm/1606011L/1606011L.nrrd")
    # Example usage
    detector = YOLODetector(model_type="yolo11-obb")

    detector.predict(
        volume,
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )
    groups_detector = UniqueObjectProcessor(detector, volume)

    groups_detector.objects(
        grouping_iou=0.2,  # Example IOU threshold for grouping
        grouping_interval=50,  # Example interval
        grouping_min_depth=20,  # Example minimum depth for grouping
    )
