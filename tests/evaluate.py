"""
You will need pytorch and pytorch3d installed to run this testing script. Please refer to the INSTALL.md file in pytorch3d
for further instructions on how to install pytorch3d.
Additionally you will need your own dataset of segmentations setup in the format described below.

 The ground truth segmentations are in the format shown in SEGMENTATION_LABELS.
 When there are multiple instance of the same label, the ground truth segmentations have to be seperated (i.e _left and _right )
 into seperate .seg.nrrd files. This is necessary to avoid  sitk creating a bounding box around both bones instead of around
 each like the obb-ml model does.

 The output of the obb-ml model will be in the format shown in MODEL_LABELS, since there is no use predicting the background label.
"""

from pathlib import Path
from typing import Dict, Set, List
from copy import deepcopy
import pandas as pd
import SimpleITK as sitk
import numpy as np
import armcrop

import torch
from pytorch3d.ops.iou_box3d import box3d_overlap

# Label mapping constants
SEGMENTATION_LABELS = {
    0: "background",
    1: "clavicle",
    2: "scapula",
    3: "humerus",
    4: "radius_ulna",
    5: "hand",
}

MODEL_LABELS = {0: "clavicle", 1: "scapula", 2: "humerus", 3: "radius_ulna", 4: "hand"}


def get_validation_subjects(val_path: Path) -> Set[str]:
    """Get set of subject IDs from validation dataset"""
    return {s.stem.split("-")[0] for s in val_path.glob("*.txt")}


def get_ct_path(subject_id: str, ct_dirs: List[Path]) -> Path:
    """Get path to CT for subject ID"""
    # list all nrrd files in the directories
    cts_all = []
    for c in ct_dirs:
        cts_all.extend(c.rglob("*.nrrd"))
    # remove segmentation files
    cts_all = [c for c in cts_all if ".seg.nrrd" not in c.name]
    # find the CT for the subject
    ct_path = [c for c in cts_all if subject_id in c.stem]
    # warn user if there are multiple CTs for the subject_id
    if len(ct_path) != 1:
        raise ValueError(f"Found {len(ct_path)} CTs for subject {subject_id}")
    return ct_path[0]


def count_bone_instances(obb_filter) -> Dict[str, int]:
    """Count instances of each bone type in segmentation"""

    # Initialize counts
    bone_counts = {label: 0 for label in MODEL_LABELS.values()}

    # Count instances
    for label in obb_filter.GetLabels():
        if label == 0:  # Skip background
            continue
        bone_type = MODEL_LABELS[label - 1]
        bone_counts[bone_type] += 1

    return bone_counts


def format_pytorch3d_boxes(obb_filter, label):
    # get the 8 points of the oriented bounding box
    obb_points = np.array(obb_filter.GetOrientedBoundingBoxVertices(label)).reshape(8, 3)
    trn = np.array(obb_filter.GetOrientedBoundingBoxDirection(label)).reshape(3, 3).T

    # box3d overlap needs the 8 points of the box in the following format:
    n, x = np.min, np.max
    p = obb_points @ trn
    p = np.array(
        [
            [n(p[:, 0]), n(p[:, 1]), n(p[:, 2])],
            [x(p[:, 0]), n(p[:, 1]), n(p[:, 2])],
            [x(p[:, 0]), x(p[:, 1]), n(p[:, 2])],
            [n(p[:, 0]), x(p[:, 1]), n(p[:, 2])],
            [n(p[:, 0]), n(p[:, 1]), x(p[:, 2])],
            [x(p[:, 0]), n(p[:, 1]), x(p[:, 2])],
            [x(p[:, 0]), x(p[:, 1]), x(p[:, 2])],
            [n(p[:, 0]), x(p[:, 1]), x(p[:, 2])],
        ]
    )
    p = p @ np.linalg.inv(trn)

    box = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

    return box


def main():
    # Input paths
    seg_dir = Path("/home/greg/projects/segment/stage1_yolo_training/database/seg")
    val_subj_dir = Path(
        "/home/greg/projects/segment/stage1_yolo_training/datasets/dataset_balanced_obb/labels/val"
    )
    ct_dirs = [
        Path("/mnt/slowdata/arthritic-clinical-half-arm"),
        Path("/mnt/slowdata/cadaveric-full-arm"),
    ]
    val_results = {0: [], 1: [], 2: [], 3: [], 4: []}

    # Get validation subjects
    val_subjects = get_validation_subjects(val_subj_dir)

    # initialize the obb_cropper
    obb_cropper = armcrop.OBBCrop2Bone(
        z_padding=0, xy_padding=0, iou_threshold=0.1, z_iou_interval=50, z_length_min=20
    )
    # Process each segmentation
    subjects = []
    for seg_path in sorted(seg_dir.glob("*.seg.nrrd")):
        subject_id = seg_path.stem.split(".")[0].split("_")[0]

        if subject_id not in val_subjects:
            continue
        subjects.append(seg_path.stem.split(".")[0])
        # get matching CT
        ct_path = get_ct_path(subject_id, ct_dirs)

        # Load and analyze segmentation for number of instances of each bone
        label_image = sitk.ReadImage(str(seg_path))
        obb_filter = sitk.LabelShapeStatisticsImageFilter()
        obb_filter.ComputeOrientedBoundingBoxOn()
        obb_filter.Execute(label_image)
        bone_counts = count_bone_instances(obb_filter)

        print(f"Processing {seg_path.name}")
        print(bone_counts)

        # get the predictions from the model
        obb_cropper = obb_cropper(ct_path)

        for i in range(5):
            if i + 1 not in obb_filter.GetLabels():
                val_results[i].append([None, None, None])
                continue
            else:
                lbl = i + 1
            print(lbl - 1)

            # get the 8 points of the oriented bounding box
            true_obb = format_pytorch3d_boxes(obb_filter, lbl)
            center_true = np.array(obb_filter.GetOrientedBoundingBoxOrigin(lbl))
            vol_true = np.prod(obb_filter.GetOrientedBoundingBoxSize(lbl))

            # get the prediction from the model
            obb_cropper._align(lbl - 1, [0.5, 0.5, 0.5])
            pred_obb = []
            vol_pred = []
            center_pred_error = []
            for _pred_obb in obb_cropper._obb_filters:
                pred_obb.append(format_pytorch3d_boxes(_pred_obb, 1))
                vol_pred.append(np.prod(_pred_obb.GetOrientedBoundingBoxSize(1)))
                center_pred_error.append(
                    np.array(_pred_obb.GetOrientedBoundingBoxOrigin(1)) - center_true
                )
                print(_pred_obb.GetOrientedBoundingBoxOrigin(1), center_true)
            closest_idx = np.argmin(
                [np.sqrt(np.sum((err**2), axis=0)) for err in center_pred_error]
            )
            pred_obb = torch.cat(pred_obb, dim=0)

            # calculate the iou
            vol_overlap, _ = box3d_overlap(
                pred_obb,
                true_obb,
            )
            # keep iou for the correct side, and volumes
            vol_overlap = np.max(vol_overlap.flatten().numpy(), axis=0)
            vol_pred = vol_pred[closest_idx]
            center_pred_error = center_pred_error[closest_idx]

            print(vol_overlap / vol_true, vol_overlap, vol_true)
            print(center_pred_error)

            val_results[lbl - 1].append(
                [
                    vol_overlap / vol_true,
                    np.sqrt(np.sum(center_pred_error**2, axis=0)),
                    center_pred_error[-1],
                ]
            )
            print("\n\n")

    df = pd.DataFrame(val_results)
    df["subjects"] = subjects
    df.to_csv("data/val_results.csv")

    print(val_results)


if __name__ == "__main__":
    main()
