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
import h5py
from scipy.optimize import minimize, differential_evolution

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


def save_dict_arrays(dictionary: dict, filepath: Path):
    """Save dictionary of numpy arrays to HDF5"""
    new_dict = {k: v for k, v in dictionary.items() if v is not None}

    with h5py.File(filepath, "w") as f:
        # Create a group for arrays
        for key, array in new_dict.items():
            # Convert int key to string (HDF5 requirement)
            f.create_dataset(str(key), data=array, compression="gzip")


def load_dict_arrays(filepath: Path) -> dict:
    """Load dictionary of numpy arrays from HDF5"""
    loaded_dict = {}
    with h5py.File(filepath, "r") as f:
        # Load each array and convert key back to int
        for key in f.keys():
            loaded_dict[int(key)] = f[key][:]
    # add None values for missing keys to match the original dictionary
    for key in range(5):
        if key not in loaded_dict:
            loaded_dict[key] = None
    return loaded_dict


def optimization_loop(croppers, lbl, iou_threshold, z_iou_interval):
    goal = []
    for c in croppers:
        obb_cropper, obb_true, center_true, vol_true = c

        obb_pred = []
        vol_pred = []
        center_pred_error = []
        # get the prediction from the model
        for p_obb in obb_cropper._obb(
            lbl,
            iou_threshold,
            z_iou_interval,
            20,
        ):
            obb_pred.append(format_pytorch3d_boxes(p_obb, 1))
            vol_pred.append(np.prod(p_obb.GetOrientedBoundingBoxSize(1)))
            center_pred_error.append(np.array(p_obb.GetOrientedBoundingBoxOrigin(1)) - center_true)

        # select the predicition for the correct label
        closest_idx = np.argmin([np.sqrt(np.sum((err**2), axis=0)) for err in center_pred_error])
        obb_pred = torch.cat(obb_pred, dim=0)
        vol_pred = vol_pred[closest_idx]
        center_pred_error = center_pred_error[closest_idx]

        # calculate the iou
        vol_overlap, iou = box3d_overlap(
            obb_pred,
            obb_true,
        )

        # calculate the goal
        vol_overlap = np.max(vol_overlap.flatten().numpy(), axis=0)
        # goal.append(vol_overlap / vol_true)
        goal.append(float(np.max(iou.numpy())))

    return np.mean(goal, axis=0)


def objective_function(params, croppers, lbl):
    """
    Negative goal function for minimization
    (we minimize negative since we want to maximize goal)
    """
    a, b = params
    print(f"Optimizing with parameters: {a}, {b}")
    goal = optimization_loop(
        croppers,
        lbl,
        iou_threshold=float(a),
        z_iou_interval=int(b),
    )
    print(f"Goal: {goal}")
    return -1 * goal  # Negative since we want to maximize


def optimize_parameters(croppers, lbl):
    # Initial guess
    x0 = np.array([0.1, 50])  # Starting values for a, b,

    # Parameter bounds
    bounds = [
        (0.1, 0.6),  # iou_threshold bounds
        (10, 100),  # z_iou_interval bounds
    ]

    # Run optimization
    # result = minimize(
    #     objective_function,
    #     x0,
    #     args=(croppers, lbl),
    #     bounds=bounds,
    #     method="Nelder-Mead",
    # )
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(croppers, lbl),
    )

    return {
        "iou_threshold": result.x[0],
        "z_iou_interval": int(result.x[1]),
        "optimal_goal": -result.fun,  # Convert back to positive
    }


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
    # select label to optimize
    lbl = 1

    # Get validation subjects
    val_subjects = get_validation_subjects(val_subj_dir)

    croppers = []
    for seg_path in sorted(seg_dir.glob("*.seg.nrrd")):
        subject_id = seg_path.stem.split(".")[0].split("_")[0]

        if subject_id not in val_subjects:
            continue
        # get matching CT
        ct_path = get_ct_path(subject_id, ct_dirs)

        # load from cache if available
        cache_file = Path(f"tests/cache/{subject_id}.h5")
        if cache_file.exists():
            _class_dict = load_dict_arrays(cache_file)
            obb_cropper = armcrop.OBBCrop2Bone(ct_path, debug=True)
            obb_cropper._class_dict = _class_dict
            print(f"Loaded {subject_id}")
        # get the predictions from the model if not cached
        else:
            obb_cropper = armcrop.OBBCrop2Bone(ct_path, debug=False)
            # cache the predictions for the model
            save_dict_arrays(obb_cropper._class_dict, cache_file)
            print(f"Cached {subject_id}")

        # Load and analyze segmentation for number of instances of each bone
        label_image = sitk.ReadImage(str(seg_path))
        obb_filter = sitk.LabelShapeStatisticsImageFilter()
        obb_filter.ComputeOrientedBoundingBoxOn()
        obb_filter.Execute(label_image)

        # record data needed for optimization
        if lbl + 1 in obb_filter.GetLabels():
            obb_true = format_pytorch3d_boxes(obb_filter, lbl + 1)
            center_true = np.array(obb_filter.GetOrientedBoundingBoxOrigin(lbl + 1))
            vol_true = np.prod(obb_filter.GetOrientedBoundingBoxSize(lbl + 1))
            croppers.append((obb_cropper, obb_true, center_true, vol_true))

    # optimize
    optimal_params = optimize_parameters(croppers, lbl)
    print(f"Optimized parameters for {MODEL_LABELS[lbl]}:")
    print(optimal_params)


if __name__ == "__main__":
    main()
