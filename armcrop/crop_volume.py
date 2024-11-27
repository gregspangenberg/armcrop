import pathlib
import numpy as np
import math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from networkx.utils.union_find import UnionFind
from typing import List
from copy import deepcopy

import SimpleITK as sitk
import onnxruntime as rt
import time
import huggingface_hub


def get_model():
    model_path = huggingface_hub.hf_hub_download(
        repo_id="gregspangenberg/armcrop",
        filename="yolov9c_upperlimb.onnx",
    )
    return model_path


def i_within_x_mm(target_i, mm, spacing, start_i, stop_i):
    """
    This function finds the k closest numbers to a target number within a specified range.

    Args:
        target: The number to find the closest neighbors for.
        range_min: The minimum value of the range (inclusive). Defaults to 0.
        range_max: The maximum value of the range (exclusive). Defaults to 100.
        k: The number of closest neighbors to find. Defaults to 10.

    Returns:
        A list of the k closest numbers to the target within the specified range.
    """

    k = int(np.ceil(mm / spacing))
    # Generate a list of numbers within the range (inclusive)
    numbers = range(start_i, stop_i)

    # Calculate the absolute difference between each number and the target
    differences = [abs(num - target_i) for num in numbers]

    # Sort together by difference (ascending) and then by number (ascending)
    sorted_data = sorted(zip(differences, numbers))

    # Extract the k closest numbers based on the difference
    closest_k = sorted_data[1 : k + 1]

    # Return only the numbers (remove the difference)
    return np.array(sorted([num for _, num in closest_k]))


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    # print(f'box{box} boxes: {boxes}')
    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def extract_boxes(predictions):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    # boxes = rescale_boxes(boxes)

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    return boxes


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def unique_encompassing_boxes(boxes, scores, class_ids, indices, iou_threshold=0.1):
    # iterate over the nms boxes
    encompassing_boxes = []
    for b_nms, s_nms, c_nms in zip(boxes[indices], scores[indices], class_ids[indices]):
        # compute iou for the nms box with all other boxes in class
        boxes_class = boxes[np.where(class_ids == c_nms)[0], :]
        ious = compute_iou(b_nms, boxes_class)
        iou_indicies = np.where(ious > iou_threshold)[0]
        boxes_intersect = boxes_class[iou_indicies, :]
        # construct biggest box that encompasses all intersecting boxes
        x1 = np.min(boxes_intersect[:, 0])
        y1 = np.min(boxes_intersect[:, 1])
        x2 = np.max(boxes_intersect[:, 2])
        y2 = np.max(boxes_intersect[:, 3])
        encompassing_boxes.append(np.array([x1, y1, x2, y2]))

    encompassing_boxes = np.vstack(encompassing_boxes)
    return encompassing_boxes, scores[indices], class_ids[indices]


def post_process_image(output, conf_threshold=0.5, iou_threshold=0.5):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshol
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)

    boxes, scores, class_ids = unique_encompassing_boxes(boxes, scores, class_ids, indices)

    return boxes, scores, class_ids


def load_volume(volume_path: pathlib.Path, img_size=(640, 640)):
    vol = sitk.ReadImage(str(volume_path))
    vol_t = deepcopy(vol)
    # x and y axes were flipped during training so we need to permute the axes
    # in the future we will fix this in the training pipeline
    vol_t = sitk.PermuteAxes(vol_t, [1, 0, 2])
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


def load_model(img_size):
    # load model
    with open(get_model(), "rb") as file:
        # set order of providers to try
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
        model = rt.InferenceSession(file.read(), providers=providers)

    # prime the model on a random image, to get the slow first inference out of the way
    model.run(
        None,
        {"images": np.random.rand(1, 3, img_size[0], img_size[1]).astype(np.float32)},
    )

    return model


def load(volume_path, img_size):
    with ThreadPoolExecutor() as executor:
        # Apply the tasks asynchronously
        volume_result = executor.submit(load_volume, volume_path, img_size)
        model_result = executor.submit(load_model, img_size)

        # Wait for results
        vol, vol_t = volume_result.result()
        model = model_result.result()
    return model, vol, vol_t


def predict(
    volume_path,
    z_padding,
    xy_padding,
    max_gap,
    iou_threshold,
    discard_threshold,
):
    # load model and volume
    img_size = (640, 640)
    model, vol, vol_t = load(volume_path, img_size)

    # loop over axial images and predict
    data = []
    for i_img in range(vol_t.GetDepth()):
        # prepare the image for inference
        arr = sitk.GetArrayFromImage(vol_t[:, :, i_img])
        arr = np.expand_dims(arr, axis=0)
        arr = np.expand_dims(arr, axis=0)
        arr = np.repeat(arr, 3, axis=1)
        arr = arr.astype(np.float32)

        # run inference on the image
        output = model.run(None, {"images": arr})
        boxes, scores, labels = post_process_image(output, conf_threshold=0.4, iou_threshold=0.3)
        # record the data
        data.extend(list(zip(np.repeat(i_img, len(labels)), boxes, scores, labels)))

    # get the dict of crop of the bounding volume, from the predicted boxes on each axial slice
    crop_dict = post_process_volume(
        data,
        vol,
        img_size,
        z_padding,
        xy_padding,
        max_gap,
        iou_threshold,
        discard_threshold,
    )

    return vol, crop_dict


def post_process_volume(
    data,
    vol: sitk.Image,
    img_size,
    z_padding,
    xy_padding,
    max_gap,
    iou_threshold,
    discard_threshold,
):
    names = {
        0: "clavicle",
        1: "scapula",
        2: "humerus",
        3: "radius_ulna",
        4: "hand",
    }

    Z_PADDING = math.ceil(z_padding / vol.GetSpacing()[-1])  #  extra space on each end
    XY_PADDING = math.ceil(xy_padding / vol.GetSpacing()[-2])  # space on edges
    MAX_Z_GAP = math.ceil(max_gap / vol.GetSpacing()[-1])  # new object after gap

    # df: 0: slice_i, 1: box, 2: score, 3: class
    df = pd.DataFrame(data)
    crop_classes = {value: [] for value in names.keys()}
    # iterate over classes
    for i in df[3].unique():

        # get all boxes of the same class
        cdf = df[df[3] == i]

        cdf_max_i_img = cdf[0].max()
        cdf_min_i_img = cdf[0].min()

        # get max distance between boxes to be considered the same object

        # compute intersections
        groups = []
        for _, row in cdf.iterrows():
            i_img = int(row[0])
            # find the number of slices adjacent to current slice
            near_i_img = np.arange(i_img - MAX_Z_GAP, i_img + MAX_Z_GAP + 1)
            near_i_img = near_i_img[(near_i_img >= cdf_min_i_img) & (near_i_img <= cdf_max_i_img)]

            # get the boxes of the adjacent slices
            c_boxes = np.vstack(cdf[1][cdf[0].isin(near_i_img)].values)

            # find the boxes that intersect with the current box
            intersected = compute_iou(row[1], c_boxes) > iou_threshold
            # get the index of the intersected boxes
            index_intersect = cdf[cdf[0].isin(near_i_img)].index[intersected]
            groups.append(list(index_intersect))

        # find overlapping unions
        ds = UnionFind()
        for gp in groups:
            ds.union(*gp)

        # sort through unions and find crop diemnsions
        ds_sets = sorted([sorted(s) for s in ds.to_sets()], key=len, reverse=True)
        for s in ds_sets:
            # if set is smaller than 10 mm discard
            if len(s) < (discard_threshold / float(vol.GetSpacing()[-1])):
                continue

            dff = df.loc[s]
            xmin, ymin, xmax, ymax = (
                np.vstack(dff[1].values) / img_size[0] * float(vol.GetWidth())
            ).T
            # get the bounds of the region of interest
            bounds = [
                [
                    math.floor(xmin.min()) - XY_PADDING,
                    math.ceil(xmax.max()) + XY_PADDING,
                ],
                [
                    math.floor(ymin.min()) - XY_PADDING,
                    math.ceil(ymax.max()) + XY_PADDING,
                ],
                [
                    math.floor(dff[0].min()) - Z_PADDING,
                    math.ceil(dff[0].max()) + Z_PADDING,
                ],
            ]
            # clip the bounds to the volume size
            clip_intervals = [
                (0, vol.GetSize()[0]),
                (0, vol.GetSize()[1]),
                (0, vol.GetSize()[2]),
            ]
            bounds = [np.clip(b, *ci) for b, ci in zip(bounds, clip_intervals)]
            bounds = np.array(bounds).flatten().tolist()

            # get center of the region of interest and the size
            roi_center = [bounds[0], bounds[2], bounds[4]]
            roi_size = [
                math.floor((bounds[1] - bounds[0])),
                math.floor((bounds[3] - bounds[2])),
                math.floor((bounds[5] - bounds[4])),
            ]
            # swap x any axes due to issue in training pipeline
            # will fix in the future
            roi_center = [roi_center[1], roi_center[0], roi_center[2]]
            roi_size = [roi_size[1], roi_size[0], roi_size[2]]

            # append to class crop
            crop_classes[i].append([roi_size, roi_center])

    crop_classes = {names.get(old_key, old_key): value for old_key, value in crop_classes.items()}

    return crop_classes


def save_volume(vol, path):
    sitk.WriteImage(vol, path, imageIO="NrrdImageIO")


class Crop2Bone:
    """
    Crops a bounding box volume to the bone of interest.

    Attributes:
        z_padding (int): Padding in the z-dimension.
        xy_padding (int): Padding in the x and y dimensions.
        max_gap (int): Maximum gap allowed.
        iou_threshold (float): Intersection over Union threshold.
        discard_threshold (int): Threshold for discarding small regions.

    Methods:
        __call__(volume_path):
            Predicts and crops the volume based on the provided parameters.

        clavicle() -> List[sitk.Image]:
            Returns the cropped volume for the clavicle.

        scapula() -> List[sitk.Image]:
            Returns the cropped volume for the scapula.

        humerus() -> List[sitk.Image]:
            Returns the cropped volume for the humerus.

        radius_ulna() -> List[sitk.Image]:
            Returns the cropped volume for the radius and ulna.

        hand() -> List[sitk.Image]:
            Returns the cropped volume for the hand.
    """

    def __init__(
        self,
        z_padding=2,
        xy_padding=2,
        max_gap=15,
        iou_threshold=0.1,
        discard_threshold=30,
    ):
        self.z_padding = z_padding
        self.xy_padding = xy_padding
        self.max_gap = max_gap
        self.iou_threshold = iou_threshold
        self.discard_threshold = discard_threshold

    def __call__(self, volume_path):
        self.vol, self._crop_dict = predict(
            volume_path,
            self.z_padding,
            self.xy_padding,
            self.max_gap,
            self.iou_threshold,
            self.discard_threshold,
        )
        return self

    def _croppa(self, roi) -> list:
        crop_vols = []
        for r in roi:
            vol_crop = sitk.RegionOfInterest(self.vol, r[0], r[1])
            crop_vols.append(vol_crop)
        return crop_vols


def clavicle(self) -> List[sitk.Image]:
    """
    Extracts and crops the clavicle region from the CT volume.

    Returns:
        List[sitk.Image]: List of cropped volumes containing detected clavicles
    """
    return self._croppa(self._crop_dict["clavicle"])


def scapula(self) -> List[sitk.Image]:
    """
    Extracts and crops the scapula region from the CT volume.

    Returns:
        List[sitk.Image]: List of cropped volumes containing detected scapulae
    """
    return self._croppa(self._crop_dict["scapula"])


def humerus(self) -> List[sitk.Image]:
    """
    Extracts and crops the humerus region from the CT volume.

    Returns:
        List[sitk.Image]: List of cropped volumes containing detected humeri
    """
    return self._croppa(self._crop_dict["humerus"])


def radius_ulna(self) -> List[sitk.Image]:
    """
    Extracts and crops the radius and ulna regions from the CT volume.

    Returns:
        List[sitk.Image]: List of cropped volumes containing detected radius/ulna pairs
    """
    return self._croppa(self._crop_dict["radius_ulna"])


def hand(self) -> List[sitk.Image]:
    """
    Extracts and crops the hand region from the CT volume.

    Returns:
        List[sitk.Image]: List of cropped volumes containing detected hands
    """
    return self._croppa(self._crop_dict["hand"])


if __name__ == "__main__":
    t0 = time.time()

    volume_path = pathlib.Path("/mnt/slowdata/cadaveric-full-arm/S202048/S202048.nrrd")

    croppa = Crop2Bone()
    croppa(volume_path)
    print(croppa._crop_dict)
    print(sitk.ReadImage(volume_path).GetSize())
    for i in range(len(croppa.scapula())):
        s = croppa.scapula()[i]
        print(croppa.scapula()[i].GetSize())
        save_volume(s, f"S202048-{i}.nrrd")
    print(f"Elapsed time: {time.time()-t0}")
