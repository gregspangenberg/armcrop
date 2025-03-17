import onnxruntime as rt
import SimpleITK as sitk
import numpy as np
import pathlib
from typing import Tuple, List, Dict
from math import ceil
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from networkx.utils.union_find import UnionFind
import huggingface_hub

# disable logging
rt.set_default_logger_severity(3)


def get_model() -> str:
    """
    Download the ML model from hugginface for inference

    Returns:
        model_path: Path to the ML model
    """
    model_path = huggingface_hub.hf_hub_download(
        repo_id="gregspangenberg/armcrop",
        filename="upperarm_yolo11_obb_6.onnx",
    )
    return model_path


def load_model(img_size) -> rt.InferenceSession:
    """
    Load the ML model for inference

    Args:
        img_size: Image size required for the ML model

    Returns:
        model: The ML model for inference
    """
    # load model
    with open(get_model(), "rb") as file:
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
    volume: pathlib.Path | sitk.Image, img_size=(640, 640)
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


def load(volume, img_size) -> Tuple[rt.InferenceSession, sitk.Image, sitk.Image]:
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
        volume_result = executor.submit(load_volume, volume, img_size)
        model_result = executor.submit(load_model, img_size)

        # Wait for results
        vol, vol_t = volume_result.result()
        model = model_result.result()
    return model, vol, vol_t


def non_max_suppression_rotated(prediction, conf_thres, iou_thres) -> List[np.ndarray]:
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (np.ndarray): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
    Returns:
        (List[np.ndarray]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    bs = prediction.shape[0]  # batch size (BCN,. Defaults to 1,84,6300)
    nc = 5  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres
    prediction = prediction.transpose((0, 2, 1))  # shape(1,84,6300) to shape(1,6300,84)

    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box = x[:, :4]  # xyxy
        cls = x[:, 4:mi]  # classes
        rot = x[:, mi : mi + 1]  # rotation
        conf = np.max(cls, axis=1, keepdims=True)
        class_id = np.argmax(cls, axis=1, keepdims=True)

        x = np.concatenate((box, rot, conf, class_id), 1)[conf.squeeze(1) > conf_thres]

        if not x.shape[0]:  # no boxes
            continue

        # Batched NMS
        boxes = x[:, :5]  # xywhr
        scores = x[:, 5]  # scores

        i = nms_rotated(boxes, scores, iou_thres)
        output[xi] = x[i]

    return output


def nms_rotated(boxes, scores, threshold=0.45) -> np.ndarray:
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (np.ndarray): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (np.ndarray): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.

    Returns:
        (np.ndarray): Indices of boxes to keep after NMS.
    """

    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)

    # Sort boxes by confidence scores in descending order
    sorted_idx = np.argsort(scores, axis=-1)[::-1]
    boxes = boxes[sorted_idx]

    ious = np.triu(batch_probiou(boxes, boxes), 1)
    # Filter boxes based on IOU threshold to remove overlapping detections
    pick = np.nonzero(ious.max(axis=0) < threshold)[0]

    return sorted_idx[pick]


def nms_rotated_conf_only(boxes, scores, threshold=0.45) -> np.ndarray:
    """
    NMS for oriented bounding boxes using probiou and fast-nms.

    Args:
        boxes (np.ndarray): Rotated bounding boxes, shape (N, 5), format xywhr.
        scores (np.ndarray): Confidence scores, shape (N,).
        threshold (float, optional): IoU threshold. Defaults to 0.45.

    Returns:
        (np.ndarray): Indices of boxes to keep after NMS.
    """

    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)

    # Sort boxes by confidence scores in descending order
    sorted_idx = np.argmax(scores, axis=-1)

    return sorted_idx


def batch_probiou(obb1, obb2, eps=1e-7) -> np.ndarray:
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A tensor of shape (N, M) representing obb similarities.
    """

    # Split coordinates into separate x,y components for both sets of boxes
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))

    # Calculate covariance matrices for both sets of boxes
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    # Calculate first term of Bhattacharyya distance using quadratic forms
    t1 = (
        ((a1 + a2) * np.power((y1 - y2), 2) + (b1 + b2) * np.power((x1 - x2), 2))
        / ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2) + eps)
    ) * 0.25

    # Calculate second term (cross term) of Bhattacharyya distance
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2) + eps)
    ) * 0.5

    # Calculate third term (determinant term) of Bhattacharyya distance
    t3 = (
        np.log(
            ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2))
            / (
                4
                * np.sqrt(
                    (a1 * b1 - (np.power(c1, 2)).clip(0)) * (a2 * b2 - (np.power(c2, 2)).clip(0))
                )
                + eps
            )
            + eps
        )
        * 0.5
    )
    # Convert Bhattacharyya distance to Hellinger distance and then to IOU
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1 - np.exp(-bd) + eps)

    return 1 - hd


def _get_covariance_matrix(boxes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (np.ndarray): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (np.ndarray): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    # gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:5]), axis=-1)
    a, b, c = np.split(gbbs, 3, axis=-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = np.power(cos, 2)
    sin2 = np.power(sin, 2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def round2pminf_add_remainder(x, r):
    return np.copysign(np.ceil(np.abs(x) + r), x)


def round2pminf(x):
    return np.copysign(np.ceil(np.abs(x)), x)


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


def xywhr2xyxyxyxy(x, round=True):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4].

    Args:
        x (numpy.ndarray): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    # Regularize the input boxes first
    rboxes = regularize_rboxes(x)

    ctr = rboxes[..., :2]
    w, h, angle = (rboxes[..., i : i + 1] for i in range(2, 5))

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


def class_dict_construct(data) -> Dict:
    """
    Creates the dictionary with the data for each class

    Args:
        data: The data from the ML model

    Returns:
        class_data: The dictionary with the ml data broken down by class
    """
    # break up by class
    sorted_cls_idx = np.argsort(data[:, -2])
    sorted_data = data[sorted_cls_idx]
    unique_cls, unique_cls_idx = np.unique(sorted_data[:, -2], return_index=True)
    split_data = np.split(sorted_data, unique_cls_idx[1:])

    # Create a dictionary with keys from 0 to 4, defaulting to None
    class_data = {cls: None for cls in range(5)}

    # Update the dictionary with the actual data
    for cls, cls_data in zip(unique_cls, split_data):
        # sort by z
        cls_data = cls_data[np.argsort(cls_data[:, -1])]
        class_data[int(cls)] = cls_data

    return class_data


def iou_volume(class_dict, c, vol, iou_threshold=0.1, z_iou_interval=50, z_length_min=50) -> List:
    """#
    This function groups the bounding boxes in the z direction based on their IoU. It groups boxes that have an IoU greater than 0.1 in the z direction.

    Args:
        class_dict: dictionary containing the bounding boxes for each class
        vol: volume of the image
        z_interval: Each bounding box has a z-interval in mm that determines how many neraby bounding boxes are included when calculating IoU. Defaults to 50 mm.
        min_z_size: The minimmum length in mm that a group of overlapping bounding boxes for it to be considered a detected object . Defaults to 50.

    Returns:
        _description_
    """
    # Calculate the z-interval and minimum z size in pixels
    z_iou_interval = int(z_iou_interval / vol.GetSpacing()[-1])
    z_length_min = int(z_length_min / vol.GetSpacing()[-1])

    # Process bounding boxes for each class if they exist
    xywhr, _, _, z = np.split(class_dict[c], [5, 6, 7], axis=1)
    z = z.flatten()
    # Calculate overlapping boxes within z-interval range
    groups = []
    for i in range(len(z)):
        range_min = np.clip(z[i] - z_iou_interval, z[0], z[-1])
        range_max = np.clip(z[i] + z_iou_interval, z[0], z[-1] + 1)
        nearby_i = (z >= range_min) & (z <= range_max)

        # Calculate IoU between current box and nearby boxes
        ious = batch_probiou(xywhr[i : i + 1, :], xywhr[nearby_i]).flatten()
        idx_iou = np.where(nearby_i)[0][np.where(ious > iou_threshold)[0]]
        groups.append(list(idx_iou))

    # Group overlapping boxes using UnionFind data structure
    ds = UnionFind()
    for gp in groups:
        ds.union(*gp)
    ds_sets = sorted([sorted(s) for s in ds.to_sets()], key=len, reverse=True)
    ds_sets = [s for s in ds_sets if len(s) >= z_length_min]

    return ds_sets


def predict(
    volume: str | pathlib.Path | sitk.Image, conf_thres, iou_thres
) -> Tuple[sitk.Image, Dict]:
    """
    Predict the oriented bounding boxes for each class in the input volume

    Args:
        volume: Path to the input volume for inference, or the sitk.Image object

    Returns:
        vol: The original volume
        aligned_vol_dict: Dictionary with keys for each class containing a list of sitk.Image 's for each detected object in the class
    """
    # Initialize model with specified image dimensions and load volume
    img_size = (640, 640)
    model, vol, vol_t = load(volume, img_size)

    # Process each axial slice for prediction
    data = []
    for z in range(vol_t.GetDepth()):
        # Prepare the 2D slice for model inference by expanding dimensions and repeating channels
        arr = sitk.GetArrayFromImage(vol_t[:, :, z])
        arr = np.expand_dims(arr, axis=0)
        arr = np.expand_dims(arr, axis=0)
        arr = np.repeat(arr, 3, axis=1)
        arr = arr.astype(np.float32)

        # run inference on the image
        output = model.run(None, {"images": arr})
        # perform non-max supression on the output
        preds = non_max_suppression_rotated(output[0], conf_thres, iou_thres)[0]
        # add the z coordinate to the predictions
        preds = np.c_[preds, np.ones((preds.shape[0], 1)) * z]
        data.extend(preds)

    # organize predictions by class
    cls_dict = class_dict_construct(np.array(data))

    return vol, cls_dict


class OBBCrop2Bone:
    """
    Aligns oriented bounding box volumes to bones in the input volume

    Args:
        volume: Path to the input volume for inference, or the sitk.Image object
        confidence_threshold: The confidence threshold below which boxes will be filtered out. Valid values are between 0.0 and 1.0. Defaults to 0.5.
        iou_supress_threshold: The IoU threshold above which boxes will be considered duplicates and filtered out during NMS. Valid values are between 0.0 and 1.0. Defaults to 0.4.

    Methods:

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
        volume: str | pathlib.Path | sitk.Image,
        confidence_threshold=0.5,
        iou_supress_threshold=0.4,
        debug_class=False,
    ):
        self.debug_points = False
        self.interpolator = sitk.sitkBSpline3
        if not debug_class:
            self.vol, self._class_dict = predict(
                volume, confidence_threshold, iou_supress_threshold
            )
        else:
            self.vol = sitk.ReadImage(str(volume))
            self._class_dict = {}

    def _obb(
        self,
        c_idx: int,
        iou_group_threshold: float,
        z_iou_interval: int,
        z_length_min: int,
    ) -> List[sitk.LabelShapeStatisticsImageFilter] | List:

        # Process all instances of the class and align volumes according to oriented bounding boxes
        if self._class_dict[c_idx] is None:
            return []
        else:
            obb_filters_list = []
            # group bounding boxes in the z direction based on IoU
            groups = iou_volume(
                self._class_dict,
                c_idx,
                self.vol,
                iou_group_threshold,
                z_iou_interval,
                z_length_min,
            )

            xywhr, _, _, z = np.split(self._class_dict[c_idx], [5, 6, 7], axis=1)
            for group in groups:
                # get the vertices of the group
                xywhr_group = xywhr[group]
                # Scale coordinates back to original volume dimensions
                xywhr_group[:, :-1] /= 640.0
                xywhr_group[:, :-1] *= float(self.vol.GetHeight())
                # convert to xy format
                xyxyxyxy_group = xywhr2xyxyxyxy(xywhr_group)
                # clip to volume dimensions to avoid out of bounds errors
                xyxyxyxy_group = np.clip(xyxyxyxy_group, 0, int(self.vol.GetHeight() - 1))
                # add in the z coordinate
                z_group = np.repeat(np.expand_dims(z[group], axis=1), 4, axis=1)
                xyz = np.concatenate([xyxyxyxy_group, z_group], axis=-1)
                xyz = xyz.reshape(-1, 3)
                xyz = xyz.astype(int)

                # Create boolean image marking box vertices
                zyx_bool = np.zeros(np.flip(self.vol.GetSize()))
                zyx_bool[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = 1
                zyx_bool = sitk.GetImageFromArray(zyx_bool)
                zyx_bool.CopyInformation(self.vol)
                zyx_bool = sitk.Cast(zyx_bool, sitk.sitkUInt8)

                if self.debug_points:
                    sitk.WriteImage(zyx_bool, f"zyx_bool_{c_idx}.seg.nrrd")

                # Calculate oriented bounding box using SimpleITK filter
                obb_filter = sitk.LabelShapeStatisticsImageFilter()
                obb_filter.ComputeOrientedBoundingBoxOn()
                obb_filter.Execute(zyx_bool)

                # record obb filter for testing purposes
                obb_filters_list.append(obb_filter)

        return obb_filters_list

    def _align(
        self,
        c_idx: int,
        obb_spacing: List,
        iou_threshold: float,
        z_iou_interval: int,
        z_length_min: int,
        xy_padding: int,
        z_padding: int,
    ) -> List[sitk.Image] | List:

        aligned_imgs = []
        self._resamplers = []
        self.detection_means = []
        # Process all instances of the class and align volumes according to oriented bounding boxes
        obb_filters_list = self._obb(c_idx, iou_threshold, z_iou_interval, z_length_min)

        for obb_filter in obb_filters_list:
            # get mean of detection not obb
            self.detection_means.append(obb_filter.GetCentroid(1))

            # get the direction of the obb and transpose it
            _d = obb_filter.GetOrientedBoundingBoxDirection(1)
            aligned_img_dir = np.array(_d).reshape(3, 3).T.flatten().tolist()

            # boot up the resampler
            resampler = sitk.ResampleImageFilter()
            # calculate the size of the aligned image
            aligned_img_size = [
                int(ceil(obb_filter.GetOrientedBoundingBoxSize(1)[i] / obb_spacing[i]))
                for i in range(3)
            ]
            # along the long axis, add the z padding, and xy padding for the other 2
            xy_padding_pix = xy_padding / obb_spacing[0]
            z_padding_pix = z_padding / obb_spacing[2]
            aligned_img_size = [
                (
                    int(aligned_img_size[i] + 2 * z_padding_pix)
                    if aligned_img_size[i] == max(aligned_img_size)
                    else int(aligned_img_size[i] + 2 * xy_padding_pix)
                )
                for i in range(3)
            ]
            # need to offset the origin by the padding
            obb_origin = obb_filter.GetOrientedBoundingBoxOrigin(1)
            obb_centroid = obb_filter.GetCentroid(1)
            offset_dir = np.sign(np.array(obb_origin) - np.array(obb_centroid))
            origin_offset = offset_dir * [
                z_padding if aligned_img_size[i] == max(aligned_img_size) else xy_padding
                for i in range(3)
            ]
            new_origin = np.array(obb_origin) + origin_offset

            resampler.SetOutputDirection(aligned_img_dir)
            resampler.SetOutputOrigin(new_origin)
            resampler.SetOutputSpacing(obb_spacing)
            resampler.SetSize(aligned_img_size)
            # resampler.SetOutputPixelType(sitk.sitkUInt8)
            resampler.SetInterpolator(self.interpolator)
            resampler.SetDefaultPixelValue(-1024)
            # record the resampler used for testing purposes
            self._resamplers.append(resampler)
            # get the aligned image, and its array
            aligned_img = resampler.Execute(self.vol)

            aligned_imgs.append(aligned_img)

        return aligned_imgs

    def clavicle(
        self,
        obb_spacing=[0.5, 0.5, 0.5],
        z_iou_threshold=0.23,
        z_iou_interval=21,
        z_length_min=20,
        xy_padding=0,
        z_padding=0,
    ) -> List[sitk.Image] | List:
        """
        Aligns an oriented bounding box volume to each clavicle in the scan

        Args:
            obb_spacing: The spacing of the oriented bounding box along all 3 axes in mm.
            io_threshold: The IoU threshold for grouping bounding boxes in the z direction.
            z_iou_interval: The z-interval in mm for grouping bounding boxes in the z direction.
            z_length_min: The minimum length in mm for a group of overlapping bounding boxes to be considered a detected object.
            xy_padding: The padding in mm to add to the xy dimensions of the aligned image.
            z_padding: The padding in mm to add to the z dimension of the aligned image.

        Returns:
            aligned_imgs: A list of the aligned images
        """
        return self._align(
            0,
            obb_spacing,
            z_iou_threshold,
            z_iou_interval,
            z_length_min,
            xy_padding,
            z_padding,
        )

    def scapula(
        self,
        obb_spacing=[0.5, 0.5, 0.5],
        z_iou_threshold=0.24,
        z_iou_interval=40,
        z_length_min=70,
        xy_padding=0,
        z_padding=0,
    ) -> List[sitk.Image] | List:
        """
        Aligns an oriented bounding box volume to each scapula in the scan

        Args:
            obb_spacing: The spacing of the oriented bounding box along all 3 axes in mm.
            io_threshold: The IoU threshold for grouping bounding boxes in the z direction.
            z_iou_interval: The z-interval in mm for grouping bounding boxes in the z direction.
            z_length_min: The minimum length in mm for a group of overlapping bounding boxes to be considered a detected object.
            xy_padding: The padding in mm to add to the xy dimensions of the aligned image.
            z_padding: The padding in mm to add to the z dimension of the aligned image.
        Returns:
            aligned_imgs: A list of the aligned images
        """

        return self._align(
            1,
            obb_spacing,
            z_iou_threshold,
            z_iou_interval,
            z_length_min,
            xy_padding,
            z_padding,
        )

    def humerus(
        self,
        obb_spacing=[0.5, 0.5, 0.5],
        z_iou_threshold=0.20,
        z_iou_interval=50,
        z_length_min=70,
        xy_padding=0,
        z_padding=0,
    ) -> List[sitk.Image] | List:
        """
        Aligns an oriented bounding box volume to each humerus in the scan

        Args:
            obb_spacing: The spacing of the oriented bounding box along all 3 axes in mm.
            io_threshold: The IoU threshold for grouping bounding boxes in the z direction.
            z_iou_interval: The z-interval in mm for grouping bounding boxes in the z direction.
            z_length_min: The minimum length in mm for a group of overlapping bounding boxes to be considered a detected object.
            xy_padding: The padding in mm to add to the xy dimensions of the aligned image.
            z_padding: The padding in mm to add to the z dimension of the aligned image.
        Returns:
            aligned_imgs: A list of the aligned images
        """
        return self._align(
            2,
            obb_spacing,
            z_iou_threshold,
            z_iou_interval,
            z_length_min,
            xy_padding,
            z_padding,
        )

    def radius_ulna(
        self,
        obb_spacing=[0.5, 0.5, 0.5],
        z_iou_threshold=0.28,
        z_iou_interval=30,
        z_length_min=70,
        xy_padding=0,
        z_padding=0,
    ) -> List[sitk.Image] | List:
        """
        Aligns an oriented bounding box volume to each radius ulna in the scan

        Args:
            obb_spacing: The spacing of the oriented bounding box along all 3 axes in mm.
            io_threshold: The IoU threshold for grouping bounding boxes in the z direction.
            z_iou_interval: The z-interval in mm for grouping bounding boxes in the z direction.
            z_length_min: The minimum length in mm for a group of overlapping bounding boxes to be considered a detected object.
            xy_padding: The padding in mm to add to the xy dimensions of the aligned image.
            z_padding: The padding in mm to add to the z dimension of the aligned image.
        Returns:
            aligned_imgs: A list of the aligned images
        """

        return self._align(
            3,
            obb_spacing,
            z_iou_threshold,
            z_iou_interval,
            z_length_min,
            xy_padding,
            z_padding,
        )

    def hand(
        self,
        obb_spacing=[0.5, 0.5, 0.5],
        z_iou_threshold=0.2,
        z_iou_interval=40,
        z_length_min=50,
        xy_padding=0,
        z_padding=0,
    ) -> List[sitk.Image] | List:
        """
        Aligns an oriented bounding box volume to each hand in the scan

        Args:
            obb_spacing: The spacing of the oriented bounding box along all 3 axes in mm.
            io_threshold: The IoU threshold for grouping bounding boxes in the z direction.
            z_iou_interval: The z-interval in mm for grouping bounding boxes in the z direction.
            z_length_min: The minimum length in mm for a group of overlapping bounding boxes to be considered a detected object.
            xy_padding: The padding in mm to add to the xy dimensions of the aligned image.
            z_padding: The padding in mm to add to the z dimension of the aligned image.
        Returns:
            aligned_imgs: A list of the aligned images
        """
        return self._align(
            4,
            obb_spacing,
            z_iou_threshold,
            z_iou_interval,
            z_length_min,
            xy_padding,
            z_padding,
        )


if __name__ == "__main__":
    ct_path = "/mnt/slowdata/cadaveric-full-arm/1602058L/1602058L.nrrd"

    # test obb crop
    obb_crop = OBBCrop2Bone(ct_path)
    # print(obb_crop._class_dict)
    for i, img in enumerate(obb_crop.scapula([0.5, 0.5, 0.5])):
        # print("obb")
        # print(img.GetSize())
        # print(img.GetOrigin())
        # print(img.GetDirection())
        # ct = sitk.ReadImage(ct_path)
        # print("ct")
        # print(ct.GetSize())
        # print(ct.GetOrigin())
        # print(ct.GetDirection())

        sitk.WriteImage(img, f"scapula-{i}.nrrd")
