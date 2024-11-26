import onnxruntime as rt
import SimpleITK as sitk
import numpy as np
import pathlib
from math import ceil
from copy import deepcopy

from concurrent.futures import ThreadPoolExecutor
from networkx.utils.union_find import UnionFind
import huggingface_hub


def get_model():
    model_path = huggingface_hub.hf_hub_download(
        repo_id="gregspangenberg/armcrop",
        filename="upperarm_yolo11_obb.onnx",
    )
    # model_path = "/home/greg/projects/segment/stage1_yolo_training/models/upperarm_yolo11_obb.onnx"
    return model_path


def load_model(img_size):
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
            providers = ["CPUExecutionProvider"]
        model = rt.InferenceSession(file.read(), providers=providers)

    # prime the model on a random image, to get the slow first inference out of the way
    model.run(
        None,
        {"images": np.random.rand(1, 3, img_size[0], img_size[1]).astype(np.float32)},
    )

    return model


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


def load(volume_path, img_size):
    with ThreadPoolExecutor() as executor:
        # Apply the tasks asynchronously
        volume_result = executor.submit(load_volume, volume_path, img_size)
        model_result = executor.submit(load_model, img_size)

        # Wait for results
        vol, vol_t = volume_result.result()
        model = model_result.result()
    return model, vol, vol_t


def non_max_suppression_rotated(prediction, conf_thres=0.4, iou_thres=0.2):
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

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
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


def nms_rotated(boxes, scores, threshold=0.45):
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

    sorted_idx = np.argsort(scores, axis=-1)[::-1]
    boxes = boxes[sorted_idx]

    ious = np.triu(batch_probiou(boxes, boxes), 1)
    pick = np.nonzero(ious.max(axis=0) < threshold)[0]

    return sorted_idx[pick]


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A tensor of shape (N, M) representing obb similarities.
    """

    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * np.power((y1 - y2), 2) + (b1 + b2) * np.power((x1 - x2), 2))
        / ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2) + eps)
    ) * 0.25

    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2) + eps)
    ) * 0.5

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

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1 - np.exp(-bd) + eps)

    return 1 - hd


def _get_covariance_matrix(boxes):
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


def xywhr2xyxyxyxy(x, round=False):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in radians from 0 to pi/2.

    Args:
        x (numpy.ndarray): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (np.cos, np.sin, np.concatenate, np.stack)

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)

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

    return stack([pt1, pt2, pt3, pt4], -2)


def predict(
    volume_path,
):
    # load model and volume
    img_size = (640, 640)
    model, vol, vol_t = load(volume_path, img_size)

    # loop over axial images and predict
    data = []
    for z in range(vol_t.GetDepth()):
        # prepare the image for inference
        arr = sitk.GetArrayFromImage(vol_t[:, :, z])
        arr = np.expand_dims(arr, axis=0)
        arr = np.expand_dims(arr, axis=0)
        arr = np.repeat(arr, 3, axis=1)
        arr = arr.astype(np.float32)

        # run inference on the image
        output = model.run(None, {"images": arr})
        preds = non_max_suppression_rotated(output[0])[0]

        preds = np.c_[preds, np.ones((preds.shape[0], 1)) * z]
        data.extend(preds)

    cls_dict = class_dict_construct(np.array(data))
    iou_dict = iou_volume(cls_dict, vol)
    aligned_vol_dict = obb_volume(cls_dict, iou_dict, vol, vol_t)

    return aligned_vol_dict


def class_dict_construct(data):
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


def iou_volume(class_dict, vol, z_interval=50, min_z_size=50):
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

    z_interval = int(z_interval / vol.GetSpacing()[-1])
    min_z_size = int(min_z_size / vol.GetSpacing()[-1])

    iou_dict = deepcopy(class_dict)
    for c in class_dict:
        groups = []
        if class_dict[c] is not None:
            xywhr, _, _, z = np.split(class_dict[c], [5, 6, 7], axis=1)
            z = z.flatten()
            for i in range(len(z)):
                range_min = np.clip(z[i] - z_interval, z[0], z[-1])
                range_max = np.clip(z[i] + z_interval, z[0], z[-1] + 1)
                nearby_i = (z >= range_min) & (z <= range_max)

                ious = batch_probiou(xywhr[i : i + 1, :], xywhr[nearby_i]).flatten()
                idx_iou = np.where(nearby_i)[0][np.where(ious > 0.1)[0]]
                groups.append(list(idx_iou))

            ds = UnionFind()
            for gp in groups:
                ds.union(*gp)
            ds_sets = sorted([sorted(s) for s in ds.to_sets()], key=len, reverse=True)
            ds_sets = [s for s in ds_sets if len(s) >= min_z_size]

            iou_dict[c] = ds_sets
    return iou_dict


def obb_volume(class_dict, iou_dict, vol, vol_t, obb_spacing=[0.5, 0.5, 0.5]):
    aligned_img_dict = deepcopy(class_dict)
    for c in class_dict:
        aligned_img_dict[c] = []
        if class_dict[c] is not None:
            xywhr, _, _, z = np.split(class_dict[c], [5, 6, 7], axis=1)
            for group_i, group in enumerate(iou_dict[c]):
                # get the vertices of the group
                xywhr_group = xywhr[group]
                # scale the vertices to the original volume
                xywhr_group[:, :-1] *= vol.GetWidth() / vol_t.GetWidth()
                # convert to xy format
                xyxyxyxy_group = xywhr2xyxyxyxy(xywhr_group, round=True)
                # add in the z coordinate
                z_group = np.repeat(np.expand_dims(z[group], axis=1), 4, axis=1)
                xyz = np.concatenate([xyxyxyxy_group, z_group], axis=-1)
                xyz = xyz.reshape(-1, 3)
                xyz = xyz.astype(int)

                # make a boolean image of the box vertices
                # sitk requires the image to be in zxy format because they hate
                # you and want to waste your time, i believe this has to do with RAS
                # vs LPS coordinate systems and sitk trying to be smart and interpret
                # which one you want without telling you
                zyx_bool = np.zeros(np.flip(vol.GetSize()))
                zyx_bool[xyz[:, 2], xyz[:, 0], xyz[:, 1]] = 1

                zyx_bool = sitk.GetImageFromArray(zyx_bool)
                zyx_bool.CopyInformation(vol)

                zyx_bool = sitk.Cast(zyx_bool, sitk.sitkUInt8)

                # compute the oriented bounding box
                obb_filter = sitk.LabelShapeStatisticsImageFilter()
                obb_filter.ComputeOrientedBoundingBoxOn()
                obb_filter.Execute(zyx_bool)

                # get the direction of the obb and transpose it
                _d = obb_filter.GetOrientedBoundingBoxDirection(1)
                aligned_img_dir = np.array(_d).reshape(3, 3).T.flatten().tolist()

                # boot up the resampler
                resampler = sitk.ResampleImageFilter()
                aligned_img_size = [
                    int(ceil(obb_filter.GetOrientedBoundingBoxSize(1)[i] / obb_spacing[i]))
                    for i in range(3)
                ]
                resampler.SetOutputDirection(aligned_img_dir)
                resampler.SetOutputOrigin(obb_filter.GetOrientedBoundingBoxOrigin(1))
                resampler.SetOutputSpacing(obb_spacing)
                resampler.SetSize(aligned_img_size)
                # resampler.SetOutputPixelType(sitk.sitkUInt8)
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetDefaultPixelValue(-1024)
                # get the aligned image, and its array
                aligned_img = resampler.Execute(vol)

                aligned_img_dict[c].append(aligned_img)
    return aligned_img_dict


if __name__ == "__main__":
    p = predict("/mnt/slowdata/cadaveric-full-arm/S202032/S202032.nrrd")

    for c in p:
        for i, img in enumerate(p[c]):
            sitk.WriteImage(img, f"aligned_{c}_{i}.nrrd")