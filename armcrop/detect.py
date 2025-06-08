import pathlib
import numpy as np
from typing import Tuple, List, Dict
import onnxruntime as rt
import SimpleITK as sitk

from copy import deepcopy
from armcrop.model_manager import ModelManager
import abc

# disable logging
rt.set_default_logger_severity(3)


class Detector(abc.ABC):
    """
    Base class for cropping volumes using a machine learning model.
    """

    def __init__(self, img_size=(640, 640), model_type="default"):
        """
        Initialize the BaseCrop class.

        Args:
            img_size: Image size required for the ML model. Defaults to (640, 640).
        """
        self.img_size = img_size
        self.model = None
        self.model_type = model_type
        self.results = {}

    def format_volume(self, volume: sitk.Image, img_size=(640, 640)) -> sitk.Image:
        """
        Load a volume and preprocess it for inference

        Args:
            volume: Path to the input volume for inference, or the sitk.Image volume
            img_size: Image size required for the ML model. Defaults to (640, 640).

        Returns:
            vol: The original volume
            vol_t: The preprocessed volume
        """
        vol_t = deepcopy(volume)
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

        return vol_t

    def load_model(self) -> rt.InferenceSession:
        """Load the ML model for inference.

        Returns:
            rt.InferenceSession: The loaded ML model.
        """
        # Use the model manager to get the cached model
        model_manager = ModelManager.get_instance(self.model_type)
        model = model_manager.load_model()

        return model

    def _preprocess_array(self, arr) -> np.ndarray:
        arr = np.expand_dims(arr, axis=0)
        arr = np.expand_dims(arr, axis=0)
        arr = np.repeat(arr, 3, axis=1)
        arr = arr.astype(np.float32)

        return arr

    def _class_dict_construct(self, data) -> Dict:
        """
        Creates the dictionary with the data for each class

        Args:
            data: The data from the ML model

        Returns:
            class_data: The dictionary with the ml data broken down by class
        """
        # break up by class
        sorted_cls_idx = np.argsort(data[:, -1])
        sorted_data = data[sorted_cls_idx]
        unique_cls, unique_cls_idx = np.unique(sorted_data[:, -1], return_index=True)
        split_data = np.split(sorted_data, unique_cls_idx[1:])

        # Create a dictionary with keys from 0 to 4, defaulting to None
        class_data = {cls: None for cls in range(5)}

        # Update the dictionary with the actual data
        for cls, cls_data in zip(unique_cls, split_data):
            # sort by z
            cls_data = cls_data[np.argsort(cls_data[:, -1])]
            class_data[int(cls)] = cls_data

        return class_data

    def predict(
        self, volume: sitk.Image, confidence_threshold, iou_threshold
    ) -> Dict[int, np.ndarray]:
        """
        Run inference on the volume and return the cropped volume.

        Args:
            volume: The input volume for inference.
            confidence_threshold: Confidence threshold for predictions.
            iou_threshold: IoU threshold for non-max suppression.

        Returns:
            A dictionary with class indices as keys and numpy arrays of predictions as values.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class YOLODetector(Detector):
    """
    Detector for cropping volumes using an Oriented Bounding Box (OBB) model.
    """

    def __init__(self, img_size=(640, 640), model_type="yolo11-obb"):
        super().__init__(img_size, model_type)

    def nms_rotated_batch(self, prediction, conf_thres, iou_thres) -> List[np.ndarray]:
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
            List[np.ndarray]: A list of numpy arrays, each containing the filtered boxes for each image in the batch.
             The columns are x,y,w,h, rotation, confidence,class_id
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

            # Detections matrix nx6 (xywh, conf, cls, rot)
            box = x[:, :4]  # xywh
            cls = x[:, 4:mi]  # classes
            rot = x[:, mi : mi + 1]  # rotation
            if rot.shape[1] == 0:  # if rotation is not present, add a column of zeros
                rot = np.zeros((box.shape[0], 1), dtype=np.float32)
            conf = np.max(cls, axis=1, keepdims=True)
            class_id = np.argmax(cls, axis=1, keepdims=True)

            x = np.concatenate((box, rot, conf, class_id), 1)[conf.squeeze(1) > conf_thres]

            if not x.shape[0]:  # no boxes
                continue

            # Batched NMS
            boxes = x[:, :5]  # xywhr
            scores = x[:, 5]  # scores

            i = self._nms_rotated(boxes, scores, iou_thres)
            output[xi] = x[i]

        return output

    def _nms_rotated(self, boxes, scores, threshold) -> np.ndarray:
        """
        NMS for oriented bounding boxes using probiou and fast-nms.

        Args:
            boxes (np.ndarray): Rotated bounding boxes, shape (N, 5), format xywhr.
            scores (np.ndarray): Confidence scores, shape (N,).
            threshold (float, optional): IoU threshold

        Returns:
            (np.ndarray): Indices of boxes to keep after NMS.
        """

        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int8)

        # Sort boxes by confidence scores in descending order
        sorted_idx = np.argsort(scores, axis=-1)[::-1]
        boxes = boxes[sorted_idx]

        ious = np.triu(self._batch_probiou(boxes, boxes), 1)
        # Filter boxes based on IOU threshold to remove overlapping detections

        pick = np.nonzero(ious.max(axis=0) < threshold)[0]

        return sorted_idx[pick]

    def _batch_probiou(self, obb1, obb2, eps=1e-7) -> np.ndarray:
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
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = (x.squeeze(-1)[None] for x in self._get_covariance_matrix(obb2))

        # Calculate first term of Bhattacharyya distance using quadratic forms
        t1 = (
            ((a1 + a2) * np.power((y1 - y2), 2) + (b1 + b2) * np.power((x1 - x2), 2))
            / ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2) + eps)
        ) * 0.25

        # Calculate second term (cross term) of Bhattacharyya distance
        t2 = (
            ((c1 + c2) * (x2 - x1) * (y1 - y2))
            / ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2) + eps)
        ) * 0.5

        # Calculate third term (determinant term) of Bhattacharyya distance
        t3 = (
            np.log(
                ((a1 + a2) * (b1 + b2) - np.power((c1 + c2), 2))
                / (
                    4
                    * np.sqrt(
                        (a1 * b1 - (np.power(c1, 2)).clip(0))
                        * (a2 * b2 - (np.power(c2, 2)).clip(0))
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

    def _get_covariance_matrix(self, boxes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generating covariance matrix from obbs.

        Args:
            boxes (np.ndarray): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

        Returns:
            (np.ndarray): Covariance matrices corresponding to original rotated bounding boxes.
        """
        # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
        gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:5]), axis=-1)
        a, b, c = np.split(gbbs, 3, axis=-1)
        cos = np.cos(c)
        sin = np.sin(c)
        cos2 = np.power(cos, 2)
        sin2 = np.power(sin, 2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    def predict(
        self, volume: sitk.Image, confidence_threshold, iou_threshold
    ) -> Dict[int, np.ndarray]:
        """
        Run inference on the volume and return the cropped volume.

        Args:
            volume: The input volume for inference.
            confidence_threshold: Confidence threshold for predictions.
            iou_threshold: IoU threshold for non-max suppression.

        Returns:
            vol: The original volume.
            vol_t: The cropped volume based on the OBB model predictions.
        """
        model = self.load_model()
        original_size = (volume.GetWidth(), volume.GetHeight())
        vol_t = self.format_volume(volume, self.img_size)
        # get original width of the volume

        # Process each axial slice for prediction
        data = []
        for z in range(vol_t.GetDepth()):
            # Prepare the 2D slice for model inference by expanding dimensions and repeating channels
            arr = sitk.GetArrayFromImage(vol_t[:, :, z])
            arr = self._preprocess_array(arr)
            # run inference on the image
            output = model.run(None, {"images": arr})[0]

            # perform non-max supression on the output
            preds = self.nms_rotated_batch(output, confidence_threshold, iou_threshold)[0]

            # convert xywh out of 640 to original volume size
            preds[:, 0] = preds[:, 0] * original_size[0] / self.img_size[0]
            preds[:, 1] = preds[:, 1] * original_size[1] / self.img_size[1]
            preds[:, 2] = preds[:, 2] * original_size[0] / self.img_size[0]
            preds[:, 3] = preds[:, 3] * original_size[1] / self.img_size[1]

            # add the z coordinate to the predictions
            preds = np.c_[np.ones((preds.shape[0], 1)) * z, preds]
            data.extend(preds)

        # organize predictions by class
        # output is z, xywhr, confidence, class_id,
        cls_dict = self._class_dict_construct(np.array(data))
        self.results = cls_dict

        return self.results


if __name__ == "__main__":
    # Example usage
    detector = YOLODetector()
    volume = sitk.ReadImage("/mnt/slowdata/ct/cadaveric-full-arm/1606011L/1606011L.nrrd")
    conf_threshold = 0.5
    iou_threshold = 0.5

    predictions = detector.predict(volume, conf_threshold, iou_threshold)
    for cls, preds in predictions.items():
        print(f"Class {cls}: {preds.shape[0]} predictions")
        print(preds.shape)
        print(np.max(preds[:, 1]), np.min(preds[:, 1]))
