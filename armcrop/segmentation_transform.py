import SimpleITK as sitk
import SimpleITK.utilities.vtk
import vtk
import numpy as np
import cv2
import pathlib
from typing import Tuple, List, Dict


def combine_arrays(array_stack):
    """Combine array stack from bottom up, earlier arrays overwrite later ones"""
    # Start with deepest layer
    result = array_stack[-1].copy()

    # Work backwards, using addition and multiplication for masking
    for arr in array_stack[-2::-1]:
        # Zero out areas where higher priority array has values
        result *= arr == 0
        # Add the higher priority array
        result += arr

    return result


class UnalignOBBSegmentation:
    """
    Unaligns a segmentation made in and oriented bounding box to the original volume

    Args:
        volume: Path to the original input volume, or sitk.Image used for obb inference
        thin_regions: A dictionary with the index of the thin outer region as the key and a tuple of the inner and outer indices to combine as the value. During unalignment thin regions can generate holes in the segmention. If there is an inner and outer region as seperate classes the outer region will have holes generated in its surface during unalignmnet. To prevent this you can combine the outer and inner regions during unalignment and the seperate them again after unalignment.
        force
        face_connectivity_regions: The index of the regions that should have face connectivity forced on them. This is useful for regions that are thin and have gaps in them after unalignment. This is different than a closing operation becasue it connects forces pixels that are 8-point connected to the rest of the segmetnation to be 4-point connected.
        face_connectivity_repeats: How many times to repeat the face connectivity operation.

    __call__(segmentation_path: str | pathlib.Path) -> sitk.Image:
    """

    def __init__(
        self,
        volume: str | pathlib.Path | sitk.Image,
        thin_regions: Dict[int, Tuple] = {},
        face_connectivity_regions: List[int] = [],
        face_connectivity_repeats: int = 1,
    ):
        if isinstance(volume, sitk.Image):
            self.volume = volume
        else:
            self.volume = sitk.ReadImage(str(volume))

        self.thin_regions = thin_regions
        self.face_conncectivity_regions = face_connectivity_regions
        self.face_conncectivity_repeats = face_connectivity_repeats

    def _force_face_connectivity(self, arr_og: np.ndarray) -> np.ndarray:
        """Force 4-point connectivity on each slice in each direction in the segmentation. This bridges small gaps in the segmentation that closing can't fix."""

        def force_4_connectivity_2d(arr):
            # Label components using 4-connectivity
            _, labeled = cv2.connectedComponents(arr, connectivity=4)

            # Find both left and right diagonal neighbors with different labels in one operation
            upleft = np.roll(np.roll(labeled, -1, axis=0), -1, axis=1)  # Up-Left
            upright = np.roll(np.roll(labeled, -1, axis=0), 1, axis=1)  # Up-Right
            downleft = np.roll(np.roll(labeled, 1, axis=0), -1, axis=1)  # Down-Left
            downright = np.roll(np.roll(labeled, 1, axis=0), 1, axis=1)  # Down-Right

            diag_diff = (
                ((labeled != upleft) & (labeled > 0) & (upleft > 0))
                | ((labeled != upright) & (labeled > 0) & (upright > 0))
                | ((labeled != downleft) & (labeled > 0) & (downleft > 0))
                | ((labeled != downright) & (labeled > 0) & (downright > 0))
            )

            # Create orthogonal connections once
            result = arr.copy()
            result |= np.roll(diag_diff, 1, axis=1)  # Horizontal connection
            result |= np.roll(diag_diff, 1, axis=0)  # Vertical connection

            return result

        arr = arr_og.copy().astype(np.uint8)
        for z in range(arr.shape[0]):
            arr[z, :, :] = force_4_connectivity_2d(arr[z, :, :])
        for x in range(arr.shape[1]):
            arr[:, x, :] = force_4_connectivity_2d(arr[:, x, :])
        for x in range(arr.shape[2]):
            arr[:, :, x] = force_4_connectivity_2d(arr[:, :, x])

        return arr

    def __call__(self, obb_segmentation: str | pathlib.Path |sitk.Image) -> sitk.Image:
    
        if isinstance(obb_segmentation, sitk.Image):
            seg_sitk = obb_segmentation
        else:
            seg_sitk = sitk.ReadImage(str(obb_segmentation))
        seg_sitk = sitk.Cast(seg_sitk, sitk.sitkInt8)

        # Get unique labels
        seg_array = sitk.GetArrayFromImage(seg_sitk)
        unique_labels = np.unique(seg_array)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background

        # transform multi seg to list of binary segs
        binary_segs = []
        label_order = []
        for label in unique_labels:
            # Create binary mask where label == value
            if label in self.thin_regions:
                mask = np.zeros_like(seg_array)
                for value in self.thin_regions[label]:
                    mask += (seg_array == value).astype(np.uint8)
                mask = np.clip(mask, 0, 1)

            else:
                mask = (seg_array == label).astype(np.uint8)

            binary_sitk = sitk.GetImageFromArray(mask)
            binary_sitk.CopyInformation(seg_sitk)  # Copy metadata

            # the later elements in the list are overwrite the earlier ones
            if label in self.thin_regions:
                binary_segs.insert(0, binary_sitk)
                label_order.insert(0, label)
            else:
                binary_segs.append(binary_sitk)
                label_order.append(label)

        # create sitk image of 0s that match shape of volume
        B_seg = sitk.Image(self.volume.GetSize(), sitk.sitkUInt8)
        B_seg.CopyInformation(self.volume)  # Copy origin, spacing, direction
        B_seg = SimpleITK.utilities.vtk.sitk2vtk(B_seg)  # convert to VTK

        # generate meshes for each binary seg
        for i, bin_seg in enumerate(binary_segs):
            # convert to vtk
            A_seg = SimpleITK.utilities.vtk.sitk2vtk(bin_seg)

            # convert to polydata
            flying_edges = vtk.vtkDiscreteFlyingEdges3D()
            flying_edges.SetInputData(A_seg)
            flying_edges.GenerateValues(1, 1, 1)
            flying_edges.Update()
            poly = flying_edges.GetOutput()

            # apply windowed sinc filter
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(poly)
            # less smoothing
            smoother.SetNumberOfIterations(20)
            smoother.SetPassBand(0.1)
            # # more smoothing
            # smoother.SetNumberOfIterations(40)
            # smoother.SetPassBand(0.01)
            smoother.BoundarySmoothingOff()
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.Update()
            poly = smoother.GetOutput()

            # get B direction
            B_dir = np.array(B_seg.GetDirectionMatrix().GetData()).reshape(3, 3)
            # get current origin
            B_origin = np.array(B_seg.GetOrigin()).reshape(3, 1)

            # convert to image stencil
            PolyStencil = vtk.vtkPolyDataToImageStencil()
            PolyStencil.SetInputData(poly)
            PolyStencil.SetOutputOrigin(B_seg.GetOrigin())
            PolyStencil.SetOutputSpacing(B_seg.GetSpacing())
            PolyStencil.SetOutputWholeExtent(B_seg.GetExtent())

            PolyStencil.Update()

            # apply stencil to volume
            stencil = vtk.vtkImageStencil()
            stencil.SetInputData(B_seg)
            stencil.SetStencilConnection(PolyStencil.GetOutputPort())
            stencil.ReverseStencilOn()
            stencil.SetBackgroundValue(label_order[i])
            stencil.Update()
            B_seg = stencil.GetOutput()

        B_seg = SimpleITK.utilities.vtk.vtk2sitk(B_seg)

        if self.face_conncectivity_regions:
            seg_array = sitk.GetArrayFromImage(B_seg)

            # setup correct order of binary masks
            binary_masks = []
            for label in self.face_conncectivity_regions:
                bm = seg_array == label
                for _ in range(self.face_conncectivity_repeats):
                    bm = self._force_face_connectivity(bm)
                binary_masks.append(bm * label)

            for label in unique_labels[~np.isin(unique_labels, self.face_conncectivity_regions)]:
                binary_masks.append((seg_array == label) * label)

            arrs_combine = combine_arrays(binary_masks).astype(np.uint8)

            new_seg_corrected = sitk.GetImageFromArray(arrs_combine)
            new_seg_corrected.CopyInformation(B_seg)

            return new_seg_corrected

        else:
            return B_seg


class AlignOBBSegmentation(UnalignOBBSegmentation):
    """
    Aligns a segmentation made in the original volume to the Oriented Bounding Box

    Args:
        obb_volume: Path to the oriented bounding box volume or sitk.Image created after inference
        thin_regions: A dictionary with the index of the thin outer region as the key and a tuple of the inner and outer indices to combine as the value. During unalignment thin regions can generate holes in the segmention. If there is an inner and outer region as seperate classes the outer region will have holes generated in its surface during unalignmnet. To prevent this you can combine the outer and inner regions during unalignment and the seperate them again after unalignment.
        force
        face_connectivity_regions: The index of the regions that should have face connectivity forced on them. This is useful for regions that are thin and have gaps in them after unalignment. This is different than a closing operation becasue it connects forces pixels that are 8-point connected to the rest of the segmetnation to be 4-point connected.
        face_connectivity_repeats: How many times to repeat the face connectivity operation.

    __call__(segmentation_path: str | pathlib.Path) -> sitk.Image:
    """

    def __init__(
        self,
        obb_volume: str | pathlib.Path | sitk.Image,
        thin_regions: Dict[int, Tuple] = {},
        face_connectivity_regions: List[int] = [],
        face_connectivity_repeats: int = 1,
    ):
        super().__init__(
            obb_volume,
            thin_regions,
            face_connectivity_regions,
            face_connectivity_repeats,
        )

    def __call__(self, segmentation: str | pathlib.Path |sitk.Image) -> sitk.Image:
        if isinstance(segmentation, sitk.Image):
            seg_sitk = segmentation
        else:
            seg_sitk = sitk.ReadImage(str(segmentation))
        seg_sitk = sitk.Cast(seg_sitk, sitk.sitkInt8)

        # Get unique labels
        seg_array = sitk.GetArrayFromImage(seg_sitk)
        unique_labels = np.unique(seg_array)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background

        # transform multi seg to list of binary segs
        binary_segs = []
        label_order = []
        for label in unique_labels:
            # Create binary mask where label == value
            if label in self.thin_regions:
                mask = np.zeros_like(seg_array)
                for value in self.thin_regions[label]:
                    mask += (seg_array == value).astype(np.uint8)
                mask = np.clip(mask, 0, 1)

            else:
                mask = (seg_array == label).astype(np.uint8)

            binary_sitk = sitk.GetImageFromArray(mask)
            binary_sitk.CopyInformation(seg_sitk)  # Copy metadata

            # the later elements in the list are overwrite the earlier ones
            if label in self.thin_regions:
                binary_segs.insert(-1, binary_sitk)
                label_order.insert(-1, label)
            else:
                binary_segs.insert(0, binary_sitk)
                label_order.insert(0, label)

        # generate meshes for each binary seg
        arrays_bin_segs = []
        for i, A_seg in enumerate(binary_segs):

            # resample
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(self.volume)
            resampler.SetInterpolator(sitk.sitkLabelLinear)
            B_bin_seg = resampler.Execute(A_seg)

            B_bin_seg_array = sitk.GetArrayFromImage(B_bin_seg).astype(np.int16)

            if label in self.face_conncectivity_regions:
                for _ in range(self.face_conncectivity_repeats):
                    B_bin_seg_array = self._force_face_connectivity(B_bin_seg_array)

            # add in label
            B_bin_seg_array *= label_order[i]
            # record array
            arrays_bin_segs.append(B_bin_seg_array)

        arrs_combine = combine_arrays(arrays_bin_segs)

        B_seg = sitk.GetImageFromArray(arrs_combine)
        B_seg.CopyInformation(self.volume)

        return B_seg


if __name__ == "__main__":
    ct_path = "/mnt/slowdata/cadaveric-full-arm/1602058L/1602058L.nrrd"
    obb_path = "/home/greg/projects/armcrop/scapula-0.nrrd"

    # test unaligner
    segmentation_path = "/home/greg/projects/armcrop/scapula_obb.seg.nrrd"
    # segmentation_path = "/home/greg/projects/armcrop/scapula_obb2og.seg.nrrd"
    unaligner = AlignOBBSegmentation(
        ct_path,
        # obb_path,
        thin_regions={1: (1, 2)},
        face_connectivity_regions=[1],
        face_connectivity_repeats=2,
    )
    unalgined_sitk = unaligner(segmentation_path)
    print(np.unique(sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path)), return_counts=True))
    print(np.unique(sitk.GetArrayFromImage(unalgined_sitk), return_counts=True))
    sitk.WriteImage(unalgined_sitk, "scapula_obb2og2obb.seg.nrrd")
