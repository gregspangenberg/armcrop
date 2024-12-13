import SimpleITK as sitk
import SimpleITK.utilities.vtk
import vtk
import numpy as np
import cv2
import pathlib
from typing import Tuple, List, Dict

import time


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
        volume_path: Path to the original input volume used for obb inference
        thin_regions: A dictionary with the index of the thin outer region as the key and a tuple of the inner and outer indices to combine as the value. During unalignment thin regions can generate holes in the segmention. If there is an inner and outer region as seperate classes the outer region will have holes generated in its surface during unalignmnet. To prevent this you can combine the outer and inner regions during unalignment and the seperate them again after unalignment.
        force
        face_connectivity_regions: The index of the regions that should have face connectivity forced on them. This is useful for regions that are thin and have gaps in them after unalignment. This is different than a closing operation becasue it connects forces pixels that are 8-point connected to the rest of the segmetnation to be 4-point connected.
        face_connectivity_repeats: How many times to repeat the face connectivity operation.

    __call__(segmentation_path: str | pathlib.Path) -> sitk.Image:
    """

    def __init__(
        self,
        volume_path: str | pathlib.Path,
        thin_regions: Dict[int, Tuple] = {},
        face_connectivity_regions: List[int] = [],
        face_connectivity_repeats: int = 1,
    ):
        self.volume_path = volume_path
        self.volume = sitk.ReadImage(str(volume_path))

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

    def __call__(self, segmentation_path: str | pathlib.Path) -> sitk.Image:
        seg_sitk = sitk.ReadImage(str(segmentation_path))
        seg_sitk = sitk.Cast(seg_sitk, sitk.sitkInt8)

        # Get unique labels
        seg_array = sitk.GetArrayFromImage(seg_sitk)
        unique_labels = np.unique(seg_array)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background

        # time it
        start = time.time()

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
        new_seg = sitk.Image(self.volume.GetSize(), sitk.sitkUInt8)
        new_seg.CopyInformation(self.volume)  # Copy origin, spacing, direction
        new_seg = SimpleITK.utilities.vtk.sitk2vtk(new_seg)  # convert to VTK

        print(f"Time to create binary segs: {time.time() - start}")
        start = time.time()

        # generate meshes for each binary seg
        for i, bs in enumerate(binary_segs):
            # convert to vtk
            bs_vtk = SimpleITK.utilities.vtk.sitk2vtk(bs)

            # convert to polydata
            flying_edges = vtk.vtkDiscreteFlyingEdges3D()
            flying_edges.SetInputData(bs_vtk)
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

            # convert to image stencil
            PolyStencil = vtk.vtkPolyDataToImageStencil()
            PolyStencil.SetInputData(poly)
            PolyStencil.SetOutputSpacing(new_seg.GetSpacing())
            PolyStencil.SetOutputOrigin(new_seg.GetOrigin())
            PolyStencil.SetOutputWholeExtent(new_seg.GetExtent())
            PolyStencil.Update()

            # apply stencil to volume
            stencil = vtk.vtkImageStencil()
            stencil.SetInputData(new_seg)
            stencil.SetStencilConnection(PolyStencil.GetOutputPort())
            stencil.ReverseStencilOn()
            stencil.SetBackgroundValue(label_order[i])
            stencil.Update()
            new_seg = stencil.GetOutput()
        new_seg = SimpleITK.utilities.vtk.vtk2sitk(new_seg)

        print(f"Time to create stencils: {time.time() - start}")
        start = time.time()

        if self.face_conncectivity_regions:
            seg_array = sitk.GetArrayFromImage(new_seg)

            # setup correct order of binary masks
            binary_masks = []
            for label in self.face_conncectivity_regions:
                bm = seg_array == label
                for _ in range(self.face_conncectivity_repeats):
                    bm = self._force_face_connectivity(bm)
                binary_masks.append(bm * label)
                print(label, np.unique(bm * label, return_counts=True))

            for label in unique_labels[~np.isin(unique_labels, self.face_conncectivity_regions)]:
                binary_masks.append((seg_array == label) * label)
                print(label, np.unique((seg_array == label) * label, return_counts=True))

            arrs_combine = combine_arrays(binary_masks).astype(np.uint8)
            print(np.unique(arrs_combine, return_counts=True))

            new_seg_corrected = sitk.GetImageFromArray(arrs_combine)
            new_seg_corrected.CopyInformation(new_seg)

            print(f"Time to force face connectivity: {time.time() - start}")
            return new_seg_corrected

        else:
            return new_seg


if __name__ == "__main__":
    ct_path = "/mnt/slowdata/cadaveric-full-arm/1602058L/1602058L.nrrd"

    # test unaligner
    segmentation_path = "/home/greg/projects/armcrop/scapula-obb.seg.nrrd"
    unaligner = UnalignOBBSegmentation(
        ct_path,
        thin_regions={1: (1, 2)},
        face_connectivity_regions=[1],
        face_connectivity_repeats=2,
    )
    sitk.WriteImage(unaligner(segmentation_path), "scapula_obb2og-test.seg.nrrd")
