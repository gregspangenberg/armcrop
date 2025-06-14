# *armcrop*

[![PyPI Latest Release](https://img.shields.io/pypi/v/armcrop.svg)](https://pypi.org/project/armcrop)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package contains a machine learning model that can crop a CT scan to any of the following classes:
- clavicle
- scapula
- humerus
- radius_ulna
- hand

It can do both axis aligned cropping and oriented bounding box (obb) cropping. The obb cropping allows for rigid registration of CT scans.

This model was trained on arm only scans and proximal half chest scans. A limitation of this model is that when using a full chest CT where the spine is present the model will struggle to differentiate between it and the scapula.

## Installation
To use CPU execution install with:
```
pip install armcrop
```
To use CUDA execution you will need to also install torch and onnxruntime-gpu.


## Usage
```python
import armcrop
import SimpleITK as sitk

# load CT scan
volume = sitk.ReadImage("path/to/ct_scan.nrrd")

# Oriented Bounding Box Cropping
cropper = armcrop.CropOrientedBoundingBox(volume)
cropped_images = cropper.process(bone="humerus")
for i, img in enumerate(cropped_images):
    sitk.WriteImage(img, f"aligned_humerus-{i}.nrrd")

# Bounding Box Cropping
cropper = armcrop.Crop(volume)
output = cropper.process(bone="scapula")
for i, img in enumerate(output):
    sitk.WriteImage(img, f"cropped_scapula-{i}.nrrd")

