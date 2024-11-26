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

This model was trained on arm only scans and proximal half chest scans. The model will struggle to correctly identify the scapula correctly with full chest CT scans as this orientation was't included in the dataset.



## Installation
To use CPU execution install with:
```
pip install armcrop
```
To use CUDA execution you will need to also install torch and onnxruntime-gpu.


## Contributing 
Clone the repo, open the cloned folder containing the poetry.lock file, then install the development dependencies using poetry. 
```
poetry install --with dev
``` 

