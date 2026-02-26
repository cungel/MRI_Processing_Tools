# MRI Toolbox

A Python toolbox providing general utilities to process MRI NIfTI and DICOM files.

---
## Requirements

```bash
pip install -r requirements.txt
```
---
## Overview

This repository is continuously updated with new tools for MRI data processing.

---
### DICOM Utils

- **`multiecho_processing`** — Convert multi-echo DICOMs into separate NIfTI files, one per echo time.

---
### Volume Utils

- **`radial2SA`** Reorient a 3D cardiac segmentation (and optionally its image) from any acquisition plane into short-axis (SA) view. See [volume_utils/README.md](volume_utils/README.md).

---
### Label Utils

- **`label_cleaning`** Clean a multi-class segmentation map label by label (hole filling, small cluster removal, morphological operations, Gaussian smoothing).
- **`label_registration`** Register a moving segmentation map onto a fixed one using ANTs (rigid → affine). Optionally applies cleaning to the registered output.

---
### Examples

- **`HVSMR_processing`** Notebook demonstrating the pipeline: radial-to-SA reorientation and segmentation registration on publicly available data from the [HVSMR dataset](https://www.nature.com/articles/s41597-024-03469-9).
