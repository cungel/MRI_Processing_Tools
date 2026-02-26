# MultiEcho Processing

Python tool to split multi-echo MRI volumes into independent echo images compatible with 3D Slicer visualization.

---

## Usage

```bash
python multiecho_processing.py --input_path <path> --output_path <path> --seq_name <name>
```

`--input_path` : Folder containing the DICOM files 
`--output_path` : Folder where NIfTI files will be saved
`--seq_name` : Series description to filter

---
## Notes
- `seq_name` filters the DICOM files using the **SeriesDescription** tag.
- Echo time (TE) is read from:  
  `PerFrameFunctionalGroupsSequence > MREchoSequence > EffectiveEchoTime`.  
- Affine matrix is constructed from:
  - `ImageOrientationPatient`
  - `PixelSpacing`
  - `SliceThickness`
  - `ImagePositionPatient`

