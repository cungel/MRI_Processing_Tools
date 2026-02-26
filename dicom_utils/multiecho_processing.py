import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pydicom


def dicom_to_affine(ds):
    """
    Build an affine matrix from a DICOM dataset.

    Extracts ImageOrientationPatient, PixelSpacing, SliceThickness, and
    ImagePositionPatient from the first frame's functional group sequences
    to construct the voxel-to-world mapping expected by NIfTI.

    Parameters
    ----------
    ds : pydicom.Dataset
        Enhanced DICOM dataset (single file, multi-frame).

    Returns
    -------
    affine : np.ndarray, shape (4, 4)
        Affine matrix mapping voxel indices (cols, rows, slices) to mm coordinates.
    """
    orientation = np.array(
        ds.PerFrameFunctionalGroupsSequence[0]
        .PlaneOrientationSequence[0]
        .ImageOrientationPatient
    ).reshape(2, 3)
    row_cosine = orientation[0]
    col_cosine = orientation[1]
    normal = np.cross(row_cosine, col_cosine)

    spacing = list(map(float, ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing))
    slice_thickness = float(ds.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
    origin = np.array(ds.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient)

    affine = np.eye(4)
    affine[:3, 0] = row_cosine * spacing[1]
    affine[:3, 1] = col_cosine * spacing[0]
    affine[:3, 2] = normal * slice_thickness
    affine[:3, 3] = origin

    return affine


def convert_multi_echo_nii(input_path, output_path, seq_name):
    """
    Convert multi-echo DICOM files matching a sequence name to NIfTI, one file per echo.

    Scans the input folder for DICOM files matching the given sequence name,
    extracts the echo time (TE) from each DICOM, reconstructs the volume with
    the proper affine, and saves it as a NIfTI file.

    Parameters
    ----------
    input_path : str
        Path to the folder containing the DICOM files.
    output_path : str
        Path to the folder where NIfTI files will be saved (created if it does not exist).
    seq_name : str
        Sequence name (SeriesDescription) used to filter DICOM files.

    Returns
    -------
    None
    """
    os.makedirs(output_path, exist_ok=True)
    dicom_files = sorted(glob.glob(os.path.join(input_path, "*")))

    count = 0

    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f)

            if not hasattr(ds, "SeriesDescription"):
                continue

            if ds.SeriesDescription != seq_name:
                continue

            try:
                te = ds.PerFrameFunctionalGroupsSequence[0].MREchoSequence[0].EffectiveEchoTime
            except Exception:
                print(f"Skipping {f} (no TE found)")
                continue

            te = float(te)
            te_str = f"{te:.2f}".replace(".", "_")

            vol = ds.pixel_array
            vol = np.transpose(vol, (2, 1, 0))
            affine = dicom_to_affine(ds)
            nifti = nib.Nifti1Image(vol, affine)
            output_file = os.path.join(output_path, f"{seq_name}_TE_{te_str}.nii.gz")
            nib.save(nifti, output_file)

            count += 1

        except Exception as e:
            print(f"Error processing {f}: {e}")

    print(f"{count} volumes converted and saved to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert multi-echo DICOMs to NIfTI, one file per echo.")
    parser.add_argument("--input_path", required=True, help="Folder containing the DICOM files.")
    parser.add_argument("--output_path", required=True, help="Folder to save NIfTI files.")
    parser.add_argument("--seq_name", required=True, help="Sequence name (SeriesDescription) to filter.")
    args = parser.parse_args()

    convert_multi_echo_nii(args.input_path, args.output_path, args.seq_name)
