import argparse
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation


def compute_center_of_mass(mask, spacing=(1.0, 1.0, 1.0)):
    """
    Compute the center of mass of a 3D binary mask in physical coordinates.

    Parameters
    ----------
    mask : np.ndarray
        3D binary volume (non-zero voxels are considered foreground).
    spacing : tuple of float
        Voxel size in mm along each axis (x, y, z).

    Returns
    -------
    com : np.ndarray, shape (3,)
        Center of mass in physical coordinates (mm).
    """
    coords = np.argwhere(mask > 0)

    if coords.shape[0] == 0:
        raise ValueError("Mask is empty.")

    coords_phys = coords * np.array(spacing)
    com = coords_phys.mean(axis=0)
    return com


def apply_rotation_matrix(volume, R, order=0, output_shape=None):
    """
    Apply a 3x3 rotation matrix to a 3D volume, rotating around its center.

    The output shape is automatically computed from the rotated bounding box
    if not provided. Uses inverse mapping (scipy affine_transform) with
    interpolation order controlled by `order`.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume to rotate.
    R : np.ndarray, shape (3, 3)
        Orthonormal rotation matrix (rows are the new basis vectors).
    order : int
        Interpolation order (0 = nearest neighbor, 1 = linear, ...).
    output_shape : tuple of int, optional
        Shape of the output volume. Computed automatically if None.

    Returns
    -------
    rotated : np.ndarray
        Rotated volume with the computed or specified output shape.
    """
    shape = np.array(volume.shape)

    if output_shape is None:
        corners = np.array([
            [0,        0,        0       ],
            [shape[0], 0,        0       ],
            [0,        shape[1], 0       ],
            [0,        0,        shape[2]],
            [shape[0], shape[1], 0       ],
            [shape[0], 0,        shape[2]],
            [0,        shape[1], shape[2]],
            [shape[0], shape[1], shape[2]],
        ])

        rotated_corners = corners @ R.T
        mins = rotated_corners.min(axis=0)
        maxs = rotated_corners.max(axis=0)
        output_shape = np.ceil(maxs - mins).astype(int)

    center_input = shape / 2.0
    center_output = np.array(output_shape) / 2.0

    R_inv = R.T
    offset = center_input - R_inv @ center_output

    rotated = affine_transform(
        volume,
        R_inv,
        offset=offset,
        output_shape=tuple(output_shape),
        order=order,
        mode="constant",
        cval=0,
    )
    return rotated


def compute_rotation_to_align_z(long_axis):
    """
    Compute a rotation matrix aligning the cardiac long-axis with the Z axis.

    Uses scipy's Rotation.align_vectors to compute the optimal rotation
    that maps long_axis → [0, 0, 1].

    Parameters
    ----------
    long_axis : np.ndarray, shape (3,)
        Direction vector (need not be normalized).

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix.
    """
    long_axis = long_axis / np.linalg.norm(long_axis)
    target_axis = np.array([0.0, 0.0, 1.0])

    rot, _ = Rotation.align_vectors([target_axis], [long_axis])
    return rot.as_matrix()


def volume_to_SA(segmentation_path, output_path, image_path=None, output_filename='SA'):
    """
    Reorient a cardiac segmentation (and optionally its image) into short-axis view.

    Computes the cardiac long axis from the centers of mass of the atria and
    ventricles, then rotates the volume so that this axis aligns with Z.
    Saves the reoriented NIfTI files with a corrected affine matrix.

    Labels expected in the segmentation:
        1-2 = ventricles, 3-4 = atria.

    Parameters
    ----------
    segmentation_path : str
        Path to the input segmentation NIfTI file.
    output_path : str
        Directory where output files will be saved (created if it does not exist).
    image_path : str or None
        Path to the corresponding image NIfTI file. Skipped if None or missing.
    output_filename : str
        Base name for output files (suffixed with ``_seg.nii.gz`` and ``_im.nii``).

    Returns
    -------
    None
    """
    seg_nii = nib.load(segmentation_path)
    seg = seg_nii.get_fdata()
    spacing = seg_nii.header.get_zooms()[:3]

    mask_ventricles = np.isin(seg, [1,2])
    mask_atrium = np.isin(seg, [3, 4])

    com_atrium = compute_center_of_mass(mask_atrium, spacing)
    com_ventricles = compute_center_of_mass(mask_ventricles, spacing)

    long_axis = com_ventricles - com_atrium
    long_axis = long_axis / np.linalg.norm(long_axis)

    R = compute_rotation_to_align_z(long_axis)
    seg_rot = apply_rotation_matrix(seg, R, order=0)

    out_spacing = np.linalg.norm(np.diag(spacing) @ R.T, axis=0)
    new_affine = np.diag([*out_spacing, 1.0])

    os.makedirs(output_path, exist_ok=True)
    seg_output_path = os.path.join(output_path, output_filename + "_seg.nii.gz")
    nib.save(nib.Nifti1Image(seg_rot, new_affine), seg_output_path)

    if image_path and os.path.exists(image_path):
        im_nii = nib.load(image_path)
        im = im_nii.get_fdata()
        img_rot = apply_rotation_matrix(im, R, order=1)
        im_output_path = os.path.join(output_path, output_filename + "_im.nii")
        nib.save(nib.Nifti1Image(img_rot, new_affine), im_output_path)

    print("Saved:", output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reorient a 3D cardiac NIfTI volume into short-axis view."
    )
    parser.add_argument("--segmentation_path", required=True, help="Path to the segmentation NIfTI file.")
    parser.add_argument("--output_path", required=True, help="Output directory.")
    parser.add_argument("--image_path", required=False, help="Path to the image NIfTI file (optional).")
    parser.add_argument("--output_filename",required=False,default="SAX",help="Base name for output files. Default: SAX")
    args = parser.parse_args()

    volume_to_SA(args.segmentation_path, args.image_path, args.output_path, args.output_filename)
