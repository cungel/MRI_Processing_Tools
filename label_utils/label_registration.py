import argparse
import os

import ants
import SimpleITK as sitk

from .label_cleaning import clean_segmentation_sitk_parallel


def labelmaps_registration(
    fixed_seg_path,
    moving_seg_path,
    out_dir,
    moving_im_path=None,
    filename='registered',
    clean=True,
    min_cluster_size=5,
    gaussian_sigma=1.5,
    sensitive_class=None,
):
    """
    Register a moving labelmap onto a fixed labelmap and save the result.

    Parameters
    ----------
    fixed_seg_path : str
        Path to the fixed reference segmentation.
    moving_seg_path : str
        Path to the moving segmentation to register.
    out_dir : str
        Output directory (created if it does not exist).
    moving_im_path : str, optional
        Path to the moving image. If provided, the same transform is applied to it. Default None.
    filename : str
        Base name for output files (e.g. ``"registered"``).
    clean : bool
        If True, apply segmentation cleaning after registration. Default True.
    min_cluster_size : int
        Minimum voxel count to keep a connected component (used if clean=True). Default 5.
    gaussian_sigma : float
        Sigma for Gaussian smoothing (used if clean=True). Default 1.5.
    sensitive_class : list of int, optional
        Label values for which binary opening is skipped during cleaning. Default None.

    Returns
    -------
    None
    """
    if sensitive_class is None:
        sensitive_class = []

    os.makedirs(out_dir, exist_ok=True)

    fixed = ants.image_read(fixed_seg_path)
    moving = ants.image_read(moving_seg_path)

    tx_rigid = ants.registration(fixed=fixed,moving=moving,type_of_transform="Rigid",verbose=False,)

    tx_affine = ants.registration(fixed=fixed,moving=moving,initial_transform=tx_rigid["fwdtransforms"][0],type_of_transform="Affine",verbose=False,)

    registered_label = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=tx_affine["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    
    if moving_im_path is not None:
        moving_im = ants.image_read(moving_im_path)
        registered_im = ants.apply_transforms(
            fixed=fixed,
            moving=moving_im,
            transformlist=tx_affine["fwdtransforms"],
        )
        registered_im.to_file(os.path.join(out_dir, filename + '_im.nii.gz'))

    registered_label.to_file("moving_registered.nii")
    registered_sitk = sitk.ReadImage("moving_registered.nii")

    if clean:
        result = clean_segmentation_sitk_parallel(
            registered_sitk,
            min_cluster_size=min_cluster_size,
            gaussian_sigma=gaussian_sigma,
            sensitive_class=sensitive_class,
        )
        sitk.WriteImage(result, os.path.join(out_dir, filename + '_seg.nii.gz'))
    else:
        sitk.WriteImage(registered_sitk, os.path.join(out_dir, filename+ '_seg.nii.gz'))

    os.remove("moving_registered.nii")
  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Register a moving labelmap onto a fixed labelmap and save the result."
    )
    parser.add_argument("--fixed_seg_path", required=True, help="Path to the fixed segmentation.")
    parser.add_argument("--moving_seg_path", required=True, help="Path to the moving segmentation.")
    parser.add_argument("--output_path", required=True, help="Output directory.")
    parser.add_argument("--moving_im_path", required=False, help="Path to the moving image.")
    parser.add_argument("--output_filename", required=False, default="registered_segm.nii.gz", help="Output filename. Default: registered_segm.nii.gz",)
    parser.add_argument("--clean",required=False,default=True,type=lambda x: x.lower() == "true",help="Clean the segmentation after registration. Default: True",)
    parser.add_argument( "--min_cluster_size",required=False,default=5,type=int, help="Remove connected components smaller than this voxel count. Default: 5",)
    parser.add_argument("--gaussian_sigma",required=False,default=1.5,type=float,help="Sigma of the Gaussian filter applied during cleaning. Default: 1.5",)
    parser.add_argument("--sensitive_class",required=False,default=[],type=int,help="Label values for which binary opening is skipped. Default: none",)
    args = parser.parse_args()

    labelmaps_registration(
        args.fixed_seg_path,
        args.moving_seg_path,
        args.output_path,
        args.moving_im_path,
        args.output_filename,
        clean=args.clean,
        min_cluster_size=args.min_cluster_size,
        gaussian_sigma=args.gaussian_sigma,
        sensitive_class=args.sensitive_class,
    )
