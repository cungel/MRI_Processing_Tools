import numpy as np
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, gaussian_filter, label


def apply_cleaning_logic(mask_np, lbl, min_cluster_size=5, gaussian_sigma=1.5, sensitive_class=None):
    """
    Clean a single binary label mask: fill holes, remove small clusters,
    apply closing/opening and Gaussian smoothing.

    Parameters
    ----------
    mask_np : np.ndarray
        Binary mask for one label.
    lbl : int
        Label value.
    min_cluster_size : int
        Minimum voxel count to keep a connected component. Default 5.
    gaussian_sigma : float
        Sigma for Gaussian smoothing. Default 1.5.
    sensitive_class : list of int, optional
        Labels for which binary opening is skipped. Default None.

    Returns
    -------
    mask : np.ndarray
        Cleaned binary mask.
    lbl : int
        The input label value (passed through for parallel result mapping).
    """
    if sensitive_class is None:
        sensitive_class = []

    mask = mask_np.copy()
    mask = binary_fill_holes(mask)

    labeled, num = label(mask)
    for cc in range(1, num + 1):
        cluster = labeled == cc
        if np.sum(cluster) < min_cluster_size:
            mask[cluster] = False

    mask = binary_closing(mask, iterations=1)

    if lbl not in sensitive_class:
        mask = binary_opening(mask, iterations=1)

    dist = gaussian_filter(mask.astype(float), sigma=gaussian_sigma)
    mask = dist > 0.5
    mask = binary_fill_holes(mask)

    return mask, lbl


def clean_segmentation_sitk_parallel(img_sitk, min_cluster_size=5, gaussian_sigma=1.5, sensitive_class=None):
    """
    Clean all labels in a segmentation in parallel.

    Parameters
    ----------
    img_sitk : sitk.Image
        Input segmentation.
    min_cluster_size : int
        Minimum voxel count to keep a connected component. Default 5.
    gaussian_sigma : float
        Sigma for Gaussian smoothing. Default 1.5.
    sensitive_class : list of int, optional
        Labels for which binary opening is skipped. Default None.

    Returns
    -------
    sitk.Image
        Cleaned segmentation with the same spatial metadata as the input.
    """
    if sensitive_class is None:
        sensitive_class = []

    full_arr = sitk.GetArrayFromImage(img_sitk)
    unique_labels = np.unique(full_arr)
    unique_labels = unique_labels[unique_labels != 0]

    results = Parallel(n_jobs=-1)(
        delayed(apply_cleaning_logic)(
            full_arr == lbl, lbl, min_cluster_size, gaussian_sigma, sensitive_class
        )
        for lbl in unique_labels
    )

    final_cleaned_arr = np.zeros_like(full_arr, dtype=np.uint8)
    for cleaned_mask, lbl in results:
        final_cleaned_arr[cleaned_mask] = int(lbl)

    cleaned_img = sitk.GetImageFromArray(final_cleaned_arr)
    cleaned_img.CopyInformation(img_sitk)
    return cleaned_img
