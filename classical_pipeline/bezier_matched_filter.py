"""Bézier curve matched filter baseline for retinal vascular extraction.

Reproduces the classical matched filter algorithm from:
Kanjanasurat, I., Purahong, B., Pintavirooj, C., and Benjangkaprasert, C. (2020).
Vascular Extraction by using matched filter on retinal image.
Journal of Physics: Conference Series, 1457(1), 012013.
DOI: 10.1088/1742-6596/1457/1/012013
"""

from typing import Tuple, Optional
import numpy as np
import cv2

# --- Bézier Convolution Kernel Definition ---

BEZIER_KERNEL_A = np.array(
    [
        [3.0, -0.136, -2.856, -2.856, -0.136, 3.0],
        [3.0, -0.136, -2.856, -2.856, -0.136, 3.0],
        [3.0, -0.136, -2.856, -2.856, -0.136, 3.0],
    ],
    dtype=np.float32,
)


def get_bezier_kernel() -> np.ndarray:
    """Returns a copy of the canonical 3x6 Bézier convolution matrix A."""
    return BEZIER_KERNEL_A.copy()


def extract_vessels_bezier(
    image: np.ndarray,
    gaussian_kernel_size: Tuple[int, int] = (5, 5),
    gaussian_sigma: float = 1.0,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: Tuple[int, int] = (8, 8),
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts retinal vasculature using Bézier curve matched filtering.

    Parameters
    ----------
    image : np.ndarray
        Input fundus image, either RGB (H, W, 3) or single-channel Green (H, W).
    gaussian_kernel_size : Tuple[int, int], default=(5, 5)
        Kernel size for initial noise cancellation Gaussian filter.
    gaussian_sigma : float, default=1.0
        Standard deviation for Gaussian smoothing.
    clahe_clip_limit : float, default=2.0
        Contrast limit threshold for CLAHE enhancement.
    clahe_grid_size : Tuple[int, int], default=(8, 8)
        Contextual tile grid size for CLAHE.
    threshold : Optional[float], default=None
        Binarization cutoff. If None, returns continuous normalized response.

    Returns
    -------
    vessel_map : np.ndarray
        Fused vascular response map (float32 in [0, 1] or uint8 binary if thresholded).
    e_horiz : np.ndarray
        Horizontal edge response from convolution with matrix A.
    e_vert : np.ndarray
        Vertical edge response from convolution with transpose A^T.
    """
    # --- Preprocessing & Spectral Channel Selection ---
    if image.ndim == 3:
        if image.shape[2] == 3:
            green_channel = image[:, :, 1]
        else:
            green_channel = image[:, :, 0]
    else:
        green_channel = image

    if green_channel.dtype != np.uint8:
        green_channel = np.clip(green_channel, 0, 255).astype(np.uint8)

    smoothed = cv2.GaussianBlur(green_channel, gaussian_kernel_size, gaussian_sigma)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    enhanced = clahe.apply(smoothed).astype(np.float32)

    # --- Directional Bézier Matched Filtering ---
    kernel_h = BEZIER_KERNEL_A
    kernel_v = BEZIER_KERNEL_A.T

    e_horiz = cv2.filter2D(enhanced, cv2.CV_32F, kernel_h)
    e_vert = cv2.filter2D(enhanced, cv2.CV_32F, kernel_v)

    # --- Response Fusion & Normalization ---
    fused_response = e_horiz + e_vert
    fused_response = np.maximum(0.0, fused_response)

    max_val = np.max(fused_response)
    if max_val > 0:
        normalized_response = fused_response / max_val
    else:
        normalized_response = np.zeros_like(fused_response, dtype=np.float32)

    if threshold is not None:
        binary_mask = (normalized_response >= threshold).astype(np.uint8) * 255
        return binary_mask, e_horiz, e_vert

    return normalized_response, e_horiz, e_vert
