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
    fov_mask: Optional[np.ndarray] = None,
    use_otsu: bool = False,
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
    fov_mask : Optional[np.ndarray], default=None
        Binary field-of-view mask. If provided, eroded by 6% margin to suppress circular aperture step responses.
    use_otsu : bool, default=False
        Whether to apply Otsu adaptive binarization (restricted to inner FOV if fov_mask is provided).

    Returns
    -------
    vessel_map : np.ndarray
        Fused vascular response map (float32 in [0, 1] or uint8 binary if thresholded).
    e_horiz : np.ndarray
        Horizontal edge response from convolution with transpose A^T (detecting horizontal vessels).
    e_vert : np.ndarray
        Vertical edge response from convolution with matrix A (detecting vertical vessels).
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
    kernel_a = BEZIER_KERNEL_A
    kernel_a_t = BEZIER_KERNEL_A.T

    resp_vert_edges = cv2.filter2D(enhanced, cv2.CV_32F, kernel_a)
    resp_horiz_edges = cv2.filter2D(enhanced, cv2.CV_32F, kernel_a_t)

    # --- Response Fusion & Normalization ---
    fused_response = resp_vert_edges + resp_horiz_edges
    fused_response = np.maximum(0.0, fused_response)

    if fov_mask is not None:
        h, w = fused_response.shape[:2]
        ksize = max(15, int(round(0.06 * min(h, w)))) | 1
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        eroded_fov = cv2.erode((fov_mask > 0).astype(np.uint8), kernel_erode)
        inside = eroded_fov > 0
        normalized_response = np.zeros_like(fused_response, dtype=np.float32)
        if np.any(inside):
            p995 = float(np.percentile(fused_response[inside], 99.5))
            norm_denom = p995 if p995 > 1e-6 else float(np.max(fused_response[inside]))
            normalized_response[inside] = np.clip(fused_response[inside] / max(norm_denom, 1e-6), 0.0, 1.0)
    else:
        max_val = np.max(fused_response)
        if max_val > 0:
            normalized_response = fused_response / max_val
        else:
            normalized_response = np.zeros_like(fused_response, dtype=np.float32)

    e_horiz = resp_horiz_edges
    e_vert = resp_vert_edges

    if use_otsu:
        norm_u8 = (normalized_response * 255.0).astype(np.uint8)
        if fov_mask is not None and np.any(inside):
            pixels_in = norm_u8[inside]
            otsu_th, _ = cv2.threshold(pixels_in, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_mask = np.zeros_like(norm_u8)
            binary_mask[inside] = (norm_u8[inside] >= otsu_th).astype(np.uint8) * 255
        else:
            otsu_th, binary_mask = cv2.threshold(norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_mask, e_horiz, e_vert

    if threshold is not None:
        binary_mask = (normalized_response >= threshold).astype(np.uint8) * 255
        if fov_mask is not None:
            binary_mask = binary_mask * (eroded_fov > 0).astype(np.uint8)
        return binary_mask, e_horiz, e_vert

    return normalized_response, e_horiz, e_vert
