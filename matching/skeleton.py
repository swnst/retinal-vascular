import numpy as np
from skimage.morphology import skeletonize

from classical_pipeline.skeleton_metrics import prune_short_spurs


# --- Retinal Vessel Skeletonization Algorithms ---
def skeletonize_zhang_suen(binary_mask: np.ndarray) -> np.ndarray:
    """
    Canonical parallel thinning algorithm (Zhang & Suen, 1984).
    Iteratively strips contour pixels using alternating 8-neighborhood deletion conditions.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape: {binary_mask.shape}")
    bool_mask = binary_mask > 0
    return skeletonize(bool_mask, method="zhang").astype(np.uint8)


def skeletonize_vessel(
    binary_mask: np.ndarray,
    method: str = "zhang",
    prune_spurs: bool = True,
    spur_length_threshold: int = 8,
) -> np.ndarray:
    """
    Skeletonizes binary vessel mask with optional spur pruning.
    Supports canonical Zhang-Suen ('zhang') and 3D/2D medial axis thinning ('lee').
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape: {binary_mask.shape}")

    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask, method=method)
    if prune_spurs and spur_length_threshold > 0:
        skeleton = prune_short_spurs(skeleton, threshold=spur_length_threshold)
    return skeleton.astype(np.uint8)
