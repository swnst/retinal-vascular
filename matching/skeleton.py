import numpy as np
from skimage.morphology import skeletonize


from classical_pipeline.skeleton_metrics import prune_short_spurs


# --- Retinal Vessel Skeletonization ---
def skeletonize_vessel(
    binary_mask: np.ndarray,
    prune_spurs: bool = True,
    spur_length_threshold: int = 8,
) -> np.ndarray:
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape: {binary_mask.shape}")

    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    if prune_spurs and spur_length_threshold > 0:
        skeleton = prune_short_spurs(skeleton, threshold=spur_length_threshold)
    return skeleton.astype(np.uint8)
