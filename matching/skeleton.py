import numpy as np
from skimage.morphology import skeletonize


# --- Retinal Vessel Skeletonization ---
def skeletonize_vessel(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape: {binary_mask.shape}")

    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    return skeleton.astype(np.uint8)
