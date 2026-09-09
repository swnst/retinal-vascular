import cv2
import numpy as np
from scipy import ndimage


# --- Morphological Post-processing ---
def remove_small_components(binary_mask: np.ndarray, min_size: int = 30) -> np.ndarray:
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    clean_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            clean_mask[labels == i] = 1

    return clean_mask


def fill_holes(binary_mask: np.ndarray, max_hole_size: int = 20) -> np.ndarray:
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0).astype(np.uint8)

    inverted = (binary_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=4)
    filled_mask = binary_mask.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_hole_size:
            filled_mask[labels == i] = 1

    return filled_mask


def refine_vessel_mask(
    binary_mask: np.ndarray, min_size: int = 30, max_hole_size: int = 20
) -> np.ndarray:
    cleaned = remove_small_components(binary_mask, min_size=min_size)
    filled = fill_holes(cleaned, max_hole_size=max_hole_size)
    return filled
