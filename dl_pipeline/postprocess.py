import cv2
import numpy as np


# --- Morphological Post-processing ---
def remove_small_components(binary_mask: np.ndarray, min_size: int = 25) -> np.ndarray:
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    keep = stats[:, cv2.CC_STAT_AREA] >= min_size
    keep[0] = False
    return keep[labels].astype(np.uint8)


def fill_holes(binary_mask: np.ndarray, max_hole_size: int = 15) -> np.ndarray:
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0).astype(np.uint8)

    h, w = binary_mask.shape[:2]
    inverted = (binary_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=4)
    if num_labels <= 1:
        return binary_mask.copy()

    touches_border = (
        (stats[:, cv2.CC_STAT_LEFT] == 0)
        | (stats[:, cv2.CC_STAT_TOP] == 0)
        | (stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] >= w)
        | (stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] >= h)
    )

    fill = (stats[:, cv2.CC_STAT_AREA] <= max_hole_size) & (~touches_border)
    fill[0] = False
    filled_mask = binary_mask.copy()
    filled_mask[fill[labels]] = 1
    return filled_mask


def refine_vessel_mask(
    binary_mask: np.ndarray, min_size: int = 25, max_hole_size: int = 15
) -> np.ndarray:
    cleaned = remove_small_components(binary_mask, min_size=min_size)
    filled = fill_holes(cleaned, max_hole_size=max_hole_size)
    return filled
