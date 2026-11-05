import cv2
import numpy as np


# --- Morphological Post-processing ---
def remove_small_components(binary_mask: np.ndarray, min_size: int = 25) -> np.ndarray:
    if binary_mask.size == 0:
        return binary_mask.copy()

    orig_dtype = binary_mask.dtype
    fg_val = 255 if binary_mask.max() > 1 else 1
    mask_u8 = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        out = np.zeros_like(mask_u8)
    else:
        keep = stats[:, cv2.CC_STAT_AREA] >= min_size
        keep[0] = False
        out = (keep[labels].astype(np.uint8)) * fg_val

    if orig_dtype == np.bool_:
        return out.astype(bool)
    return out.astype(orig_dtype)


def fill_holes(binary_mask: np.ndarray, max_hole_size: int = 15) -> np.ndarray:
    if binary_mask.size == 0:
        return binary_mask.copy()

    orig_dtype = binary_mask.dtype
    fg_val = 255 if binary_mask.max() > 1 else 1
    mask_u8 = ((binary_mask > 0).astype(np.uint8)) * fg_val

    h, w = mask_u8.shape[:2]
    inverted = (mask_u8 == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=4)
    if num_labels <= 1:
        if orig_dtype == np.bool_:
            return (mask_u8 > 0).astype(bool)
        return mask_u8.astype(orig_dtype)

    touches_border = (
        (stats[:, cv2.CC_STAT_LEFT] == 0)
        | (stats[:, cv2.CC_STAT_TOP] == 0)
        | (stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] >= w)
        | (stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] >= h)
    )

    fill = (stats[:, cv2.CC_STAT_AREA] <= max_hole_size) & (~touches_border)
    fill[0] = False
    filled_mask = mask_u8.copy()
    filled_mask[fill[labels]] = fg_val

    if orig_dtype == np.bool_:
        return (filled_mask > 0).astype(bool)
    return filled_mask.astype(orig_dtype)


def refine_vessel_mask(
    binary_mask: np.ndarray, min_size: int = 25, max_hole_size: int = 15
) -> np.ndarray:
    cleaned = remove_small_components(binary_mask, min_size=min_size)
    filled = fill_holes(cleaned, max_hole_size=max_hole_size)
    return filled
