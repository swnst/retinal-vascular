import numpy as np
import pytest

from dl_pipeline.postprocess import fill_holes, refine_vessel_mask, remove_small_components


# --- Test Morphological Post-processing ---
def test_remove_small_components_isolated_noise():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # วัตถุขนาดใหญ่ (พื้นที่ 100 พิกเซล)
    mask[10:20, 10:20] = 1
    # วัตถุขนาดเล็กขยะ (พื้นที่ 4 พิกเซล)
    mask[50:52, 50:52] = 1

    cleaned = remove_small_components(mask, min_size=20)
    assert cleaned[10:20, 10:20].all() == 1
    assert cleaned[50:52, 50:52].sum() == 0


def test_fill_holes_internal_gap():
    mask = np.ones((50, 50), dtype=np.uint8)
    # สร้างรูพรุนตรงกลางขนาด 3x3 (9 พิกเซล)
    mask[20:23, 20:23] = 0

    filled = fill_holes(mask, max_hole_size=15)
    assert (filled == 1).all()


def test_refine_vessel_mask_all_zeros_and_ones():
    zeros = np.zeros((64, 64), dtype=np.uint8)
    ones = np.ones((64, 64), dtype=np.uint8)

    refined_zeros = refine_vessel_mask(zeros)
    refined_ones = refine_vessel_mask(ones)

    assert refined_zeros.shape == zeros.shape
    assert (refined_zeros == 0).all()
    assert refined_ones.shape == ones.shape
    assert (refined_ones == 1).all()
