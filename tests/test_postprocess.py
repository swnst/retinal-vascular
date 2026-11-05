import numpy as np

from dl_pipeline.postprocess import fill_holes, refine_vessel_mask, remove_small_components


# --- Test Morphological Post-processing ---
def test_remove_small_components_isolated_noise():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[50:52, 50:52] = 1

    cleaned = remove_small_components(mask, min_size=20)
    assert cleaned[10:20, 10:20].all() == 1
    assert cleaned[50:52, 50:52].sum() == 0


def test_fill_holes_internal_gap():
    mask = np.ones((50, 50), dtype=np.uint8)
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


def test_fill_holes_dtypes_and_empty():
    mask_255 = np.zeros((50, 50), dtype=np.uint8)
    mask_255[10:40, 10:40] = 255
    mask_255[20:25, 20:25] = 0
    filled_255 = fill_holes(mask_255, max_hole_size=50)
    assert filled_255.dtype == np.uint8
    assert set(np.unique(filled_255)).issubset({0, 255})
    assert 1 not in set(np.unique(filled_255))

    mask_1 = (mask_255 > 0).astype(np.uint8)
    filled_1 = fill_holes(mask_1, max_hole_size=50)
    assert filled_1.dtype == np.uint8
    assert set(np.unique(filled_1)).issubset({0, 1})
    assert 255 not in set(np.unique(filled_1))

    mask_bool = (mask_255 > 0)
    filled_bool = fill_holes(mask_bool, max_hole_size=50)
    assert filled_bool.dtype == np.bool_
    assert filled_bool[22, 22] == True

    empty_arr = np.zeros((0, 0), dtype=np.uint8)
    filled_empty = fill_holes(empty_arr, max_hole_size=50)
    assert filled_empty.shape == (0, 0)
