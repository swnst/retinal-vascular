"""Unit tests for Bézier curve matched filter baseline (Kanjanasurat et al., 2020, IOP)."""

import numpy as np
from classical_pipeline.bezier_matched_filter import (
    get_bezier_kernel,
    extract_vessels_bezier,
)


def test_bezier_kernel_dimensions_and_values():
    kernel = get_bezier_kernel()
    assert kernel.shape == (3, 6)
    assert kernel.dtype == np.float32

    expected_row = np.array([3.0, -0.136, -2.856, -2.856, -0.136, 3.0], dtype=np.float32)
    for i in range(3):
        np.testing.assert_allclose(kernel[i], expected_row, rtol=1e-5)


def test_bezier_kernel_zero_mean_property():
    kernel = get_bezier_kernel()
    row_sums = np.sum(kernel, axis=1)
    for s in row_sums:
        assert abs(s - 0.016) < 1e-4
    assert abs(np.mean(kernel)) < 0.01


def test_extract_vessels_bezier_synthetic_rgb():
    np.random.seed(42)
    synthetic_fundus = np.full((128, 128, 3), 100, dtype=np.uint8)
    synthetic_fundus[60:64, :, 1] = 40
    synthetic_fundus[:, 60:64, 1] = 40

    vessel_map, e_horiz, e_vert = extract_vessels_bezier(synthetic_fundus)

    assert vessel_map.shape == (128, 128)
    assert vessel_map.dtype == np.float32
    assert e_horiz.shape == (128, 128)
    assert e_vert.shape == (128, 128)
    assert np.min(vessel_map) >= 0.0
    assert np.max(vessel_map) <= 1.0


def test_extract_vessels_bezier_thresholding():
    np.random.seed(42)
    synthetic_gray = np.full((100, 100), 120, dtype=np.uint8)
    synthetic_gray[48:52, :] = 30

    binary_mask, e_horiz, e_vert = extract_vessels_bezier(synthetic_gray, threshold=0.5)

    assert binary_mask.shape == (100, 100)
    assert binary_mask.dtype == np.uint8
    unique_vals = np.unique(binary_mask)
    assert set(unique_vals).issubset({0, 255})


def test_extract_vessels_bezier_with_fov_mask_and_otsu():
    np.random.seed(42)
    synthetic_rgb = np.full((120, 120, 3), 100, dtype=np.uint8)
    synthetic_rgb[55:60, :, 1] = 40
    fov_mask = np.zeros((120, 120), dtype=np.uint8)
    fov_mask[10:110, 10:110] = 255

    otsu_mask, e_horiz, e_vert = extract_vessels_bezier(synthetic_rgb, fov_mask=fov_mask, use_otsu=True)

    assert otsu_mask.shape == (120, 120)
    assert otsu_mask.dtype == np.uint8
    unique_vals = np.unique(otsu_mask)
    assert set(unique_vals).issubset({0, 255})
    assert np.all(otsu_mask[:10, :] == 0)
    assert np.all(otsu_mask[110:, :] == 0)
