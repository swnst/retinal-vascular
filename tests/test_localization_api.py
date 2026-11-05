import numpy as np
import pytest

from classical_pipeline.localization_api import find_fovea, find_optic_disk


# --- Tests for Classical Localization API ---
def test_find_optic_disk_and_fovea_synthetic():
    h, w = 200, 200
    green = np.full((h, w), 128, dtype=np.uint8)
    y, x = np.ogrid[:h, :w]
    od_mask = ((x - 50) ** 2 + (y - 100) ** 2) <= 15**2
    green[od_mask] = 250

    fov_spot = ((x - 130) ** 2 + (y - 100) ** 2) <= 10**2
    green[fov_spot] = 20

    vessel = np.zeros((h, w), dtype=np.uint8)
    vessel[100, 30:170] = 1
    vessel[80:120, 50] = 1

    od_x, od_y = find_optic_disk(vessel, green)
    assert 0 <= od_x < w
    assert 0 <= od_y < h

    fovea_x, fovea_y = find_fovea(vessel, (od_x, od_y), green)
    assert 0 <= fovea_x < w
    assert 0 <= fovea_y < h


def test_find_optic_disk_rgb_input():
    h, w = 180, 180
    color = np.full((h, w, 3), 100, dtype=np.uint8)
    color[80:100, 80:100] = 250

    vessel = np.zeros((h, w), dtype=np.uint8)
    vessel[90, 40:140] = 1

    od_x, od_y = find_optic_disk(vessel, color)
    assert 0 <= od_x < w
    assert 0 <= od_y < h


def test_find_optic_disk_uint16_rgb_over_255():
    h, w = 180, 180
    color_u16 = np.full((h, w, 3), 10000, dtype=np.uint16)
    color_u16[80:100, 80:100] = 55000

    vessel = np.zeros((h, w), dtype=np.uint8)
    vessel[90, 40:140] = 1

    od_x, od_y = find_optic_disk(vessel, color_u16)
    assert 0 <= od_x < w
    assert 0 <= od_y < h


def test_find_optic_disk_additional_dtypes():
    h, w = 150, 150
    vessel = np.zeros((h, w), dtype=np.uint8)
    vessel[75, 20:130] = 1

    gray_u16_high = np.full((h, w), 50000, dtype=np.uint16)
    gray_u16_high[70:80, 70:80] = 65000
    od_x1, od_y1 = find_optic_disk(vessel, gray_u16_high)
    assert 0 <= od_x1 < w
    assert 0 <= od_y1 < h

    f32_01 = np.full((h, w, 3), 0.4, dtype=np.float32)
    f32_01[70:80, 70:80] = 0.95
    od_x2, od_y2 = find_optic_disk(vessel, f32_01)
    assert 0 <= od_x2 < w
    assert 0 <= od_y2 < h

    bool_3d = np.ones((h, w, 3), dtype=np.bool_)
    with pytest.raises(TypeError):
        find_optic_disk(vessel, bool_3d)

    neg_float = np.full((h, w), -0.5, dtype=np.float32)
    with pytest.raises(ValueError):
        find_optic_disk(vessel, neg_float)
