import numpy as np

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
