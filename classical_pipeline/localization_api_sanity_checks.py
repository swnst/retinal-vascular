from __future__ import annotations

import numpy as np

try:
    from .localization_api import find_fovea, find_optic_disk
except ImportError:
    from localization_api import find_fovea, find_optic_disk


def synthetic_case():
    height, width = 150, 230
    od_xy = (72.0, 63.0)
    fovea_xy = (142.0, 72.0)
    y_grid, x_grid = np.indices((height, width))
    fov = (
        ((x_grid - width / 2.0) / 108.0) ** 2
        + ((y_grid - height / 2.0) / 68.0) ** 2
    ) <= 1.0

    vessel = np.zeros((height, width), dtype=bool)
    for angle in np.linspace(0.0, np.pi, 10, endpoint=False):
        for radius in range(-55, 56):
            x = int(round(od_xy[0] + np.cos(angle) * radius))
            y = int(round(od_xy[1] + np.sin(angle) * radius))
            if 0 <= x < width and 0 <= y < height and fov[y, x]:
                vessel[max(0, y - 1): min(height, y + 2), max(0, x - 1): min(width, x + 2)] = True

    od_spot = np.exp(
        -((x_grid - od_xy[0]) ** 2 + (y_grid - od_xy[1]) ** 2) / (2 * 10.0 ** 2)
    )
    fovea_spot = np.exp(
        -((x_grid - fovea_xy[0]) ** 2 + (y_grid - fovea_xy[1]) ** 2) / (2 * 8.0 ** 2)
    )
    green = (90.0 + 125.0 * od_spot - 55.0 * fovea_spot).clip(0, 255).astype(np.uint8)
    green[~fov] = 0
    vessel[~fov] = False
    return vessel, green


def assert_point_inside(point_xy, shape):
    assert len(point_xy) == 2
    x, y = point_xy
    assert np.isfinite(x) and np.isfinite(y)
    assert 0.0 <= x < shape[1]
    assert 0.0 <= y < shape[0]


def assert_raises_clearly(callback, expected_text):
    try:
        callback()
    except (TypeError, ValueError) as exc:
        assert expected_text in str(exc)
        return
    raise AssertionError(f"Expected failure containing: {expected_text}")


def run_checks():
    vessel_bool, green = synthetic_case()
    variants = [
        vessel_bool,
        vessel_bool.astype(np.uint8),
        vessel_bool.astype(np.uint8) * 255,
    ]
    od_points = []
    fovea_points = []

    for vessel in variants:
        vessel_before = vessel.copy()
        green_before = green.copy()
        od_xy = find_optic_disk(vessel, green)
        fovea_xy = find_fovea(vessel, od_xy, green_image=green)
        assert_point_inside(od_xy, vessel.shape)
        assert_point_inside(fovea_xy, vessel.shape)
        assert np.array_equal(vessel, vessel_before)
        assert np.array_equal(green, green_before)
        od_points.append(od_xy)
        fovea_points.append(fovea_xy)

    assert np.allclose(od_points, od_points[0], atol=1e-9)
    assert np.allclose(fovea_points, fovea_points[0], atol=1e-9)
    assert abs(od_points[0][0] - 72.0) < 10.0
    assert abs(od_points[0][1] - 63.0) < 10.0
    assert abs(fovea_points[0][0] - 142.0) < 12.0
    assert abs(fovea_points[0][1] - 72.0) < 12.0

    normalized_green = green.astype(np.float32) / 255.0
    normalized_od = find_optic_disk(vessel_bool, normalized_green)
    normalized_fovea = find_fovea(
        vessel_bool,
        normalized_od,
        green_image=normalized_green,
    )
    assert np.allclose(normalized_od, od_points[0], atol=1e-9)
    assert np.allclose(normalized_fovea, fovea_points[0], atol=1e-9)

    assert_raises_clearly(
        lambda: find_optic_disk(vessel_bool[:-1], green),
        "does not match",
    )
    assert_raises_clearly(
        lambda: find_fovea(vessel_bool, od_points[0]),
        "green_image is required",
    )
    assert_raises_clearly(
        lambda: find_fovea(
            vessel_bool,
            od_points[0],
            green_image=green[:-1],
        ),
        "does not match",
    )
    assert_raises_clearly(
        lambda: find_fovea(vessel_bool, (-1.0, 20.0), green_image=green),
        "outside image bounds",
    )
    probability_map = vessel_bool.astype(np.float32) * 0.8
    assert_raises_clearly(
        lambda: find_optic_disk(probability_map, green),
        "threshold probability maps",
    )
    assert_raises_clearly(
        lambda: find_optic_disk(np.zeros_like(vessel_bool), green),
        "no vessel pixels",
    )
    assert_raises_clearly(
        lambda: find_fovea(
            np.ones_like(vessel_bool),
            od_points[0],
            green_image=green,
        ),
        "every pixel as vessel",
    )
    assert_raises_clearly(
        lambda: find_optic_disk(vessel_bool, green.astype(np.float32) - 128.0),
        "non-negative",
    )

    read_only_vessel = vessel_bool.copy()
    read_only_green = green.copy()
    read_only_vessel.setflags(write=False)
    read_only_green.setflags(write=False)
    read_only_od = find_optic_disk(read_only_vessel, read_only_green)
    read_only_fovea = find_fovea(
        read_only_vessel,
        read_only_od,
        green_image=read_only_green,
    )
    assert_point_inside(read_only_od, vessel_bool.shape)
    assert_point_inside(read_only_fovea, vessel_bool.shape)

    print("Localization API sanity checks passed.")


if __name__ == "__main__":
    run_checks()
