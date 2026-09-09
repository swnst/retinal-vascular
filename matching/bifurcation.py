import math
import numpy as np
from scipy import ndimage


# --- Vascular Landmark Detection ---
def _get_crossing_number_map(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if skeleton.ndim != 2:
        raise ValueError(f"Expected 2D skeleton array, got shape: {skeleton.shape}")

    skel_binary = (skeleton > 0).astype(np.uint8)

    # 3x3 kernel count excluding center
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndimage.convolve(skel_binary, kernel, mode="constant", cval=0)
    neighbor_count = neighbor_count * skel_binary

    # Rutovitz Crossing Number: CN = 0.5 * sum(|P_i - P_{i+1}|) along 8-neighborhood cycle
    # Ordered 8-neighbors: P1=(0,1), P2=(-1,1), P3=(-1,0), P4=(-1,-1), P5=(0,-1), P6=(1,-1), P7=(1,0), P8=(1,1)
    p = [
        np.roll(np.roll(skel_binary, 0, axis=0), 1, axis=1),    # (0, 1)
        np.roll(np.roll(skel_binary, -1, axis=0), 1, axis=1),   # (-1, 1)
        np.roll(np.roll(skel_binary, -1, axis=0), 0, axis=1),   # (-1, 0)
        np.roll(np.roll(skel_binary, -1, axis=0), -1, axis=1),  # (-1, -1)
        np.roll(np.roll(skel_binary, 0, axis=0), -1, axis=1),   # (0, -1)
        np.roll(np.roll(skel_binary, 1, axis=0), -1, axis=1),   # (1, -1)
        np.roll(np.roll(skel_binary, 1, axis=0), 0, axis=1),    # (1, 0)
        np.roll(np.roll(skel_binary, 1, axis=0), 1, axis=1),    # (1, 1)
    ]

    diff_sum = np.zeros_like(skel_binary, dtype=np.float32)
    for i in range(8):
        diff_sum += np.abs(p[i].astype(np.float32) - p[(i + 1) % 8].astype(np.float32))

    crossing_number = (0.5 * diff_sum) * skel_binary
    return neighbor_count, crossing_number


def detect_bifurcations(
    skeleton: np.ndarray, border_margin: int = 20
) -> list[tuple[float, float]]:
    neighbor_count, crossing_number = _get_crossing_number_map(skeleton)
    h, w = skeleton.shape

    # Bifurcation: on skeleton, crossing_number >= 3 or neighbor_count >= 3
    is_bifurcation = np.logical_and(skeleton > 0, crossing_number >= 3)
    y_coords, x_coords = np.where(is_bifurcation)

    points: list[tuple[float, float]] = []
    for y, x in zip(y_coords, x_coords):
        if border_margin <= x < w - border_margin and border_margin <= y < h - border_margin:
            points.append((float(x), float(y)))

    return _suppress_close_points(points, min_distance=8.0)


def detect_crossovers(
    skeleton: np.ndarray, border_margin: int = 20
) -> list[tuple[float, float]]:
    neighbor_count, crossing_number = _get_crossing_number_map(skeleton)
    h, w = skeleton.shape

    # Crossover: on skeleton, crossing_number >= 4
    is_crossover = np.logical_and(skeleton > 0, crossing_number >= 4)
    y_coords, x_coords = np.where(is_crossover)

    points: list[tuple[float, float]] = []
    for y, x in zip(y_coords, x_coords):
        if border_margin <= x < w - border_margin and border_margin <= y < h - border_margin:
            points.append((float(x), float(y)))

    return _suppress_close_points(points, min_distance=8.0)


def filter_points_around_od(
    points: list[tuple[float, float]],
    od_center: tuple[float, float],
    min_radius: float = 30.0,
    max_radius: float = 450.0,
) -> list[tuple[float, float]]:
    od_x, od_y = od_center
    filtered: list[tuple[float, float]] = []

    for x, y in points:
        dist = math.hypot(x - od_x, y - od_y)
        if min_radius <= dist <= max_radius:
            filtered.append((x, y))

    return filtered


def _suppress_close_points(
    points: list[tuple[float, float]], min_distance: float = 8.0
) -> list[tuple[float, float]]:
    if not points:
        return []

    accepted: list[tuple[float, float]] = []
    for pt in points:
        x, y = pt
        too_close = False
        for ax, ay in accepted:
            if math.hypot(x - ax, y - ay) < min_distance:
                too_close = True
                break
        if not too_close:
            accepted.append(pt)

    return accepted
