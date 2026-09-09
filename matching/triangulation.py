import math
from typing import Any
import numpy as np
from scipy.spatial import Delaunay


# --- Delaunay Triangulation & Invariant Feature Extraction ---
def triangle_area(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
    # 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def build_delaunay_triangulation(
    points: list[tuple[float, float]],
) -> tuple[Delaunay | None, np.ndarray]:
    if len(points) < 4:
        return None, np.array([])

    pts_array = np.array(points, dtype=np.float64)

    # Check for collinearity or near-zero variance
    if np.var(pts_array[:, 0]) < 1e-4 or np.var(pts_array[:, 1]) < 1e-4:
        return None, np.array([])

    try:
        tri = Delaunay(pts_array)
        return tri, pts_array
    except Exception:
        return None, np.array([])


def triangle_internal_angles(
    p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]
) -> list[float]:
    # Compute 3 internal angles in radians, sorted
    a = math.hypot(p2[0] - p3[0], p2[1] - p3[1])
    b = math.hypot(p1[0] - p3[0], p1[1] - p3[1])
    c = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    if a < 1e-4 or b < 1e-4 or c < 1e-4:
        return [0.0, 0.0, 0.0]

    cos_alpha = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)
    cos_beta = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0)
    cos_gamma = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)

    angles = [float(math.acos(cos_alpha)), float(math.acos(cos_beta)), float(math.acos(cos_gamma))]
    angles.sort()
    return angles


def compute_triangle_invariants(
    tri: Delaunay, pts_array: np.ndarray, min_area: float = 10.0
) -> list[dict[str, Any]]:
    # Multi-attribute Invariant Descriptors based on Kanjanasurat et al. (2020)
    simplices = tri.simplices
    neighbors = tri.neighbors
    n_triangles = len(simplices)

    areas: list[float] = []
    centroids: list[tuple[float, float]] = []
    angles_list: list[list[float]] = []

    for simplex in simplices:
        p1 = (float(pts_array[simplex[0]][0]), float(pts_array[simplex[0]][1]))
        p2 = (float(pts_array[simplex[1]][0]), float(pts_array[simplex[1]][1]))
        p3 = (float(pts_array[simplex[2]][0]), float(pts_array[simplex[2]][1]))
        area = triangle_area(p1, p2, p3)
        areas.append(area)
        cx = float((p1[0] + p2[0] + p3[0]) / 3.0)
        cy = float((p1[1] + p2[1] + p3[1]) / 3.0)
        centroids.append((cx, cy))
        angles_list.append(triangle_internal_angles(p1, p2, p3))

    triangle_features: list[dict[str, Any]] = []

    for i in range(n_triangles):
        area_i = areas[i]
        if area_i < min_area:
            continue

        adjacent_ratios: list[float] = []
        for adj_idx in neighbors[i]:
            if adj_idx != -1 and areas[adj_idx] >= min_area:
                r = float(min(area_i, areas[adj_idx]) / max(area_i, areas[adj_idx]))
                adjacent_ratios.append(r)

        if adjacent_ratios:
            adjacent_ratios.sort()
            triangle_features.append(
                {
                    "triangle_idx": i,
                    "simplex": simplices[i].tolist(),
                    "centroid": centroids[i],
                    "area": area_i,
                    "angles": angles_list[i],
                    "adjacent_ratios": adjacent_ratios,
                }
            )

    return triangle_features
