import math
from typing import Any

import numpy as np

from matching.bifurcation import detect_bifurcations, detect_crossovers, filter_points_around_od
from matching.skeleton import skeletonize_vessel
from matching.triangulation import build_delaunay_triangulation, compute_triangle_invariants


# --- Identity Template Extraction & Matching ---
def extract_identity_template(
    vessel_mask: np.ndarray,
    od_center: tuple[float, float] | None = None,
    fovea_center: tuple[float, float] | None = None,
    use_od_filtering: bool = True,
) -> dict[str, Any]:
    skeleton = skeletonize_vessel(vessel_mask)

    raw_bifurcations = detect_bifurcations(skeleton)
    crossovers = detect_crossovers(skeleton)

    if use_od_filtering and od_center is not None:
        keypoints = filter_points_around_od(raw_bifurcations, od_center, min_radius=30.0, max_radius=500.0)
        # Fallback if too few keypoints around OD
        if len(keypoints) < 6:
            keypoints = raw_bifurcations
    else:
        keypoints = raw_bifurcations

    tri, pts_array = build_delaunay_triangulation(keypoints)
    invariants = compute_triangle_invariants(tri, pts_array) if tri is not None else []

    return {
        "od_center": od_center,
        "fovea_center": fovea_center,
        "keypoints": keypoints,
        "crossovers": crossovers,
        "num_keypoints": len(keypoints),
        "delaunay_triangles": len(invariants),
        "invariants": invariants,
    }


def match_templates(
    template_a: dict[str, Any],
    template_b: dict[str, Any],
    ratio_tolerance: float = 0.05,
    angle_tolerance_rad: float = 0.08,
) -> float:
    # Rigorous Multi-Attribute Geometric Invariant Matching
    # (Kanjanasurat et al., 2020 Area Ratio + Delaunay Angle Preservation)
    invariants_a = template_a.get("invariants", [])
    invariants_b = template_b.get("invariants", [])

    if not invariants_a or not invariants_b:
        return 0.0

    matched_pairs = 0

    for tri_a in invariants_a:
        ratios_a = tri_a["adjacent_ratios"]
        angles_a = tri_a.get("angles", [])
        best_cost = float("inf")

        for tri_b in invariants_b:
            ratios_b = tri_b["adjacent_ratios"]
            angles_b = tri_b.get("angles", [])

            min_len = min(len(ratios_a), len(ratios_b))
            if min_len == 0:
                continue

            ratio_diff = sum(abs(ratios_a[k] - ratios_b[k]) for k in range(min_len)) / min_len
            if ratio_diff > ratio_tolerance:
                continue

            # Check angular shape similarity
            if angles_a and angles_b:
                angle_diff = sum(abs(angles_a[m] - angles_b[m]) for m in range(3)) / 3.0
                if angle_diff > angle_tolerance_rad:
                    continue
            else:
                angle_diff = 0.0

            cost = 0.6 * ratio_diff + 0.4 * angle_diff
            if cost < best_cost:
                best_cost = cost

        if best_cost <= ratio_tolerance:
            matched_pairs += 1

    # Symmetric Jaccard-like score
    score = float(matched_pairs / max(len(invariants_a), len(invariants_b)))
    return min(1.0, max(0.0, score))


def verify_identity(score: float, threshold: float = 0.20) -> bool:
    return bool(score >= threshold)
