import numpy as np
import pytest

from matching.bifurcation import detect_bifurcations, detect_crossovers, filter_points_around_od
from matching.matcher import extract_identity_template, match_templates, verify_identity
from matching.skeleton import skeletonize_vessel, skeletonize_zhang_suen
from matching.triangulation import build_delaunay_triangulation, compute_triangle_invariants


# --- Unit Tests for Matching Pipeline ---
def test_skeletonize_vessel():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 25] = 1  # Vertical line width 1
    skel = skeletonize_vessel(mask)
    assert skel.shape == mask.shape
    assert skel.sum() > 0

    skel_zhang = skeletonize_zhang_suen(mask)
    assert skel_zhang.shape == mask.shape
    assert skel_zhang.sum() > 0

    skel_lee = skeletonize_vessel(mask, method="lee")
    assert skel_lee.shape == mask.shape
    assert skel_lee.sum() > 0


def test_bifurcation_detection_t_junction():
    # Create synthetic T-junction skeleton
    skel = np.zeros((50, 50), dtype=np.uint8)
    skel[25, 10:40] = 1  # Horizontal line
    skel[25:45, 25] = 1  # Vertical branch downwards from (25, 25)

    bifurcations = detect_bifurcations(skel, border_margin=5)
    assert len(bifurcations) >= 1
    # Check that junction at (25, 25) is detected
    found_center = any(abs(x - 25) <= 2 and abs(y - 25) <= 2 for x, y in bifurcations)
    assert found_center


def test_delaunay_and_invariants():
    # 4 non-collinear points
    points = [(50.0, 50.0), (150.0, 50.0), (100.0, 150.0), (100.0, 100.0)]
    tri, pts_array = build_delaunay_triangulation(points)
    assert tri is not None
    assert len(pts_array) == 4

    invariants = compute_triangle_invariants(tri, pts_array, min_area=5.0)
    assert len(invariants) > 0
    for inv in invariants:
        assert "adjacent_ratios" in inv
        assert len(inv["adjacent_ratios"]) > 0


def test_template_matching_self_match():
    # Synthetic 2D vascular network with multiple branches across y and x
    mask = np.zeros((160, 160), dtype=np.uint8)
    # Upper horizontal trunk and branches
    mask[45, 30:130] = 1
    mask[25:70, 50] = 1
    mask[25:70, 110] = 1

    # Lower horizontal trunk and branches
    mask[115, 30:130] = 1
    mask[90:135, 50] = 1
    mask[90:135, 110] = 1

    # Vertical cross connection
    mask[45:115, 80] = 1

    template_a = extract_identity_template(mask, od_center=(80.0, 80.0), use_od_filtering=False)
    template_b = extract_identity_template(mask, od_center=(80.0, 80.0), use_od_filtering=False)

    assert template_a["delaunay_triangles"] > 0
    score = match_templates(template_a, template_b)
    assert score >= 0.80
    assert verify_identity(score, threshold=0.70)
