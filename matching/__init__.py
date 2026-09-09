"""
Identity Matching Pipeline using Retinal Vascular Topology and Delaunay Triangulation.
Based on Kanjanasurat et al. (2020) with Deep Learning Vessel Extraction.
"""

from matching.bifurcation import detect_bifurcations, detect_crossovers, filter_points_around_od
from matching.matcher import extract_identity_template, match_templates, verify_identity
from matching.skeleton import skeletonize_vessel
from matching.triangulation import build_delaunay_triangulation, compute_triangle_invariants

__all__ = [
    "skeletonize_vessel",
    "detect_bifurcations",
    "detect_crossovers",
    "filter_points_around_od",
    "build_delaunay_triangulation",
    "compute_triangle_invariants",
    "extract_identity_template",
    "match_templates",
    "verify_identity",
]
