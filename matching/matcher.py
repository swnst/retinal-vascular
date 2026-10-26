import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, Delaunay

from matching.bifurcation import detect_bifurcations, filter_points_around_od
from matching.skeleton import skeletonize_vessel
from matching.triangulation import (
    build_delaunay_triangulation,
    canonicalize_triangle_vertices,
    compute_triangle_invariants,
)


# --- Identity Template Extraction & Matching ---
def extract_identity_template(
    vessel_mask: np.ndarray,
    od_center: tuple[float, float] | None = None,
    fovea_center: tuple[float, float] | None = None,
    use_od_filtering: bool = True,
    prune_spurs: bool = True,
    skeleton: np.ndarray | None = None,
) -> dict[str, Any]:
    if skeleton is None:
        skeleton = skeletonize_vessel(vessel_mask, prune_spurs=prune_spurs)

    raw_bifurcations = detect_bifurcations(skeleton)
    crossovers: list[tuple[float, float]] = []

    if use_od_filtering and od_center is not None:
        keypoints = filter_points_around_od(
            raw_bifurcations,
            od_center,
            image_shape=vessel_mask.shape[:2],
        )
        if len(keypoints) < 6:
            keypoints = raw_bifurcations
    else:
        keypoints = raw_bifurcations

    h, w = vessel_mask.shape[:2]
    min_area = max(10.0, 0.00003 * float(h * w))
    tri, pts_array = build_delaunay_triangulation(keypoints)
    invariants = compute_triangle_invariants(tri, pts_array, min_area=min_area) if tri is not None else []

    return {
        "image_shape": (int(h), int(w)),
        "skeleton": skeleton,
        "od_center": od_center,
        "fovea_center": fovea_center,
        "keypoints": keypoints,
        "crossovers": crossovers,
        "num_keypoints": len(keypoints),
        "delaunay_triangles": len(invariants),
        "invariants": invariants,
    }


def _extract_canonical_props(tri_item: dict[str, Any]) -> tuple[float, int]:
    if "canonical_theta" in tri_item and "chirality" in tri_item:
        return float(tri_item["canonical_theta"]), int(tri_item["chirality"])
    verts = tri_item.get("vertices", [])
    if len(verts) == 3:
        _, _, theta, chirality = canonicalize_triangle_vertices(verts)
        return theta, chirality
    return 0.0, 1


def match_templates_detailed(
    template_a: dict[str, Any],
    template_b: dict[str, Any],
    ratio_tolerance: float = 0.05,
    angle_tolerance_rad: float = 0.08,
) -> dict[str, Any]:
    invariants_a = template_a.get("invariants", [])
    invariants_b = template_b.get("invariants", [])
    kps_a = np.asarray(template_a.get("keypoints", []), dtype=np.float64)
    kps_b = np.asarray(template_b.get("keypoints", []), dtype=np.float64)

    len_a = len(invariants_a)
    len_b = len(invariants_b)
    if (len_a == 0 or len_b == 0) and (len(kps_a) < 4 or len(kps_b) < 4):
        return {
            "score": 0.0,
            "matched_pairs": [],
            "num_triangles_a": len_a,
            "num_triangles_b": len_b,
        }

    props_a = [_extract_canonical_props(t) for t in invariants_a]
    props_b = [_extract_canonical_props(t) for t in invariants_b]

    eff_angle_tol = max(angle_tolerance_rad, 0.11)
    eff_ratio_tol = max(ratio_tolerance, 0.10)

    # --- Stage 1: Intrinsic + Neighbor-Ratio Triangle Seeds ---
    seeds: list[dict[str, Any]] = []
    if len_a > 0 and len_b > 0:
        cost_matrix = np.full((len_a, len_b), fill_value=1.0, dtype=np.float32)
        for i, tri_a in enumerate(invariants_a):
            ratios_a = tri_a["adjacent_ratios"]
            angles_a = tri_a.get("angles", [])
            _, chir_a = props_a[i]
            area_a = float(tri_a.get("area", 1.0))

            for j, tri_b in enumerate(invariants_b):
                _, chir_b = props_b[j]
                if chir_a != chir_b:
                    continue
                angles_b = tri_b.get("angles", [])
                if angles_a and angles_b:
                    angle_diff = sum(abs(angles_a[m] - angles_b[m]) for m in range(3)) / 3.0
                    if angle_diff > eff_angle_tol:
                        continue
                else:
                    angle_diff = 0.0

                area_b = float(tri_b.get("area", 1.0))
                area_ratio = min(area_a, area_b) / max(1e-6, max(area_a, area_b))
                if area_ratio < 0.60:
                    continue

                ratios_b = tri_b["adjacent_ratios"]
                if ratios_a and ratios_b:
                    best_r_diff = min(abs(ra - rb) for ra in ratios_a for rb in ratios_b)
                    r_term = min(1.0, best_r_diff / eff_ratio_tol)
                else:
                    r_term = 0.5

                normalized_cost = (
                    0.55 * (angle_diff / eff_angle_tol)
                    + 0.25 * r_term
                    + 0.20 * ((1.0 - area_ratio) / 0.40)
                )
                if normalized_cost < 0.95:
                    cost_matrix[i, j] = float(normalized_cost)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            cost_val = float(cost_matrix[r, c])
            if cost_val < 0.95:
                tri_a = invariants_a[r]
                tri_b = invariants_b[c]
                theta_a, _ = props_a[r]
                theta_b, _ = props_b[c]
                delta_theta = ((theta_b - theta_a + math.pi) % (2.0 * math.pi)) - math.pi
                seeds.append(
                    {
                        "idx_a": int(r),
                        "idx_b": int(c),
                        "cost": cost_val,
                        "centroid_a": [float(tri_a["centroid"][0]), float(tri_a["centroid"][1])],
                        "centroid_b": [float(tri_b["centroid"][0]), float(tri_b["centroid"][1])],
                        "vertices_a": tri_a.get("vertices", []),
                        "vertices_b": tri_b.get("vertices", []),
                        "delta_theta": delta_theta,
                    }
                )

    min_kps = min(len(kps_a), len(kps_b)) if (len(kps_a) > 0 and len(kps_b) > 0) else 0
    min_kp_gate = min(8, max(4, min_kps))

    # --- Stage 2: Hybrid Keypoint Star-Vector + Kabsch SE(2) SVD Consensus ---
    candidate_models: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    for seed in seeds:
        th0 = seed["delta_theta"]
        ca0 = np.array(seed["centroid_a"], dtype=np.float64)
        cb0 = np.array(seed["centroid_b"], dtype=np.float64)
        cos_t, sin_t = math.cos(th0), math.sin(th0)
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        candidate_models.append((ca0, cb0, rot_mat, th0))

    if len(kps_a) >= 4 and len(kps_b) >= 4:
        tree_a = cKDTree(kps_a)
        tree_b = cKDTree(kps_b)
        k_nn_a = min(5, len(kps_a))
        k_nn_b = min(5, len(kps_b))
        _, nn_a = tree_a.query(kps_a, k=k_nn_a)
        _, nn_b = tree_b.query(kps_b, k=k_nn_b)

        segs_a: list[tuple[int, float, float]] = []
        for ia in range(len(kps_a)):
            for m in range(1, min(3, k_nn_a)):
                va = kps_a[nn_a[ia, m]] - kps_a[ia]
                la = float(np.linalg.norm(va))
                if 10.0 <= la <= 140.0:
                    segs_a.append((ia, la, math.atan2(va[1], va[0])))

        segs_b: list[tuple[int, float, float]] = []
        for ib in range(len(kps_b)):
            for m in range(1, min(3, k_nn_b)):
                vb = kps_b[nn_b[ib, m]] - kps_b[ib]
                lb = float(np.linalg.norm(vb))
                if 10.0 <= lb <= 140.0:
                    segs_b.append((ib, lb, math.atan2(vb[1], vb[0])))

        for ia, la, tha in segs_a:
            for ib, lb, thb in segs_b:
                if abs(la - lb) / max(la, lb) > 0.10:
                    continue
                th0 = ((thb - tha + math.pi) % (2.0 * math.pi)) - math.pi
                if abs(th0) > math.radians(55.0):
                    continue
                cos_t, sin_t = math.cos(th0), math.sin(th0)
                rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
                candidate_models.append((kps_a[ia], kps_b[ib], rot_mat, th0))

    shape_a = template_a.get("image_shape", (584, 565))
    spatial_tol = max(14.0, 0.032 * float(min(shape_a[0], shape_a[1])))
    kp_tol = 11.5

    best_model: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    best_matched_kps: list[tuple[int, int]] = []
    best_quality_score = -1.0

    if len(kps_a) > 0 and len(kps_b) > 0:
        kp_tree_b = cKDTree(kps_b)
        for ca0, cb0, rot_mat, th0 in candidate_models:
            pred_kps_b = cb0 + (kps_a - ca0) @ rot_mat.T
            dists, idxs_b = kp_tree_b.query(pred_kps_b, distance_upper_bound=kp_tol)
            valid_mask = np.isfinite(dists)
            raw_count = int(np.sum(valid_mask))
            if raw_count < min_kp_gate or raw_count * 1.2 < best_quality_score:
                continue

            used_ib: set[int] = set()
            pairs_kp: list[tuple[int, int]] = []
            dist_sum = 0.0
            for ia in np.argsort(dists):
                if not valid_mask[ia]:
                    break
                ib = int(idxs_b[ia])
                if ib not in used_ib:
                    used_ib.add(ib)
                    pairs_kp.append((int(ia), ib))
                    dist_sum += float(dists[ia])

            if len(pairs_kp) >= min_kp_gate:
                pts_a_m = np.array([kps_a[ia] for ia, _ in pairs_kp], dtype=np.float64)
                pts_b_m = np.array([kps_b[ib] for _, ib in pairs_kp], dtype=np.float64)
                mean_a = np.mean(pts_a_m, axis=0)
                mean_b = np.mean(pts_b_m, axis=0)
                H = (pts_a_m - mean_a).T @ (pts_b_m - mean_b)
                U, _, Vt = np.linalg.svd(H)
                R_opt = Vt.T @ U.T
                if np.linalg.det(R_opt) < 0:
                    Vt[-1, :] *= -1
                    R_opt = Vt.T @ U.T
                th_opt = math.atan2(R_opt[1, 0], R_opt[0, 0])
                if abs(th_opt) <= math.radians(55.0):
                    pred_opt = mean_b + (kps_a - mean_a) @ R_opt.T
                    dists2, idxs2 = kp_tree_b.query(pred_opt, distance_upper_bound=kp_tol)
                    valid2 = np.isfinite(dists2)
                    used2: set[int] = set()
                    pairs2: list[tuple[int, int]] = []
                    dsum2 = 0.0
                    for ia in np.argsort(dists2):
                        if not valid2[ia]:
                            break
                        ib = int(idxs2[ia])
                        if ib not in used2:
                            used2.add(ib)
                            pairs2.append((int(ia), ib))
                            dsum2 += float(dists2[ia])
                    if len(pairs2) >= len(pairs_kp):
                        pairs_kp = pairs2
                        dist_sum = dsum2
                        ca0, cb0, rot_mat, th0 = mean_a, mean_b, R_opt, th_opt

            q_score = float(len(pairs_kp)) - 0.15 * (dist_sum / max(1, len(pairs_kp)))
            if q_score > best_quality_score:
                best_quality_score = q_score
                best_matched_kps = pairs_kp
                best_model = (ca0, cb0, rot_mat, th0)

    # --- Stage 3: Anatomical Consensus & Induced Common-Subgraph Delaunay Triangulation ---
    kp_inlier_count = len(best_matched_kps)
    kp_inlier_ratio = float(kp_inlier_count) / max(1.0, float(min_kps))

    if min_kps >= 9:
        is_valid_consensus = best_model is not None and (
            kp_inlier_count >= 13 or (kp_inlier_count >= 9 and kp_inlier_ratio >= 0.38)
        )
    else:
        is_valid_consensus = best_model is not None and (
            kp_inlier_count >= min_kp_gate and kp_inlier_ratio >= 0.60
        )

    inliers: list[dict[str, Any]] = []
    if is_valid_consensus and best_model is not None:
        ca0, cb0, rot_mat, th0 = best_model
        used_a: set[int] = set()
        used_b: set[int] = set()

        for cand in seeds:
            ang_err = abs(((cand["delta_theta"] - th0 + math.pi) % (2.0 * math.pi)) - math.pi)
            if ang_err <= 0.30:
                ca = np.array(cand["centroid_a"], dtype=np.float64)
                cb = np.array(cand["centroid_b"], dtype=np.float64)
                if float(np.linalg.norm((cb0 + rot_mat @ (ca - ca0)) - cb)) <= spatial_tol:
                    inliers.append(cand)
                    used_a.add(cand["idx_a"])
                    used_b.add(cand["idx_b"])

        recovery_tol = 0.85 * spatial_tol
        for i, tri_a in enumerate(invariants_a):
            if i in used_a:
                continue
            ca = np.array(tri_a["centroid"], dtype=np.float64)
            pred_cb = cb0 + rot_mat @ (ca - ca0)
            theta_a, chir_a = props_a[i]
            angles_a = tri_a.get("angles", [])

            best_j = -1
            best_dist = recovery_tol
            best_cost = 1.0

            for j, tri_b in enumerate(invariants_b):
                if j in used_b:
                    continue
                theta_b, chir_b = props_b[j]
                if chir_a != chir_b:
                    continue
                dth = ((theta_b - theta_a + math.pi) % (2.0 * math.pi)) - math.pi
                if abs(((dth - th0 + math.pi) % (2.0 * math.pi)) - math.pi) > 0.32:
                    continue
                angles_b = tri_b.get("angles", [])
                if angles_a and angles_b:
                    angle_diff = sum(abs(angles_a[m] - angles_b[m]) for m in range(3)) / 3.0
                    if angle_diff > 0.18:
                        continue
                else:
                    angle_diff = 0.0

                dist = float(np.linalg.norm(pred_cb - np.array(tri_b["centroid"], dtype=np.float64)))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    best_cost = min(0.95, 0.5 * (angle_diff / 0.18) + 0.5 * (dist / recovery_tol))

            if best_j >= 0:
                tri_b = invariants_b[best_j]
                inliers.append(
                    {
                        "idx_a": int(i),
                        "idx_b": int(best_j),
                        "cost": float(best_cost),
                        "centroid_a": [float(tri_a["centroid"][0]), float(tri_a["centroid"][1])],
                        "centroid_b": [float(tri_b["centroid"][0]), float(tri_b["centroid"][1])],
                        "vertices_a": tri_a.get("vertices", []),
                        "vertices_b": tri_b.get("vertices", []),
                    }
                )
                used_a.add(i)
                used_b.add(best_j)

        if len(best_matched_kps) >= 9 and kp_inlier_ratio >= 0.38:
            common_pts_a = np.array([kps_a[ia] for ia, _ in best_matched_kps], dtype=np.float64)
            common_pts_b = np.array([kps_b[ib] for _, ib in best_matched_kps], dtype=np.float64)
            try:
                sub_del = Delaunay(common_pts_a)
                for s_idx, simplex in enumerate(sub_del.simplices):
                    va = common_pts_a[simplex]
                    vb = common_pts_b[simplex]
                    area_va = 0.5 * abs(
                        (va[1, 0] - va[0, 0]) * (va[2, 1] - va[0, 1])
                        - (va[2, 0] - va[0, 0]) * (va[1, 1] - va[0, 1])
                    )
                    area_vb = 0.5 * abs(
                        (vb[1, 0] - vb[0, 0]) * (vb[2, 1] - vb[0, 1])
                        - (vb[2, 0] - vb[0, 0]) * (vb[1, 1] - vb[0, 1])
                    )
                    if min(area_va, area_vb) < 25.0:
                        continue
                    if min(area_va, area_vb) / max(area_va, area_vb) < 0.60:
                        continue
                    ca_sub = np.mean(va, axis=0)
                    cb_sub = np.mean(vb, axis=0)
                    if any(np.linalg.norm(ca_sub - np.array(p["centroid_a"])) < 9.0 for p in inliers):
                        continue
                    inliers.append(
                        {
                            "idx_a": int(len_a + s_idx),
                            "idx_b": int(len_b + s_idx),
                            "cost": 0.20,
                            "centroid_a": [float(ca_sub[0]), float(ca_sub[1])],
                            "centroid_b": [float(cb_sub[0]), float(cb_sub[1])],
                            "vertices_a": va.tolist(),
                            "vertices_b": vb.tolist(),
                        }
                    )
            except Exception:
                pass

    min_tri_threshold = min(6, len_a, len_b) if (len_a > 0 and len_b > 0) else 6
    if len(inliers) < min_tri_threshold:
        inliers = []

    clean_pairs = [
        {
            "idx_a": p["idx_a"],
            "idx_b": p["idx_b"],
            "cost": p["cost"],
            "centroid_a": p["centroid_a"],
            "centroid_b": p["centroid_b"],
            "vertices_a": p["vertices_a"],
            "vertices_b": p["vertices_b"],
        }
        for p in inliers
    ]

    denom = max(1.0, math.sqrt(float(len_a * len_b))) if (len_a > 0 and len_b > 0) else max(1.0, float(max(len_a, len_b)))
    score = float(len(clean_pairs)) / denom
    return {
        "score": min(1.0, max(0.0, score)),
        "matched_pairs": clean_pairs,
        "num_triangles_a": len_a,
        "num_triangles_b": len_b,
    }


def match_templates(
    template_a: dict[str, Any],
    template_b: dict[str, Any],
    ratio_tolerance: float = 0.05,
    angle_tolerance_rad: float = 0.08,
) -> float:
    details = match_templates_detailed(
        template_a=template_a,
        template_b=template_b,
        ratio_tolerance=ratio_tolerance,
        angle_tolerance_rad=angle_tolerance_rad,
    )
    return details["score"]


def verify_identity(score: float, threshold: float = 0.20) -> bool:
    return bool(score >= threshold)
