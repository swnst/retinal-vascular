from __future__ import annotations

from collections import deque

import cv2
import numpy as np


NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def as_bool(image: np.ndarray) -> np.ndarray:
    return image if image.dtype == np.bool_ else (image > 0)


def skeleton_neighbor_count(skeleton):
    skeleton_uint8 = as_bool(skeleton).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    kernel[1, 1] = 0

    return cv2.filter2D(
        skeleton_uint8,
        ddepth=cv2.CV_16U,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )


def find_endpoints(skeleton):
    skeleton_bool = as_bool(skeleton)
    neighbor_count = skeleton_neighbor_count(skeleton_bool)

    return (
        skeleton_bool
        & (neighbor_count == 1)
    )


def find_branchpoint_candidates(skeleton):
    skeleton_bool = as_bool(skeleton)
    neighbor_count = skeleton_neighbor_count(skeleton_bool)

    return (
        skeleton_bool
        & (neighbor_count >= 3)
    )


def connected_component_stats(binary):
    binary_uint8 = as_bool(binary).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_uint8,
        connectivity=8,
    )

    component_sizes = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, num_labels)
    ]

    largest = max(component_sizes) if component_sizes else 0

    return {
        "component_count": len(component_sizes),
        "component_sizes": component_sizes,
        "largest_component_size": largest,
        "labels": labels,
    }


def group_branchpoints(branchpoint_candidates):
    candidates_uint8 = as_bool(branchpoint_candidates).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidates_uint8,
        connectivity=8,
    )

    grouped_map = np.zeros_like(
        candidates_uint8,
        dtype=bool,
    )

    for label in range(1, num_labels):
        centroid_x, centroid_y = centroids[label]

        y = int(round(centroid_y))
        x = int(round(centroid_x))

        y = int(np.clip(y, 0, grouped_map.shape[0] - 1))
        x = int(np.clip(x, 0, grouped_map.shape[1] - 1))

        grouped_map[y, x] = True

    return {
        "grouped_junction_count": int(num_labels - 1),
        "grouped_junction_map": grouped_map,
        "labels": labels,
        "stats": stats,
    }


def neighbor_coords(skeleton: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    skeleton_bool = skeleton if skeleton.dtype == np.bool_ else (skeleton > 0)
    height, width = skeleton_bool.shape

    coords = []

    for dy, dx in NEIGHBOR_OFFSETS:
        ny = y + dy
        nx = x + dx

        if (
            0 <= ny < height
            and 0 <= nx < width
            and skeleton_bool[ny, nx]
        ):
            coords.append((ny, nx))

    return coords


def trace_terminal_branch(skeleton, endpoint):
    skeleton_bool = as_bool(skeleton)
    current = endpoint
    previous = None
    path = []
    visited = set()

    while True:
        path.append(current)
        visited.add(current)

        y, x = current
        neighbors = neighbor_coords(
            skeleton_bool,
            y,
            x,
        )

        degree = len(neighbors)

        if current != endpoint and degree != 2:
            if degree >= 3:
                stop_type = "junction"
            elif degree <= 1:
                stop_type = "endpoint"
            else:
                stop_type = "ambiguous"

            removable = (
                path[:-1]
                if stop_type == "junction"
                else path
            )

            return {
                "path": path,
                "removable_path": removable,
                "stop_type": stop_type,
            }

        candidates = [
            coord
            for coord in neighbors
            if coord != previous
        ]

        if not candidates:
            return {
                "path": path,
                "removable_path": path,
                "stop_type": "dead_end",
            }

        if len(candidates) > 1:
            return {
                "path": path,
                "removable_path": path,
                "stop_type": "junction",
            }

        previous = current
        current = candidates[0]

        if current in visited:
            return {
                "path": path,
                "removable_path": path,
                "stop_type": "loop",
            }


def terminal_branches(skeleton):
    endpoints = find_endpoints(skeleton)
    endpoint_coords = [
        tuple(coord)
        for coord in np.argwhere(endpoints)
    ]

    branches = []
    seen_paths = set()

    for endpoint in endpoint_coords:
        branch = trace_terminal_branch(
            skeleton,
            endpoint,
        )

        removable_path = branch["removable_path"]

        if not removable_path:
            continue

        path_key = frozenset(removable_path)

        if path_key in seen_paths:
            continue

        seen_paths.add(path_key)

        branch["length"] = len(removable_path)
        branch["endpoint"] = endpoint
        branches.append(branch)

    return branches


def prune_short_spurs(
    skeleton,
    threshold,
    max_iterations=30,
):
    pruned = as_bool(skeleton).copy()

    if threshold <= 0:
        return pruned

    for _ in range(max_iterations):
        branches = terminal_branches(pruned)

        remove_coords = []

        for branch in branches:
            if branch["length"] <= threshold:
                remove_coords.extend(branch["removable_path"])

        if not remove_coords:
            break

        for y, x in remove_coords:
            pruned[y, x] = False

    return pruned


def count_one_pixel_width_violations(skeleton):
    skeleton_uint8 = as_bool(skeleton).astype(np.uint8)

    if (
        skeleton_uint8.shape[0] < 2
        or skeleton_uint8.shape[1] < 2
    ):
        return 0

    block_sum = (
        skeleton_uint8[:-1, :-1]
        + skeleton_uint8[1:, :-1]
        + skeleton_uint8[:-1, 1:]
        + skeleton_uint8[1:, 1:]
    )

    return int(np.count_nonzero(block_sum == 4))


def count_small_components(component_sizes, max_size=10):
    return int(
        sum(
            1
            for size in component_sizes
            if size <= max_size
        )
    )


def calculate_topology_metrics(
    skeleton,
    short_spur_threshold=10,
):
    skeleton_bool = as_bool(skeleton)
    skeleton_pixels = int(np.count_nonzero(skeleton_bool))

    component_info = connected_component_stats(skeleton_bool)
    component_count = component_info["component_count"]
    component_sizes = component_info["component_sizes"]
    largest_component_size = component_info["largest_component_size"]

    largest_component_ratio = (
        largest_component_size / skeleton_pixels
        if skeleton_pixels > 0
        else 0.0
    )

    endpoints = find_endpoints(skeleton_bool)
    endpoint_count = int(np.count_nonzero(endpoints))

    branchpoint_candidates = find_branchpoint_candidates(skeleton_bool)
    candidate_branchpoint_count = int(
        np.count_nonzero(branchpoint_candidates)
    )

    grouped = group_branchpoints(branchpoint_candidates)

    branches = terminal_branches(skeleton_bool)
    branch_lengths = [
        branch["length"]
        for branch in branches
    ]

    short_spur_count = int(
        sum(
            1
            for length in branch_lengths
            if length <= short_spur_threshold
        )
    )

    mean_branch_length = (
        float(np.mean(branch_lengths))
        if branch_lengths
        else 0.0
    )

    return {
        "total_skeleton_pixels": skeleton_pixels,
        "total_skeleton_length_pixels": skeleton_pixels,
        "connected_component_count": component_count,
        "largest_component_size": largest_component_size,
        "largest_component_ratio": float(largest_component_ratio),
        "endpoint_count": endpoint_count,
        "candidate_branchpoint_count": candidate_branchpoint_count,
        "grouped_junction_count": grouped["grouped_junction_count"],
        "very_small_component_count": count_small_components(
            component_sizes,
            max_size=10,
        ),
        "short_spur_count": short_spur_count,
        "mean_branch_length": mean_branch_length,
        "one_pixel_width_violations": count_one_pixel_width_violations(
            skeleton_bool
        ),
    }


def maps_for_skeleton(skeleton):
    endpoints = find_endpoints(skeleton)
    branchpoint_candidates = find_branchpoint_candidates(skeleton)
    grouped = group_branchpoints(branchpoint_candidates)

    return {
        "endpoints": endpoints,
        "branchpoint_candidates": branchpoint_candidates,
        "grouped_junctions": grouped["grouped_junction_map"],
    }
