from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize as skimage_skeletonize

try:
    from .skeleton_metrics import prune_short_spurs
except ImportError:
    from skeleton_metrics import prune_short_spurs


@dataclass(frozen=True)
class OpticDiscConfig:
    method_name: str = "combined_vascular_intensity"
    working_max_dim: int = 640
    density_sigma_frac: float = 0.045
    intensity_sigma_frac: float = 0.035
    orientation_sigma_frac: float = 0.012
    convergence_line_length_frac: float = 0.32
    convergence_sample_stride: int = 5
    fov_border_margin_frac: float = 0.025
    top_percentile: float = 99.5
    vessel_density_weight: float = 0.25
    convergence_weight: float = 0.45
    intensity_weight: float = 0.30
    skeleton_pruning_threshold: int = 10

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class OpticDiscResult:
    center_x: float
    center_y: float
    center_x_working: float
    center_y_working: float
    scale_x: float
    scale_y: float
    maps: dict
    working_maps: dict
    working_shape: tuple


def normalize_inside_fov(values, fov_mask, lower_percentile=1.0, upper_percentile=99.0):
    values_float = values.astype(np.float32)
    valid = values_float[fov_mask.astype(bool)]

    if valid.size == 0:
        raise ValueError("FOV mask has no valid pixels.")

    lower = np.percentile(
        valid,
        lower_percentile,
    )
    upper = np.percentile(
        valid,
        upper_percentile,
    )

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros_like(values_float, dtype=np.float32)

    normalized = (
        values_float - lower
    ) / (
        upper - lower
    )
    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )
    normalized[~fov_mask.astype(bool)] = 0.0

    return normalized.astype(np.float32)


def gaussian_sigma(shape, fraction):
    return max(
        1.0,
        float(min(shape) * fraction),
    )


def resize_to_working(image_bgr, fov_mask, vessel_mask, skeleton_mask, config):
    height, width = fov_mask.shape
    max_dim = max(height, width)

    if max_dim <= config.working_max_dim:
        scale = 1.0
    else:
        scale = config.working_max_dim / float(max_dim)

    working_width = max(
        1,
        int(round(width * scale)),
    )
    working_height = max(
        1,
        int(round(height * scale)),
    )

    size = (
        working_width,
        working_height,
    )

    image_work = cv2.resize(
        image_bgr,
        size,
        interpolation=cv2.INTER_AREA,
    )
    fov_work = cv2.resize(
        fov_mask.astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    vessel_work = cv2.resize(
        vessel_mask.astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    ) > 0

    if skeleton_mask is None:
        skeleton_work = skimage_skeletonize(
            vessel_work,
        )
        if config.skeleton_pruning_threshold > 0:
            skeleton_work = prune_short_spurs(
                skeleton_work,
                threshold=config.skeleton_pruning_threshold,
            )
        skeleton_work = skeleton_work & fov_work
    else:
        skeleton_work = cv2.resize(
            skeleton_mask.astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        skeleton_work = skeleton_work & fov_work

    scale_x = width / float(working_width)
    scale_y = height / float(working_height)

    return (
        image_work,
        fov_work,
        vessel_work,
        skeleton_work,
        scale_x,
        scale_y,
    )


def appearance_response(image_bgr, fov_mask, config):
    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB,
    )
    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV,
    )
    lightness = lab[:, :, 0].astype(np.float32)
    red = image_bgr[:, :, 2].astype(np.float32)
    green = image_bgr[:, :, 1].astype(np.float32)
    blue = image_bgr[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    lightness_norm = normalize_inside_fov(
        lightness,
        fov_mask,
        lower_percentile=5.0,
        upper_percentile=99.5,
    )
    red_norm = normalize_inside_fov(
        red,
        fov_mask,
        lower_percentile=5.0,
        upper_percentile=99.5,
    )
    green_norm = normalize_inside_fov(
        green,
        fov_mask,
        lower_percentile=5.0,
        upper_percentile=99.5,
    )
    blue_norm = normalize_inside_fov(
        blue,
        fov_mask,
        lower_percentile=5.0,
        upper_percentile=99.5,
    )
    saturation_norm = normalize_inside_fov(
        saturation,
        fov_mask,
        lower_percentile=5.0,
        upper_percentile=99.5,
    )

    # --- Pale/Yellow-White OD Tissue Cues ---
    pale_yellow = (
        0.35 * lightness_norm
        + 0.25 * np.minimum(red_norm, green_norm)
        + 0.20 * green_norm
        + 0.20 * (1.0 - saturation_norm)
        - 0.10 * blue_norm
    )

    sigma = gaussian_sigma(
        fov_mask.shape,
        config.intensity_sigma_frac,
    )
    smoothed = cv2.GaussianBlur(
        pale_yellow.astype(np.float32),
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return normalize_inside_fov(
        smoothed,
        fov_mask,
        lower_percentile=5.0,
        upper_percentile=99.5,
    )


def vessel_density_response(vessel_mask, skeleton_mask, fov_mask, config):
    source = (
        0.70 * vessel_mask.astype(np.float32)
        + 0.30 * skeleton_mask.astype(np.float32)
    )
    sigma = gaussian_sigma(
        fov_mask.shape,
        config.density_sigma_frac,
    )
    density = cv2.GaussianBlur(
        source,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return normalize_inside_fov(
        density,
        fov_mask,
        lower_percentile=1.0,
        upper_percentile=99.0,
    )


def tangent_orientation_map(vessel_mask, config):
    vessel_float = vessel_mask.astype(np.float32)
    sigma = gaussian_sigma(
        vessel_mask.shape,
        config.orientation_sigma_frac,
    )
    smoothed = cv2.GaussianBlur(
        vessel_float,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    gradient_x = cv2.Sobel(
        smoothed,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gradient_y = cv2.Sobel(
        smoothed,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    tensor_sigma = max(
        1.0,
        sigma,
    )
    jxx = cv2.GaussianBlur(
        gradient_x * gradient_x,
        (0, 0),
        tensor_sigma,
    )
    jyy = cv2.GaussianBlur(
        gradient_y * gradient_y,
        (0, 0),
        tensor_sigma,
    )
    jxy = cv2.GaussianBlur(
        gradient_x * gradient_y,
        (0, 0),
        tensor_sigma,
    )

    gradient_angle = 0.5 * np.arctan2(
        2.0 * jxy,
        jxx - jyy,
    )

    return gradient_angle + (np.pi / 2.0)


def convergence_response(vessel_mask, skeleton_mask, fov_mask, config):
    if not np.any(skeleton_mask):
        if config.convergence_weight > 0:
            raise ValueError("Cannot compute convergence map from an empty skeleton.")
        return np.zeros_like(fov_mask, dtype=np.float32)

    tangent_angle = tangent_orientation_map(
        vessel_mask,
        config,
    )

    y_coords, x_coords = np.nonzero(skeleton_mask)
    if y_coords.size == 0:
        return np.zeros_like(fov_mask, dtype=np.float32)

    stride = max(
        1,
        int(config.convergence_sample_stride),
    )
    y_coords = y_coords[::stride]
    x_coords = x_coords[::stride]

    line_length = max(
        8,
        int(round(min(fov_mask.shape) * config.convergence_line_length_frac)),
    )

    votes = np.zeros_like(
        fov_mask,
        dtype=np.float32,
    )
    distance_weight = cv2.distanceTransform(
        vessel_mask.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    max_distance = float(distance_weight.max())
    if max_distance > 0:
        distance_weight = 0.75 + 0.25 * (
            distance_weight / max_distance
        )
    else:
        distance_weight = np.ones_like(
            votes,
            dtype=np.float32,
        )

    height, width = fov_mask.shape

    for y, x in zip(y_coords, x_coords):
        angle = tangent_angle[y, x]
        dx = float(np.cos(angle))
        dy = float(np.sin(angle))

        if not np.isfinite(dx) or not np.isfinite(dy):
            continue

        x1 = int(round(x - dx * line_length))
        y1 = int(round(y - dy * line_length))
        x2 = int(round(x + dx * line_length))
        y2 = int(round(y + dy * line_length))

        x1 = min(
            width - 1,
            max(0, x1),
        )
        x2 = min(
            width - 1,
            max(0, x2),
        )
        y1 = min(
            height - 1,
            max(0, y1),
        )
        y2 = min(
            height - 1,
            max(0, y2),
        )

        rr, cc = draw_line(
            y1,
            x1,
            y2,
            x2,
        )
        valid = fov_mask[rr, cc]
        if np.any(valid):
            np.add.at(
                votes,
                (
                    rr[valid],
                    cc[valid],
                ),
                float(distance_weight[y, x]),
            )

    sigma = gaussian_sigma(
        fov_mask.shape,
        config.density_sigma_frac,
    )
    votes = cv2.GaussianBlur(
        votes,
        (0, 0),
        sigma,
    )

    return normalize_inside_fov(
        votes,
        fov_mask,
        lower_percentile=1.0,
        upper_percentile=99.5,
    )


def candidate_mask_from_fov(fov_mask, config):
    margin = int(round(
        min(fov_mask.shape) * config.fov_border_margin_frac
    ))

    if margin <= 0:
        return fov_mask.astype(bool)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * margin + 1,
            2 * margin + 1,
        ),
    )
    candidate = cv2.erode(
        fov_mask.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0

    if not np.any(candidate):
        return fov_mask.astype(bool)

    return candidate


def select_center_from_score(score, candidate_mask, config):
    valid_scores = score[candidate_mask]

    if valid_scores.size == 0:
        raise ValueError("Candidate mask has no valid pixels.")

    if not np.all(np.isfinite(valid_scores)):
        raise ValueError("Score map contains NaN or Inf in candidate region.")

    masked_score = score.copy()
    masked_score[~candidate_mask] = -np.inf
    max_index = int(np.argmax(masked_score))
    max_y, max_x = np.unravel_index(
        max_index,
        score.shape,
    )

    threshold = np.percentile(
        valid_scores,
        config.top_percentile,
    )
    top = (
        (score >= threshold)
        & candidate_mask
    )

    labels_count, labels = cv2.connectedComponents(
        top.astype(np.uint8),
        connectivity=8,
    )
    label = labels[max_y, max_x]

    if labels_count <= 1 or label == 0:
        return float(max_x), float(max_y)

    component = labels == label
    weights = score[component]
    ys, xs = np.nonzero(component)
    weight_sum = float(weights.sum())

    if weight_sum <= 0 or not np.isfinite(weight_sum):
        return float(max_x), float(max_y)

    center_x = float(np.sum(xs * weights) / weight_sum)
    center_y = float(np.sum(ys * weights) / weight_sum)

    return center_x, center_y


def ensure_center_inside_fov(center_x, center_y, fov_mask):
    height, width = fov_mask.shape
    rounded_x = int(round(center_x))
    rounded_y = int(round(center_y))

    rounded_x = min(
        width - 1,
        max(0, rounded_x),
    )
    rounded_y = min(
        height - 1,
        max(0, rounded_y),
    )

    if fov_mask[rounded_y, rounded_x]:
        return float(center_x), float(center_y)

    y_coords, x_coords = np.nonzero(fov_mask)
    if y_coords.size == 0:
        raise ValueError("Cannot project center into an empty FOV mask.")

    distances = (
        (x_coords.astype(np.float32) - center_x) ** 2
        + (y_coords.astype(np.float32) - center_y) ** 2
    )
    nearest = int(np.argmin(distances))

    return float(x_coords[nearest]), float(y_coords[nearest])


def upsample_map(map_work, shape):
    height, width = shape
    return cv2.resize(
        map_work.astype(np.float32),
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )


def detect_optic_disc_center(
    image_bgr,
    fov_mask,
    vessel_mask,
    skeleton_mask=None,
    config=None,
):
    if config is None:
        config = OpticDiscConfig()

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3).")

    height, width = image_bgr.shape[:2]

    for name, mask in [
        ("fov_mask", fov_mask),
        ("vessel_mask", vessel_mask),
    ]:
        if mask.shape != (height, width):
            raise ValueError(
                f"{name} shape {mask.shape} does not match image shape {(height, width)}."
            )

    if skeleton_mask is not None and skeleton_mask.shape != (height, width):
        raise ValueError(
            f"skeleton_mask shape {skeleton_mask.shape} does not match image shape {(height, width)}."
        )

    if not np.any(fov_mask):
        raise ValueError("FOV mask is empty.")

    if not np.any(vessel_mask) and (
        config.vessel_density_weight > 0 or config.convergence_weight > 0
    ):
        raise ValueError("Vascular evidence requested but vessel mask is empty.")

    (
        image_work,
        fov_work,
        vessel_work,
        skeleton_work,
        scale_x,
        scale_y,
    ) = resize_to_working(
        image_bgr,
        fov_mask,
        vessel_mask,
        skeleton_mask,
        config,
    )

    intensity = appearance_response(
        image_work,
        fov_work,
        config,
    )
    density = vessel_density_response(
        vessel_work,
        skeleton_work,
        fov_work,
        config,
    )
    convergence = convergence_response(
        vessel_work,
        skeleton_work,
        fov_work,
        config,
    )

    weight_sum = (
        config.vessel_density_weight
        + config.convergence_weight
        + config.intensity_weight
    )

    if weight_sum <= 0:
        raise ValueError("At least one score-fusion weight must be positive.")

    combined = (
        config.vessel_density_weight * density
        + config.convergence_weight * convergence
        + config.intensity_weight * intensity
    ) / weight_sum

    candidate_mask = candidate_mask_from_fov(
        fov_work,
        config,
    )
    combined[~candidate_mask] = 0.0

    center_x_work, center_y_work = select_center_from_score(
        combined,
        candidate_mask,
        config,
    )
    center_x = center_x_work * scale_x
    center_y = center_y_work * scale_y
    center_x, center_y = ensure_center_inside_fov(
        center_x,
        center_y,
        fov_mask.astype(bool),
    )

    working_maps = {
        "vessel_density": density,
        "convergence": convergence,
        "intensity": intensity,
        "combined_score": combined,
        "candidate_mask": candidate_mask.astype(np.float32),
        "working_vessel_mask": vessel_work.astype(np.float32),
        "working_skeleton": skeleton_work.astype(np.float32),
    }

    maps = {
        name: upsample_map(
            values,
            (height, width),
        )
        for name, values in working_maps.items()
    }

    for name, values in maps.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or Inf.")

    return OpticDiscResult(
        center_x=center_x,
        center_y=center_y,
        center_x_working=center_x_work,
        center_y_working=center_y_work,
        scale_x=scale_x,
        scale_y=scale_y,
        maps=maps,
        working_maps=working_maps,
        working_shape=fov_work.shape,
    )
