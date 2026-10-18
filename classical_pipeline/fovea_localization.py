from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

try:
    from .optic_disc_localization import (
        ensure_center_inside_fov,
        normalize_inside_fov,
        select_center_from_score,
        upsample_map,
    )
except ImportError:  # Direct script execution from classical_pipeline/.
    from optic_disc_localization import (
        ensure_center_inside_fov,
        normalize_inside_fov,
        select_center_from_score,
        upsample_map,
    )


@dataclass(frozen=True)
class FoveaConfig:
    method_name: str = "combined_anatomy_darkness"
    working_max_dim: int = 768
    temporal_distance_frac: float = 0.303405
    vertical_offset_frac: float = 0.054073
    prior_sigma_x_frac: float = 0.070
    prior_sigma_y_frac: float = 0.055
    darkness_sigma_frac: float = 0.030
    vessel_sigma_frac: float = 0.050
    fov_border_margin_frac: float = 0.030
    top_percentile: float = 99.7
    anatomy_weight: float = 0.60
    darkness_weight: float = 0.40
    vessel_sparsity_weight: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FoveaResult:
    center_x: float
    center_y: float
    center_x_working: float
    center_y_working: float
    scale_x: float
    scale_y: float
    expected_x: float
    expected_y: float
    maps: dict
    working_maps: dict
    working_shape: tuple


def estimate_fov_mask(image_bgr):
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )
    _, binary = cv2.threshold(
        gray,
        18,
        255,
        cv2.THRESH_BINARY,
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        _, binary = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            raise ValueError("Cannot estimate a non-empty FOV mask.")

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fov = labels == largest_label

    return fov.astype(bool)


def gaussian_sigma(shape, fraction):
    return max(
        1.0,
        float(min(shape) * fraction),
    )


def resize_inputs(image_bgr, fov_mask, vessel_mask, optic_disc_xy, config):
    height, width = fov_mask.shape
    max_dim = max(
        height,
        width,
    )

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

    if vessel_mask is None:
        vessel_work = None
    else:
        vessel_work = cv2.resize(
            vessel_mask.astype(np.uint8),
            size,
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    scale_x = width / float(working_width)
    scale_y = height / float(working_height)
    od_x, od_y = optic_disc_xy
    od_work = (
        float(od_x) / scale_x,
        float(od_y) / scale_y,
    )

    return (
        image_work,
        fov_work,
        vessel_work,
        od_work,
        scale_x,
        scale_y,
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
    eroded = cv2.erode(
        fov_mask.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0

    if np.any(eroded):
        return eroded

    return fov_mask.astype(bool)


def fovea_expected_point(optic_disc_xy, shape, config):
    height, width = shape
    od_x, od_y = optic_disc_xy
    temporal_sign = 1.0 if od_x < (width / 2.0) else -1.0

    expected_x = od_x + temporal_sign * config.temporal_distance_frac * width
    expected_y = od_y + config.vertical_offset_frac * height

    expected_x = min(
        width - 1,
        max(0.0, expected_x),
    )
    expected_y = min(
        height - 1,
        max(0.0, expected_y),
    )

    return expected_x, expected_y


def anatomical_prior(optic_disc_xy, fov_mask, config):
    height, width = fov_mask.shape
    expected_x, expected_y = fovea_expected_point(
        optic_disc_xy,
        fov_mask.shape,
        config,
    )

    y_grid, x_grid = np.indices(
        fov_mask.shape,
        dtype=np.float32,
    )
    sigma_x = max(
        1.0,
        width * config.prior_sigma_x_frac,
    )
    sigma_y = max(
        1.0,
        height * config.prior_sigma_y_frac,
    )
    prior = np.exp(
        -0.5
        * (
            ((x_grid - expected_x) / sigma_x) ** 2
            + ((y_grid - expected_y) / sigma_y) ** 2
        )
    )
    prior[~fov_mask.astype(bool)] = 0.0

    return prior.astype(np.float32), expected_x, expected_y


def darkness_response(image_bgr, fov_mask, config):
    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB,
    )
    lightness = lab[:, :, 0].astype(np.float32)
    green = image_bgr[:, :, 1].astype(np.float32)
    red = image_bgr[:, :, 2].astype(np.float32)

    lightness_norm = normalize_inside_fov(
        lightness,
        fov_mask,
        lower_percentile=2.0,
        upper_percentile=98.5,
    )
    green_norm = normalize_inside_fov(
        green,
        fov_mask,
        lower_percentile=2.0,
        upper_percentile=98.5,
    )
    red_norm = normalize_inside_fov(
        red,
        fov_mask,
        lower_percentile=2.0,
        upper_percentile=98.5,
    )

    dark = (
        0.55 * (1.0 - lightness_norm)
        + 0.35 * (1.0 - green_norm)
        + 0.10 * (1.0 - red_norm)
    )

    sigma = gaussian_sigma(
        fov_mask.shape,
        config.darkness_sigma_frac,
    )
    smoothed = cv2.GaussianBlur(
        dark.astype(np.float32),
        (0, 0),
        sigma,
    )

    return normalize_inside_fov(
        smoothed,
        fov_mask,
        lower_percentile=1.0,
        upper_percentile=99.0,
    )


def vessel_sparsity_response(vessel_mask, fov_mask, config):
    if vessel_mask is None:
        return np.zeros_like(
            fov_mask,
            dtype=np.float32,
        )

    sigma = gaussian_sigma(
        fov_mask.shape,
        config.vessel_sigma_frac,
    )
    density = cv2.GaussianBlur(
        vessel_mask.astype(np.float32),
        (0, 0),
        sigma,
    )
    density_norm = normalize_inside_fov(
        density,
        fov_mask,
        lower_percentile=1.0,
        upper_percentile=99.0,
    )
    sparsity = 1.0 - density_norm
    sparsity[~fov_mask.astype(bool)] = 0.0

    return sparsity.astype(np.float32)


def detect_fovea_center(
    image_bgr,
    optic_disc_xy,
    fov_mask=None,
    vessel_mask=None,
    config=None,
):
    if config is None:
        config = FoveaConfig()

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3).")

    height, width = image_bgr.shape[:2]

    if fov_mask is None:
        fov_mask = estimate_fov_mask(
            image_bgr,
        )
    elif fov_mask.shape != (height, width):
        raise ValueError(
            f"fov_mask shape {fov_mask.shape} does not match image shape {(height, width)}."
        )

    if vessel_mask is not None and vessel_mask.shape != (height, width):
        raise ValueError(
            f"vessel_mask shape {vessel_mask.shape} does not match image shape {(height, width)}."
        )

    if not np.any(fov_mask):
        raise ValueError("FOV mask is empty.")

    od_x, od_y = optic_disc_xy
    if not (
        0 <= float(od_x) < width
        and 0 <= float(od_y) < height
    ):
        raise ValueError("optic_disc_xy is outside image bounds.")

    (
        image_work,
        fov_work,
        vessel_work,
        od_work,
        scale_x,
        scale_y,
    ) = resize_inputs(
        image_bgr,
        fov_mask,
        vessel_mask,
        optic_disc_xy,
        config,
    )

    prior, expected_x, expected_y = anatomical_prior(
        od_work,
        fov_work,
        config,
    )
    dark = darkness_response(
        image_work,
        fov_work,
        config,
    )
    sparsity = vessel_sparsity_response(
        vessel_work,
        fov_work,
        config,
    )

    weights = [
        config.anatomy_weight,
        config.darkness_weight,
        config.vessel_sparsity_weight if vessel_work is not None else 0.0,
    ]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("At least one fovea score weight must be positive.")

    combined = (
        weights[0] * prior
        + weights[1] * dark
        + weights[2] * sparsity
    ) / weight_sum

    candidate = candidate_mask_from_fov(
        fov_work,
        config,
    )
    combined[~candidate] = 0.0

    center_x_work, center_y_work = select_center_from_score(
        combined,
        candidate,
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
        "anatomical_prior": prior,
        "darkness": dark,
        "vessel_sparsity": sparsity,
        "combined_score": combined,
        "candidate_mask": candidate.astype(np.float32),
        "working_fov": fov_work.astype(np.float32),
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

    return FoveaResult(
        center_x=float(center_x),
        center_y=float(center_y),
        center_x_working=float(center_x_work),
        center_y_working=float(center_y_work),
        scale_x=scale_x,
        scale_y=scale_y,
        expected_x=float(expected_x * scale_x),
        expected_y=float(expected_y * scale_y),
        maps=maps,
        working_maps=working_maps,
        working_shape=fov_work.shape,
    )
