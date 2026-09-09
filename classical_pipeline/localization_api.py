"""Stable optic-disc and fovea localization API for external vessel maps."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from .fovea_localization import (
        FoveaConfig,
        detect_fovea_center,
        estimate_fov_mask,
    )
    from .optic_disc_localization import (
        OpticDiscConfig,
        detect_optic_disc_center,
    )
except ImportError:  # Direct script execution from classical_pipeline/.
    from fovea_localization import (
        FoveaConfig,
        detect_fovea_center,
        estimate_fov_mask,
    )
    from optic_disc_localization import (
        OpticDiscConfig,
        detect_optic_disc_center,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OD_CONFIG_PATH = PROJECT_ROOT / "results" / "optic_disc" / "best_config.json"
FOVEA_CONFIG_PATH = PROJECT_ROOT / "results" / "fovea" / "best_config.json"

__all__ = [
    "find_optic_disk",
    "find_fovea",
]


def _validate_2d_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (height, width); got {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must use a numeric or boolean dtype; got {array.dtype}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def _normalize_vessel_map(vessel_map: np.ndarray) -> np.ndarray:
    array = _validate_2d_array("vessel_map", vessel_map)
    if array.dtype == np.bool_:
        vessel = array.copy()
    else:
        is_zero = array == 0
        is_one = array == 1
        is_255 = array == 255
        if not np.all(is_zero | is_one | is_255):
            raise ValueError(
                "vessel_map must be binary bool, {0,1}, or {0,255}; "
                "threshold probability maps before localization."
            )
        vessel = (array > 0).astype(bool, copy=True)

    if not np.any(vessel):
        raise ValueError(
            "vessel_map contains no vessel pixels; check reconstruction and thresholding."
        )
    if np.all(vessel):
        raise ValueError(
            "vessel_map marks every pixel as vessel; check reconstruction and thresholding."
        )
    return vessel


def _normalize_green_image(green_image: np.ndarray) -> np.ndarray:
    array = _validate_2d_array("green_image", green_image)
    if array.dtype == np.bool_:
        raise TypeError("green_image must contain intensity values, not boolean data.")

    intensity = array.astype(np.float32, copy=True)
    minimum = float(intensity.min())
    maximum = float(intensity.max())
    if minimum < 0.0:
        raise ValueError("green_image intensity values must be non-negative.")
    if maximum <= 1.0:
        intensity *= 255.0
    elif maximum > 255.0:
        raise ValueError("green_image values must be in [0,1] or [0,255].")

    return np.rint(intensity).clip(0, 255).astype(np.uint8)


def _aligned_inputs(
    vessel_map: np.ndarray,
    green_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vessel = _normalize_vessel_map(vessel_map)
    green = _normalize_green_image(green_image)
    if vessel.shape != green.shape:
        raise ValueError(
            f"vessel_map shape {vessel.shape} does not match "
            f"green_image shape {green.shape}."
        )
    return vessel, green


def _green_to_bgr(green: np.ndarray) -> np.ndarray:
    # The shared contract provides only green intensity. Repeating it preserves
    # that information without inventing unavailable red/blue channels.
    return cv2.merge((green, green, green))


def _load_selected_method(path: Path, config_type):
    if not path.exists():
        raise FileNotFoundError(f"Frozen localization config is missing: {path}")
    with open(path, encoding="utf-8") as json_file:
        bundle = json.load(json_file)
    selected = bundle.get("selected_method")
    if not isinstance(selected, dict):
        raise ValueError(f"selected_method is missing from frozen config: {path}")
    return config_type(**selected)


@lru_cache(maxsize=1)
def _optic_disc_config() -> OpticDiscConfig:
    return _load_selected_method(OD_CONFIG_PATH, OpticDiscConfig)


@lru_cache(maxsize=1)
def _fovea_config() -> FoveaConfig:
    return _load_selected_method(FOVEA_CONFIG_PATH, FoveaConfig)


def _validate_point(
    name: str,
    point_xy: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[float, float]:
    point = np.asarray(point_xy, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite (x, y) coordinate pair.")
    x, y = float(point[0]), float(point[1])
    height, width = shape
    if not (0.0 <= x < width and 0.0 <= y < height):
        raise ValueError(
            f"{name} {(x, y)} is outside image bounds width={width}, height={height}."
        )
    return x, y


def _checked_output(
    name: str,
    point_xy: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[float, float]:
    x, y = _validate_point(name, point_xy, shape)
    return float(x), float(y)


def find_optic_disk(
    vessel_map: np.ndarray,
    green_image: np.ndarray,
) -> tuple[float, float]:
    """Return the optic-disc center as ``(x, y)`` in input coordinates.

    ``vessel_map`` and ``green_image`` must be aligned full-image 2D arrays.
    Binary vessel representations using bool, 0/1, or 0/255 are accepted.
    """

    vessel, green = _aligned_inputs(vessel_map, green_image)
    image_bgr = _green_to_bgr(green)
    fov_mask = estimate_fov_mask(image_bgr)
    result = detect_optic_disc_center(
        image_bgr,
        fov_mask,
        vessel,
        skeleton_mask=None,
        config=_optic_disc_config(),
    )
    return _checked_output(
        "optic-disc result",
        (result.center_x, result.center_y),
        vessel.shape,
    )


def find_fovea(
    vessel_map: np.ndarray,
    optic_disk_xy: tuple[float, float],
    green_image: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return the fovea center as ``(x, y)`` in input coordinates.

    The frozen best fovea method uses image darkness. ``green_image`` is thus
    required even though it remains optional in the signature for a clear,
    backward-compatible error when callers use the earlier two-argument plan.
    """

    if green_image is None:
        raise ValueError(
            "green_image is required by the frozen fovea method because it uses "
            "a darkness/intensity response; intensity is not inferred from vessel_map."
        )

    vessel, green = _aligned_inputs(vessel_map, green_image)
    optic_disc = _validate_point("optic_disk_xy", optic_disk_xy, vessel.shape)
    image_bgr = _green_to_bgr(green)
    fov_mask = estimate_fov_mask(image_bgr)
    result = detect_fovea_center(
        image_bgr,
        optic_disc,
        fov_mask=fov_mask,
        vessel_mask=vessel,
        config=_fovea_config(),
    )
    return _checked_output(
        "fovea result",
        (result.center_x, result.center_y),
        vessel.shape,
    )
