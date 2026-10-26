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
except ImportError:
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


def _normalize_image_input(
    image_input: np.ndarray,
    color_order: str = "RGB",
) -> np.ndarray:
    array = np.asarray(image_input)
    if array.size == 0:
        raise ValueError("image input must not be empty.")
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise TypeError(f"image input must use numeric or boolean dtype; got {array.dtype}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("image input contains NaN or Inf.")

    if array.ndim == 2:
        if array.dtype == np.bool_:
            raise TypeError("image input must contain intensity values, not boolean data.")
        intensity = array.astype(np.float32, copy=True)
        minimum = float(intensity.min())
        maximum = float(intensity.max())
        if minimum < 0.0:
            raise ValueError("image input intensity values must be non-negative.")
        if maximum <= 1.0:
            intensity *= 255.0
        elif maximum > 255.0:
            raise ValueError("image input values must be in [0,1] or [0,255].")
        uint8_2d = np.rint(intensity).clip(0, 255).astype(np.uint8)
        return cv2.merge((uint8_2d, uint8_2d, uint8_2d))
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        channels = array[:, :, :3].astype(np.float32, copy=True)
        if float(channels.max()) <= 1.0:
            channels *= 255.0
        uint8_3d = np.rint(channels).clip(0, 255).astype(np.uint8)
        # --- Color Space Normalization ---
        if color_order.upper() == "RGB":
            return cv2.cvtColor(uint8_3d, cv2.COLOR_RGB2BGR)
        return uint8_3d
    else:
        raise ValueError(f"image input must be 2D (H, W) or 3D (H, W, 3/4); got shape {array.shape}.")


def _aligned_inputs(
    vessel_map: np.ndarray,
    green_or_color_image: np.ndarray,
    color_order: str = "RGB",
) -> tuple[np.ndarray, np.ndarray]:
    vessel = _normalize_vessel_map(vessel_map)
    color_bgr = _normalize_image_input(green_or_color_image, color_order=color_order)
    if vessel.shape[:2] != color_bgr.shape[:2]:
        raise ValueError(
            f"vessel_map shape {vessel.shape[:2]} does not match "
            f"image shape {color_bgr.shape[:2]}."
        )
    return vessel, color_bgr


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
    color_order: str = "RGB",
) -> tuple[float, float]:
    """Return the optic-disc center as ``(x, y)`` in input coordinates.

    ``vessel_map`` and ``green_image`` must be aligned full-image arrays.
    ``green_image`` can be either a 2D grayscale/green array or a 3D color array.
    """

    vessel, image_bgr = _aligned_inputs(vessel_map, green_image, color_order=color_order)
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
        vessel.shape[:2],
    )


def find_fovea(
    vessel_map: np.ndarray,
    optic_disk_xy: tuple[float, float],
    green_image: np.ndarray | None = None,
    color_order: str = "RGB",
) -> tuple[float, float]:
    """Return the fovea center as ``(x, y)`` in input coordinates.

    The frozen best fovea method uses image darkness. ``green_image`` is thus
    required even though it remains optional in the signature for backward-compatibility.
    ``green_image`` can be either a 2D grayscale/green array or a 3D color array.
    """

    if green_image is None:
        raise ValueError(
            "green_image is required by the frozen fovea method because it uses "
            "a darkness/intensity response; intensity is not inferred from vessel_map."
        )

    vessel, image_bgr = _aligned_inputs(vessel_map, green_image, color_order=color_order)
    optic_disc = _validate_point("optic_disk_xy", optic_disk_xy, vessel.shape[:2])
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
        vessel.shape[:2],
    )
