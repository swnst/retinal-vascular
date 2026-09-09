"""Shared retinal landmark localization package."""

from .localization_api import find_fovea, find_optic_disk

__all__ = ["find_optic_disk", "find_fovea"]
