"""
Optic Disk and Fovea Localization Module (Classical Pipeline)

Interface stub module defining the contract between Deep Learning front-end
and Classical localization pipeline.

Constraint:
vessel_map passed to these functions must always be restored to original image dimensions.
"""

import numpy as np


def find_optic_disk(vessel_map: np.ndarray, green_image: np.ndarray) -> tuple[int, int]:
    """
    Detect the center coordinates of the Optic Disc (OD).

    Parameters
    ----------
    vessel_map : np.ndarray
        Binary 2D array matching the original image dimensions (0 or 1, where 1 = vessel).
    green_image : np.ndarray
        Green channel 2D array matching the vessel_map dimensions.

    Returns
    -------
    tuple[int, int]
        Center coordinates (x, y) of the optic disc in pixel space (x=col, y=row).

    Raises
    ------
    NotImplementedError
        Interface stub awaiting implementation from classical pipeline.
    """
    raise NotImplementedError("find_optic_disk() is a stub and will be implemented by Dawit.")


def find_fovea(vessel_map: np.ndarray, optic_disk_xy: tuple[int, int]) -> tuple[int, int]:
    """
    Detect the center coordinates of the Fovea.

    Parameters
    ----------
    vessel_map : np.ndarray
        Binary 2D array matching the original image dimensions (0 or 1, where 1 = vessel).
    optic_disk_xy : tuple[int, int]
        Center coordinates (x, y) of the detected Optic Disc.

    Returns
    -------
    tuple[int, int]
        Center coordinates (x, y) of the fovea in pixel space (x=col, y=row).

    Raises
    ------
    NotImplementedError
        Interface stub awaiting implementation from classical pipeline.
    """
    raise NotImplementedError("find_fovea() is a stub and will be implemented by Dawit.")
