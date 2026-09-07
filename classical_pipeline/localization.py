"""
Optic Disk and Fovea Localization Module (Classical Pipeline)

ดวิษ (Classical Pipeline) รับผิดชอบโมดูลนี้
วิวรรธน์ (Deep Learning Pipeline) เรียกใช้ตาม Interface Agreement ใน ROADMAP.md

ข้อกำหนดสำคัญ:
vessel_map ที่ส่งเข้าฟังก์ชันต้องถูก resize กลับเป็นขนาดเท่าภาพต้นฉบับก่อนเสมอ
"""

import numpy as np


def find_optic_disk(vessel_map: np.ndarray, green_image: np.ndarray) -> tuple[int, int]:
    """
    ตรวจหาพิกัดจุดศูนย์กลางของจานประสาทตา (Optic Disk - OD)

    Parameters
    ----------
    vessel_map : np.ndarray
        ภาพ binary 2D array ขนาดเท่าภาพต้นฉบับ (0 หรือ 1 โดยที่ 1 = vessel)
    green_image : np.ndarray
        ภาพ green channel 2D array ขนาดเดียวกันกับ vessel_map

    Returns
    -------
    tuple[int, int]
        พิกัดจุดศูนย์กลาง optic disk (x, y) ในระบบพิกัด pixel (x=col, y=row)

    Raises
    ------
    NotImplementedError
        ฟังก์ชันนี้เป็น stub รอการ implement จากดวิษ (Classical Pipeline)
    """
    raise NotImplementedError("find_optic_disk() is a stub and will be implemented by Dawit.")


def find_fovea(vessel_map: np.ndarray, optic_disk_xy: tuple[int, int]) -> tuple[int, int]:
    """
    ตรวจหาพิกัดจุดศูนย์กลางของจุดรับภาพชัด (Fovea)

    Parameters
    ----------
    vessel_map : np.ndarray
        ภาพ binary 2D array ขนาดเท่าภาพต้นฉบับ (0 หรือ 1 โดยที่ 1 = vessel)
    optic_disk_xy : tuple[int, int]
        พิกัด (x, y) ของจุดศูนย์กลาง Optic Disk ที่ตรวจพบได้ก่อนหน้า

    Returns
    -------
    tuple[int, int]
        พิกัดจุดศูนย์กลาง fovea (x, y) ในระบบพิกัด pixel (x=col, y=row)

    Raises
    ------
    NotImplementedError
        ฟังก์ชันนี้เป็น stub รอการ implement จากดวิษ (Classical Pipeline)
    """
    raise NotImplementedError("find_fovea() is a stub and will be implemented by Dawit.")
