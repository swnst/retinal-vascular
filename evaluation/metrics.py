"""
Evaluation Metrics Module for Vessel Segmentation and Landmark Localization

โมดูลคำนวณตัวชี้วัดประสิทธิภาพ:
1. pixel_metrics: Accuracy, Sensitivity (Recall / TPR), Specificity (TNR) สำหรับ vessel segmentation
2. localization_distance: Euclidean distance สำหรับการประเมินตำแหน่ง Optic Disk และ Fovea เทียบ Ground Truth
"""

from typing import Any
import numpy as np


def pixel_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """
    คำนวณค่า Pixel-level Classification Metrics สำหรับ Binary Vessel Segmentation:
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Sensitivity (Recall, True Positive Rate) = TP / (TP + FN)
    - Specificity (True Negative Rate) = TN / (TN + FP)

    Parameters
    ----------
    pred : np.ndarray
        Array ผลลัพธ์จากการทำนาย ขนาดเดียวกับ gt
        อาจเป็น probability map (0.0 ถึง 1.0) หรือ binary mask (0/1 หรือ bool)
    gt : np.ndarray
        Ground truth binary mask (ค่า 0 หรือ 1, หรือ bool) ขนาดเดียวกันกับ pred
    threshold : float, optional
        เกณฑ์ threshold สำหรับ binarize ค่า pred หากยังไม่เป็น binary (default: 0.5)

    Returns
    -------
    dict[str, float]
        พจนานุกรมผลลัพธ์:
        {
            "accuracy": float,
            "sensitivity": float,
            "specificity": float
        }
    """
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Shape mismatch: pred shape {pred_arr.shape} does not match gt shape {gt_arr.shape}"
        )

    # Binarize pred and gt
    if pred_arr.dtype == bool:
        pred_bin = pred_arr
    elif np.issubdtype(pred_arr.dtype, np.floating):
        pred_bin = pred_arr >= threshold
    else:
        pred_bin = pred_arr > 0

    if gt_arr.dtype == bool:
        gt_bin = gt_arr
    else:
        gt_bin = gt_arr > 0

    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    tn = int(np.logical_and(~pred_bin, ~gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    positives = tp + fn
    sensitivity = tp / positives if positives > 0 else 0.0

    negatives = tn + fp
    specificity = tn / negatives if negatives > 0 else 0.0

    # Dice / F1 = 2*TP / (2*TP + FP + FN)
    f1_denominator = (2 * tp) + fp + fn
    f1 = (2.0 * tp) / f1_denominator if f1_denominator > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "dice": float(f1),
    }


def compute_auc_roc(prob_map: np.ndarray, gt: np.ndarray, max_samples: int = 200000) -> float:
    """
    คำนวณ Area Under ROC Curve (AUC-ROC) สำหรับความน่าจะเป็นระดับพิกเซล

    Parameters
    ----------
    prob_map : np.ndarray
        Continuous probability map หรือ prediction scores (ช่วง 0.0 ถึง 1.0)
    gt : np.ndarray
        Ground truth binary mask (0 หรือ 1)
    max_samples : int, optional
        การสุ่มตัวอย่างกรณีภาพขนาดใหญ่มากเพื่อความรวดเร็วและป้องกัน memory overflow

    Returns
    -------
    float
        AUC-ROC score (0.0 ถึง 1.0)
    """
    from sklearn.metrics import roc_auc_score

    y_score = np.asarray(prob_map, dtype=np.float32).ravel()
    y_true = (np.asarray(gt).ravel() > 0).astype(np.int32)

    # ตรวจสอบว่าใน ground truth มีทั้ง class 0 และ 1
    if len(np.unique(y_true)) < 2:
        return 1.0

    if len(y_true) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_true), size=max_samples, replace=False)
        y_true = y_true[idx]
        y_score = y_score[idx]

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5



def localization_distance(
    pred_xy: tuple[float, float] | tuple[int, int] | list[float] | np.ndarray,
    gt_xy: tuple[float, float] | tuple[int, int] | list[float] | np.ndarray,
) -> float:
    """
    คำนวณระยะห่าง Euclidean Distance ระหว่างพิกัดที่ทำนาย (pred_xy) กับพิกัดจริง (gt_xy)

    distance = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)

    Parameters
    ----------
    pred_xy : tuple, list, or np.ndarray
        พิกัด (x, y) ที่โมเดลทำนายได้
    gt_xy : tuple, list, or np.ndarray
        พิกัด (x, y) ของ Ground Truth

    Returns
    -------
    float
        ระยะห่าง Euclidean distance มีหน่วยเป็น pixel
    """
    p = np.asarray(pred_xy, dtype=np.float64)
    g = np.asarray(gt_xy, dtype=np.float64)

    if p.size != 2 or g.size != 2:
        raise ValueError(
            f"Coordinate must have exactly 2 elements (x, y), got pred size {p.size}, gt size {g.size}"
        )

    diff = p.flatten() - g.flatten()
    return float(np.sqrt(np.sum(diff ** 2)))
