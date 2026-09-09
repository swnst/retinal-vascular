"""
Evaluation Metrics Module for Vessel Segmentation and Landmark Localization

Performance evaluation metrics module:
1. pixel_metrics: Accuracy, Sensitivity (Recall / TPR), Specificity (TNR) for vessel segmentation.
2. localization_distance: Euclidean distance for evaluating Optic Disc and Fovea landmarks against Ground Truth.
"""

from typing import Any
import numpy as np


def pixel_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """
    Calculate pixel-level classification metrics for binary vessel segmentation:
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Sensitivity (Recall, True Positive Rate) = TP / (TP + FN)
    - Specificity (True Negative Rate) = TN / (TN + FP)

    Parameters
    ----------
    pred : np.ndarray
        Array predicted by model with identical shape to gt.
        Can be a probability map (0.0 to 1.0) or binary mask (0/1 or bool).
    gt : np.ndarray
        Ground truth binary mask (values 0 or 1, or bool) with identical shape to pred.
    threshold : float, optional
        Threshold used to binarize pred if not already binary (default: 0.5).

    Returns
    -------
    dict[str, float]
        Dictionary containing metric scores:
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
    Calculate Area Under ROC Curve (AUC-ROC) for pixel-level probabilities.

    Parameters
    ----------
    prob_map : np.ndarray
        Continuous probability map or prediction scores (range 0.0 to 1.0).
    gt : np.ndarray
        Ground truth binary mask (0 or 1).
    max_samples : int, optional
        Subsample size for large images to optimize speed and avoid memory overflow.

    Returns
    -------
    float
        AUC-ROC score (0.0 to 1.0).
    """
    from sklearn.metrics import roc_auc_score

    y_score = np.asarray(prob_map, dtype=np.float32).ravel()
    y_true = (np.asarray(gt).ravel() > 0).astype(np.int32)

    # Verify ground truth contains both classes
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
    Calculate Euclidean Distance between predicted coordinates (pred_xy) and ground truth (gt_xy).

    distance = sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)

    Parameters
    ----------
    pred_xy : tuple, list, or np.ndarray
        Predicted coordinates (x, y).
    gt_xy : tuple, list, or np.ndarray
        Ground truth coordinates (x, y).

    Returns
    -------
    float
        Euclidean distance in pixels.
    """
    p = np.asarray(pred_xy, dtype=np.float64)
    g = np.asarray(gt_xy, dtype=np.float64)

    if p.size != 2 or g.size != 2:
        raise ValueError(
            f"Coordinate must have exactly 2 elements (x, y), got pred size {p.size}, gt size {g.size}"
        )

    diff = p.flatten() - g.flatten()
    return float(np.sqrt(np.sum(diff ** 2)))
