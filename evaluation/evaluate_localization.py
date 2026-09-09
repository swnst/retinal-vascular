import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classical_pipeline.localization_api import find_fovea, find_optic_disk
from dl_pipeline.predict import load_model_checkpoint, predict_vessel_mask
from evaluation.metrics import localization_distance


# --- Data Loader Functions ---
def load_idrid_groundtruth(
    od_csv_path: Path, fovea_csv_path: Path
) -> dict[str, dict[str, tuple[float, float]]]:
    gt_data: dict[str, dict[str, tuple[float, float]]] = {}

    with open(od_csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            img_id = row[0].strip()
            if not img_id:
                continue
            try:
                x = float(row[1].strip())
                y = float(row[2].strip())
                gt_data[img_id] = {"od": (x, y)}
            except (ValueError, IndexError):
                continue

    with open(fovea_csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            img_id = row[0].strip()
            if not img_id:
                continue
            try:
                x = float(row[1].strip())
                y = float(row[2].strip())
                if img_id in gt_data:
                    gt_data[img_id]["fovea"] = (x, y)
                else:
                    gt_data[img_id] = {"fovea": (x, y)}
            except (ValueError, IndexError):
                continue

    return gt_data


# --- Main Evaluation Pipeline ---
def evaluate_localization(
    images_dir: Path,
    od_csv_path: Path,
    fovea_csv_path: Path,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running localization evaluation on device: {device}")

    model = None
    if model_path and model_path.exists():
        print(f"Loading DL model from: {model_path}")
        model = load_model_checkpoint(str(model_path), device=device)
    else:
        print("Warning: Model path not found. DL vessel segmentation will be skipped or mocked.")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "localization"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_dict = load_idrid_groundtruth(od_csv_path, fovea_csv_path)
    print(f"Loaded ground truth annotations for {len(gt_dict)} images.")

    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]]
    )
    if max_samples:
        image_files = image_files[:max_samples]

    records: list[dict[str, Any]] = []

    for idx, img_path in enumerate(image_files, 1):
        img_id = img_path.stem
        if img_id not in gt_dict:
            continue

        gt_od = gt_dict[img_id].get("od")
        gt_fovea = gt_dict[img_id].get("fovea")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        green_channel = rgb[:, :, 1]
        h, w = green_channel.shape

        if model is not None:
            vessel_mask, _ = predict_vessel_mask(model, rgb, device=device)
        else:
            vessel_mask = np.zeros_like(green_channel, dtype=np.uint8)

        # 1. Optic Disc Localization
        pred_od_xy = find_optic_disk(vessel_mask, green_channel)
        od_err = (
            localization_distance(pred_od_xy, gt_od) if gt_od is not None else float("nan")
        )
        od_norm_err = (od_err / w) if not math.isnan(od_err) else float("nan")

        # 2. Fovea Localization
        pred_fovea_xy = find_fovea(vessel_mask, pred_od_xy, green_channel)
        fovea_err = (
            localization_distance(pred_fovea_xy, gt_fovea)
            if gt_fovea is not None
            else float("nan")
        )
        fovea_norm_err = (fovea_err / w) if not math.isnan(fovea_err) else float("nan")

        records.append(
            {
                "image_id": img_id,
                "width": w,
                "height": h,
                "gt_od_x": gt_od[0] if gt_od else None,
                "gt_od_y": gt_od[1] if gt_od else None,
                "pred_od_x": pred_od_xy[0],
                "pred_od_y": pred_od_xy[1],
                "od_error_px": od_err,
                "od_error_norm": od_norm_err,
                "gt_fovea_x": gt_fovea[0] if gt_fovea else None,
                "gt_fovea_y": gt_fovea[1] if gt_fovea else None,
                "pred_fovea_x": pred_fovea_xy[0],
                "pred_fovea_y": pred_fovea_xy[1],
                "fovea_error_px": fovea_err,
                "fovea_error_norm": fovea_norm_err,
            }
        )

        if idx % 5 == 0 or idx == len(image_files):
            print(f"Processed [{idx}/{len(image_files)}] -> OD Err: {od_err:.1f}px, Fov Err: {fovea_err:.1f}px")

    csv_out = output_dir / "idrid_localization_results.csv"
    fieldnames = [
        "image_id",
        "width",
        "height",
        "gt_od_x",
        "gt_od_y",
        "pred_od_x",
        "pred_od_y",
        "od_error_px",
        "od_error_norm",
        "gt_fovea_x",
        "gt_fovea_y",
        "pred_fovea_x",
        "pred_fovea_y",
        "fovea_error_px",
        "fovea_error_norm",
    ]
    with open(csv_out, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Results saved to: {csv_out}")

    # --- Summary Metrics ---
    od_errors = [r["od_error_px"] for r in records if not math.isnan(r["od_error_px"])]
    fov_errors = [r["fovea_error_px"] for r in records if not math.isnan(r["fovea_error_px"])]
    od_norm = [r["od_error_norm"] for r in records if not math.isnan(r["od_error_norm"])]
    fov_norm = [r["fovea_error_norm"] for r in records if not math.isnan(r["fovea_error_norm"])]

    summary = {
        "total_images": len(records),
        "od_mean_error_px": float(np.mean(od_errors)) if od_errors else 0.0,
        "od_median_error_px": float(np.median(od_errors)) if od_errors else 0.0,
        "od_std_error_px": float(np.std(od_errors)) if od_errors else 0.0,
        "od_mean_norm_error": float(np.mean(od_norm)) if od_norm else 0.0,
        "od_median_norm_error": float(np.median(od_norm)) if od_norm else 0.0,
        "fovea_mean_error_px": float(np.mean(fov_errors)) if fov_errors else 0.0,
        "fovea_median_error_px": float(np.median(fov_errors)) if fov_errors else 0.0,
        "fovea_std_error_px": float(np.std(fov_errors)) if fov_errors else 0.0,
        "fovea_mean_norm_error": float(np.mean(fov_norm)) if fov_norm else 0.0,
        "fovea_median_norm_error": float(np.median(fov_norm)) if fov_norm else 0.0,
        "od_success_rate_5pct": float(np.mean([e <= 0.05 for e in od_norm])) if od_norm else 0.0,
        "fovea_success_rate_5pct": float(np.mean([e <= 0.05 for e in fov_norm])) if fov_norm else 0.0,
    }

    # Generate Visualization for a subset of samples
    import matplotlib.pyplot as plt

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    for r in records[:5]:
        img_id = r["image_id"]
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 7))
        plt.imshow(rgb)

        # Plot OD: Ground Truth (Green circle), Prediction (Red cross)
        if r["gt_od_x"] and r["gt_od_y"]:
            plt.plot(r["gt_od_x"], r["gt_od_y"], "go", markersize=12, label="GT Optic Disc")
        plt.plot(r["pred_od_x"], r["pred_od_y"], "rx", markersize=14, markeredgewidth=3, label="Pred Optic Disc")

        # Plot Fovea: Ground Truth (Cyan circle), Prediction (Yellow cross)
        if r["gt_fovea_x"] and r["gt_fovea_y"]:
            plt.plot(r["gt_fovea_x"], r["gt_fovea_y"], "co", markersize=12, label="GT Fovea")
        plt.plot(r["pred_fovea_x"], r["pred_fovea_y"], "yx", markersize=14, markeredgewidth=3, label="Pred Fovea")

        plt.title(f"{img_id} | OD Err: {r['od_error_px']:.1f}px | Fovea Err: {r['fovea_error_px']:.1f}px")
        plt.legend(loc="upper right")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(vis_dir / f"{img_id}_localization_overlay.png", dpi=150, bbox_inches="tight")
        plt.close()
        vis_count += 1

    print(f"Generated {vis_count} visual overlays in: {vis_dir}")

    report_content = f"""# IDRiD Optic Disc & Fovea Localization Evaluation Report

- **Dataset**: IDRiD C. Localization Testing Set
- **Evaluated Samples**: {summary['total_images']}
- **Pipeline Architecture**: DL U-Net ResNet-34 Vessel Segmentation -> Classical Localization API

---

## 1. Quantitative Performance Summary

| Metric | Optic Disc (OD) | Fovea |
|---|---|---|
| **Mean Error (px)** | {summary['od_mean_error_px']:.2f} px | {summary['fovea_mean_error_px']:.2f} px |
| **Median Error (px)** | {summary['od_median_error_px']:.2f} px | {summary['fovea_median_error_px']:.2f} px |
| **Std Deviation (px)** | {summary['od_std_error_px']:.2f} px | {summary['fovea_std_error_px']:.2f} px |
| **Mean Normalized Error (% Width)** | {summary['od_mean_norm_error']*100:.2f}% | {summary['fovea_mean_norm_error']*100:.2f}% |
| **Median Normalized Error (% Width)** | {summary['od_median_norm_error']*100:.2f}% | {summary['fovea_median_norm_error']*100:.2f}% |
| **Success Rate (Error <= 5% Image Width)** | {summary['od_success_rate_5pct']*100:.2f}% | {summary['fovea_success_rate_5pct']*100:.2f}% |

---

## 2. Technical Observations
- Optic Disc Localization leverages convergence of the segmented retinal vascular tree combined with vessel density and brightness.
- Fovea Localization utilizes anatomical priors relative to the detected Optic Disc center and blood vessel sparsity in the macular zone.
"""

    report_path = output_dir / "localization_evaluation_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Report generated at: {report_path}")

    return summary


if __name__ == "__main__":
    test_img_dir = (
        PROJECT_ROOT
        / "data"
        / "IDRiD"
        / "C. Localization"
        / "C. Localization"
        / "1. Original Images"
        / "b. Testing Set"
    )
    od_csv = (
        PROJECT_ROOT
        / "data"
        / "IDRiD"
        / "C. Localization"
        / "C. Localization"
        / "2. Groundtruths"
        / "1. Optic Disc Center Location"
        / "b. IDRiD_OD_Center_Testing Set_Markups.csv"
    )
    fovea_csv = (
        PROJECT_ROOT
        / "data"
        / "IDRiD"
        / "C. Localization"
        / "C. Localization"
        / "2. Groundtruths"
        / "2. Fovea Center Location"
        / "IDRiD_Fovea_Center_Testing Set_Markups.csv"
    )
    model_checkpoint = PROJECT_ROOT / "models" / "best_unet_model.pth"

    evaluate_localization(
        images_dir=test_img_dir,
        od_csv_path=od_csv,
        fovea_csv_path=fovea_csv,
        model_path=model_checkpoint,
    )
