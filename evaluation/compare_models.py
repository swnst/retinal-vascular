import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# --- Environment and Module Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classical_pipeline.bezier_matched_filter import extract_vessels_bezier
from classical_pipeline.fovea_localization import estimate_fov_mask
from dl_pipeline.data_utils import discover_dataset_samples, load_rgb_image, load_binary_gt_mask
from dl_pipeline.dataset import split_samples
from dl_pipeline.postprocess import refine_vessel_mask
from dl_pipeline.predict import load_model_checkpoint, predict_vessel_mask
from evaluation.metrics import pixel_metrics, compute_auc_roc


# --- Comparative Performance Benchmarking ---
def compare_models():
    data_dir = PROJECT_ROOT / "data"
    samples = discover_dataset_samples(data_dir, "ALL") if data_dir.exists() else []
    val_samples = split_samples(samples, split="val", split_ratios=(0.8, 0.2, 0.0), seed=42) if samples else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing models on device: {device} across {len(val_samples)} validation samples")

    models_dir = PROJECT_ROOT / "models"
    available_checkpoints = sorted(list(models_dir.glob("*.pth"))) if models_dir.exists() else []
    print(f"Discovered checkpoints: {[p.name for p in available_checkpoints]}")

    if not val_samples:
        print("No validation samples discovered; benchmark completed.")
        return

    loaded_models = {}
    for ckpt in available_checkpoints:
        try:
            loaded_models[ckpt.name] = load_model_checkpoint(str(ckpt), device=device)
        except Exception as e:
            print(f"Warning: Failed to load {ckpt.name}: {e}")

    baseline_name = "Bezier Matched Filter (Inner-FOV Otsu)"
    results = {name: {"dice": [], "sensitivity": [], "specificity": []} for name in loaded_models}
    results[baseline_name] = {"dice": [], "sensitivity": [], "specificity": []}

    for img_p, mask_p in val_samples:
        rgb = load_rgb_image(img_p)
        gt_binary = load_binary_gt_mask(mask_p, target_shape=rgb.shape[:2])

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        fov = estimate_fov_mask(bgr)

        # --- Classical Bezier Baseline ---
        bezier_mask, _, _ = extract_vessels_bezier(rgb, fov_mask=fov, use_otsu=True)
        bezier_bin = (bezier_mask > 127).astype(np.uint8)
        b_met = pixel_metrics(bezier_bin, gt_binary, fov_mask=fov)
        results[baseline_name]["dice"].append(b_met["dice"])
        results[baseline_name]["sensitivity"].append(b_met["sensitivity"])
        results[baseline_name]["specificity"].append(b_met["specificity"])

        # --- Deep Learning Checkpoints ---
        for name, model in loaded_models.items():
            _, prob = predict_vessel_mask(model, rgb, device=device, threshold=0.60, use_tta=True)
            pred = (prob >= 0.60).astype(np.uint8)
            pred = refine_vessel_mask(pred, min_size=25, max_hole_size=15)
            met = pixel_metrics(pred, gt_binary, fov_mask=fov)
            results[name]["dice"].append(met["dice"])
            results[name]["sensitivity"].append(met["sensitivity"])
            results[name]["specificity"].append(met["specificity"])

    print("=" * 70)
    for name, metrics in results.items():
        if metrics["dice"]:
            mean_dice = float(np.mean(metrics["dice"]))
            mean_sens = float(np.mean(metrics["sensitivity"]))
            mean_spec = float(np.mean(metrics["specificity"]))
            print(f"{name:40s}: Dice = {mean_dice:.4f} | Sens = {mean_sens:.4f} | Spec = {mean_spec:.4f}")
    print("=" * 70)

    # --- DRIVE 20-Image Inside-FOV Benchmark ---
    drive_samples = discover_dataset_samples(data_dir, "DRIVE")
    if drive_samples:
        print(f"\nEvaluating DRIVE Benchmark Split ({len(drive_samples)} images, Inside FOV)...")
        drive_results = {
            "Bezier Matched Filter (Inner-FOV Otsu)": {"acc": [], "sens": [], "spec": [], "dice": [], "auc": []},
            "best_unet_model_v3.pth": {"acc": [], "sens": [], "spec": [], "dice": [], "auc": []},
        }
        for img_p, mask_p in drive_samples:
            rgb = load_rgb_image(img_p)
            gt_binary = load_binary_gt_mask(mask_p, target_shape=rgb.shape[:2])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            fov = estimate_fov_mask(bgr)

            # --- Classical Bezier Baseline ---
            bezier_mask, _, _ = extract_vessels_bezier(rgb, fov_mask=fov, use_otsu=True)
            norm_map, _, _ = extract_vessels_bezier(rgb, fov_mask=fov, use_otsu=False)
            bezier_bin = (bezier_mask > 127).astype(np.uint8)
            b_met = pixel_metrics(bezier_bin, gt_binary, fov_mask=fov)
            b_auc = compute_auc_roc(norm_map, gt_binary, fov_mask=fov)
            drive_results["Bezier Matched Filter (Inner-FOV Otsu)"]["acc"].append(b_met["accuracy"])
            drive_results["Bezier Matched Filter (Inner-FOV Otsu)"]["sens"].append(b_met["sensitivity"])
            drive_results["Bezier Matched Filter (Inner-FOV Otsu)"]["spec"].append(b_met["specificity"])
            drive_results["Bezier Matched Filter (Inner-FOV Otsu)"]["dice"].append(b_met["dice"])
            drive_results["Bezier Matched Filter (Inner-FOV Otsu)"]["auc"].append(b_auc)

            # --- Deep Learning Checkpoint ---
            if "best_unet_model_v3.pth" in loaded_models:
                m = loaded_models["best_unet_model_v3.pth"]
                _, prob = predict_vessel_mask(m, rgb, device=device, threshold=0.60, use_tta=True)
                pred = (prob >= 0.60).astype(np.uint8)
                pred = refine_vessel_mask(pred, min_size=25, max_hole_size=15)
                dl_met = pixel_metrics(pred, gt_binary, fov_mask=fov)
                dl_auc = compute_auc_roc(prob, gt_binary, fov_mask=fov)
                drive_results["best_unet_model_v3.pth"]["acc"].append(dl_met["accuracy"])
                drive_results["best_unet_model_v3.pth"]["sens"].append(dl_met["sensitivity"])
                drive_results["best_unet_model_v3.pth"]["spec"].append(dl_met["specificity"])
                drive_results["best_unet_model_v3.pth"]["dice"].append(dl_met["dice"])
                drive_results["best_unet_model_v3.pth"]["auc"].append(dl_auc)

        print("=" * 85)
        for name, metrics in drive_results.items():
            if metrics["dice"]:
                m_acc, s_acc = np.mean(metrics["acc"]), np.std(metrics["acc"])
                m_sens, s_sens = np.mean(metrics["sens"]), np.std(metrics["sens"])
                m_spec, s_spec = np.mean(metrics["spec"]), np.std(metrics["spec"])
                m_dice, s_dice = np.mean(metrics["dice"]), np.std(metrics["dice"])
                m_auc, s_auc = np.mean(metrics["auc"]), np.std(metrics["auc"])
                print(f"{name:40s}: ACC={m_acc:.4f} (+/- {s_acc:.4f}) | SE={m_sens:.4f} (+/- {s_sens:.4f}) | SP={m_spec:.4f} (+/- {s_spec:.4f}) | Dice={m_dice:.4f} (+/- {s_dice:.4f}) | AUC={m_auc:.4f} (+/- {s_auc:.4f})")
        print("=" * 85)


# --- Entrypoint ---
if __name__ == "__main__":
    compare_models()
