import sys
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dl_pipeline.data_utils import discover_dataset_samples
from dl_pipeline.dataset import split_samples
from dl_pipeline.predict import load_model_checkpoint, predict_vessel_mask
from evaluation.metrics import pixel_metrics

def compare_models():
    samples = discover_dataset_samples(PROJECT_ROOT / "data", "ALL")
    val_samples = split_samples(samples, split="val", split_ratios=(0.8, 0.2, 0.0), seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing on device: {device} across {len(val_samples)} validation samples")

    m1 = load_model_checkpoint(str(PROJECT_ROOT / "models" / "best_unet_model.pth"), device=device)
    m2 = load_model_checkpoint(str(PROJECT_ROOT / "models" / "best_unet_model_v2.pth"), device=device)

    dice_v1, dice_v2 = [], []
    sens_v1, sens_v2 = [], []
    spec_v1, spec_v2 = [], []

    for img_p, mask_p in val_samples:
        bgr = cv2.imread(img_p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_binary = (gt_mask > 127).astype(np.uint8)

        # Model 1
        _, prob1 = predict_vessel_mask(m1, rgb, device=device, threshold=0.60, use_tta=True)
        pred1 = (prob1 >= 0.60).astype(np.uint8)
        met1 = pixel_metrics(pred1, gt_binary)
        dice_v1.append(met1["dice"])
        sens_v1.append(met1["sensitivity"])
        spec_v1.append(met1["specificity"])

        # Model 2
        _, prob2 = predict_vessel_mask(m2, rgb, device=device, threshold=0.60, use_tta=True)
        pred2 = (prob2 >= 0.60).astype(np.uint8)
        met2 = pixel_metrics(pred2, gt_binary)
        dice_v2.append(met2["dice"])
        sens_v2.append(met2["sensitivity"])
        spec_v2.append(met2["specificity"])

    print("=" * 50)
    print(f"Model V1 (Old 30 Epochs): Mean Dice = {np.mean(dice_v1):.4f} | Sens = {np.mean(sens_v1):.4f} | Spec = {np.mean(spec_v1):.4f}")
    print(f"Model V2 (New 54 Epochs): Mean Dice = {np.mean(dice_v2):.4f} | Sens = {np.mean(sens_v2):.4f} | Spec = {np.mean(spec_v2):.4f}")
    print(f"Improvement (Delta):      {np.mean(dice_v2) - np.mean(dice_v1):+.4f} Dice")
    print("=" * 50)

if __name__ == "__main__":
    compare_models()
