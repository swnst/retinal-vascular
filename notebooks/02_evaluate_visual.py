"""
Visual Evaluation and Qualitative Comparison Script (Week 4)

Generate qualitative visual comparison figures:
4 columns:
1. Original Retinal Image
2. Ground Truth Vessel Mask
3. Predicted Vessel Mask (U-Net)
4. Error Overlay:
   - Green = True Positive (TP - correctly segmented vessel)
   - Red = False Positive (FP - incorrectly detected background as vessel)
   - Blue = False Negative (FN - missed vessel present in ground truth)

Saves output to results/vessel_prediction_samples.png
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from dl_pipeline.predict import predict_vessel_mask, load_model_checkpoint
from dl_pipeline.data_utils import discover_dataset_samples


def create_error_overlay(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    Generate RGB error overlay for vessel segmentation:
    - Green = TP (Pred=1, GT=1)
    - Red = FP (Pred=1, GT=0)
    - Blue = FN (Pred=0, GT=1)
    - Black = TN (Pred=0, GT=0)
    """
    h, w = pred_bin.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    p = pred_bin > 0
    g = gt_bin > 0

    tp = np.logical_and(p, g)
    fp = np.logical_and(p, ~g)
    fn = np.logical_and(~p, g)

    overlay[tp] = [0, 230, 0]    # Green
    overlay[fp] = [230, 0, 0]    # Red
    overlay[fn] = [0, 100, 255]  # Blue

    return overlay


def generate_visual_comparison_samples(
    model_or_checkpoint: torch.nn.Module | str | Path,
    data_dir: str | Path = "data",
    output_path: str | Path = "results/vessel_prediction_samples.png",
    num_samples_per_dataset: int = 1,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
) -> None:
    """
    Sample images from each dataset and generate a visual comparison figure.
    """
    if isinstance(model_or_checkpoint, (str, Path)):
        model = load_model_checkpoint(model_or_checkpoint, device=device)
    else:
        model = model_or_checkpoint

    data_p = Path(data_dir)
    selected_samples: list[tuple[str, str, str]] = []

    for ds_name in ["DRIVE", "HRF", "STARE"]:
        ds_path = data_p / ds_name
        if ds_path.exists():
            samples = discover_dataset_samples(ds_path, dataset_name=ds_name)
            if samples:
                for s in samples[:num_samples_per_dataset]:
                    selected_samples.append((ds_name, s[0], s[1]))

    if not selected_samples:
        print("No samples found for visualization.")
        return

    n_rows = len(selected_samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "Original Image",
        "Ground Truth",
        "Predicted Vessel Mask",
        "Error Overlay\n(Green: TP, Red: FP, Blue: FN)",
    ]

    import gzip
    import io

    for row_idx, (ds_name, img_path, mask_path) in enumerate(selected_samples):
        with Image.open(img_path) as img_pil:
            orig_img = np.array(img_pil.convert("RGB"))

        if str(mask_path).endswith(".gz"):
            with gzip.open(mask_path, "rb") as gz_file:
                with Image.open(io.BytesIO(gz_file.read())) as mask_pil:
                    gt_mask = (np.array(mask_pil.convert("L")) > 0).astype(np.uint8)
        else:
            with Image.open(mask_path) as mask_pil:
                gt_mask = (np.array(mask_pil.convert("L")) > 0).astype(np.uint8)

        pred_mask, _ = predict_vessel_mask(
            model=model,
            image=orig_img,
            device=device,
            threshold=threshold,
            return_original_size=True,
        )

        if gt_mask.shape != pred_mask.shape:
            gt_resized = Image.fromarray(gt_mask).resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
            gt_mask = np.array(gt_resized)

        error_map = create_error_overlay(pred_mask, gt_mask)


        # Plot 4 columns
        axes[row_idx, 0].imshow(orig_img)
        axes[row_idx, 0].set_ylabel(f"{ds_name}\n({Path(img_path).name})", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("on")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        axes[row_idx, 1].imshow(gt_mask, cmap="gray")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(pred_mask, cmap="gray")
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(error_map)
        axes[row_idx, 3].axis("off")

        if row_idx == 0:
            for col_idx in range(4):
                axes[0, col_idx].set_title(col_titles[col_idx], fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visual comparison figure saved successfully to: {out_p}")


if __name__ == "__main__":
    generate_visual_comparison_samples(
        model_or_checkpoint="models/best_unet_model.pth",
        data_dir="data",
        output_path="results/vessel_prediction_samples.png",
    )
