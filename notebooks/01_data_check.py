"""
Data Pipeline Check Script: RetinalVesselDataset & Augmentations

Script for testing the Dataset class implementation in dl_pipeline/dataset.py:
- Load Dataset from data/DRIVE/ (or unified dataset directories)
- Verify data augmentations (HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast)
- Extract 4 augmented sample pairs (Image + Mask)
- Plot 4 pairs side-by-side using matplotlib
- Save output visualization to notebooks/data_check_output.png without requiring interactive GUI display
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dl_pipeline.dataset import RetinalVesselDataset, get_default_train_transform
from dl_pipeline.data_utils import create_synthetic_drive_sample


def run_data_check(data_root: str = "data/DRIVE", output_plot_path: str = "notebooks/data_check_output.png"):
    root = Path(data_root)
    existing_samples = list(root.rglob("*.tif")) if root.exists() else []
    if len(existing_samples) < 4:
        print("[Notice] Ensuring at least 4 samples exist for visualization...")
        create_synthetic_drive_sample(root)
        for i in range(22, 25):
            img_dir = root / "training" / "images"
            mask_dir = root / "training" / "1st_manual"
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            src_img = img_dir / "21_training.tif"
            src_mask = mask_dir / "21_manual1.gif"
            dst_img = img_dir / f"{i}_training.tif"
            dst_mask = mask_dir / f"{i}_manual1.gif"
            if not dst_img.exists() and src_img.exists():
                with open(src_img, "rb") as fsrc, open(dst_img, "wb") as fdst:
                    fdst.write(fsrc.read())
            if not dst_mask.exists() and src_mask.exists():
                with open(src_mask, "rb") as fsrc, open(dst_mask, "wb") as fdst:
                    fdst.write(fsrc.read())

    dataset = RetinalVesselDataset(
        root_dir=root,
        dataset_name="DRIVE",
        split="all",
        img_size=512,
        transform=get_default_train_transform(img_size=512),
    )

    print(f"[Info] Total samples available in dataset: {len(dataset)}")
    num_samples_to_show = min(4, len(dataset))
    if num_samples_to_show == 0:
        raise RuntimeError("No samples could be loaded from dataset!")

    fig, axes = plt.subplots(num_samples_to_show, 2, figsize=(10, 4 * num_samples_to_show))
    if num_samples_to_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples_to_show):
        sample = dataset[i]
        img_tensor = sample["image"]
        mask_tensor = sample["mask"]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        mask_np = mask_tensor.squeeze(0).numpy()

        axes[i, 0].imshow(np.clip(img_np, 0.0, 1.0))
        axes[i, 0].set_title(f"Sample #{i+1} Augmented Image (512x512)", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title(f"Sample #{i+1} Augmented Mask (512x512)", fontsize=11)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_dir = Path(output_plot_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Data check visualization with 4 pairs saved successfully to: {output_plot_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    run_data_check()
