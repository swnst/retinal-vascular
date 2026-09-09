"""
Sanity Check Script: Load DRIVE Image and Visualize Channels

Script for verifying image loading from the DRIVE dataset:
- Locate sample image from data/DRIVE/ (training or test)
- Load Original RGB Image, extract Green Channel, and load Ground Truth Mask
- Visualize 3-channel comparison with matplotlib (Original, Green Channel, Ground Truth Mask)
- Save visualization to file (notebooks/sanity_check_output.png) without requiring interactive GUI display
- Synthetic fallback: generates synthetic retinal fundus images if data/DRIVE is empty
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dl_pipeline.data_utils import list_drive_samples, create_synthetic_drive_sample


def run_sanity_check(data_root: str = "data/DRIVE", output_plot_path: str = "notebooks/sanity_check_output.png"):
    root = Path(data_root)
    samples = list_drive_samples(root)

    if not samples:
        print(f"[Notice] No DRIVE samples found in {root}. Generating synthetic sample for validation...")
        img_p, mask_p = create_synthetic_drive_sample(root)
    else:
        img_p, mask_p = samples[0]
        print(f"[Success] Found DRIVE sample: {img_p}")

    with Image.open(img_p) as img_pil:
        img_rgb = np.array(img_pil.convert("RGB"))

    # Green channel (index 1) provides maximum vessel-to-background contrast
    green_channel = img_rgb[:, :, 1]

    with Image.open(mask_p) as mask_pil:
        mask_arr = np.array(mask_pil.convert("L"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original RGB\nShape: {img_rgb.shape}", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(green_channel, cmap="gray")
    axes[1].set_title(f"Green Channel (High Contrast)\nShape: {green_channel.shape}", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(mask_arr, cmap="gray")
    axes[2].set_title(f"Ground Truth Mask\nShape: {mask_arr.shape}", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    out_dir = Path(output_plot_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Sanity check visualization saved successfully to: {output_plot_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    run_sanity_check()
