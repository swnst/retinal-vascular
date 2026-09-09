import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "demo" / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def generate_demo_examples():
    drive_dir = PROJECT_ROOT / "data" / "DRIVE" / "training" / "images"
    img1_path = drive_dir / "21_training.tif"
    img2_path = drive_dir / "22_training.tif"

    if not img1_path.exists() or not img2_path.exists():
        print("DRIVE dataset not found at default location.")
        return

    # 1. Subject A (Original)
    bgr_a = cv2.imread(str(img1_path))
    out_subj_a = EXAMPLES_DIR / "subject1_original.png"
    cv2.imwrite(str(out_subj_a), bgr_a)

    # 2. Subject A (Rotated +5 deg and slightly adjusted brightness - Genuine Match)
    h, w, _ = bgr_a.shape
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 5.0, 1.0)
    bgr_a_perturbed = cv2.warpAffine(bgr_a, rot_mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    bgr_a_perturbed = np.clip(bgr_a_perturbed.astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
    out_subj_a_perturbed = EXAMPLES_DIR / "subject1_tilted_genuine.png"
    cv2.imwrite(str(out_subj_a_perturbed), bgr_a_perturbed)

    # 3. Subject B (Different Person - Impostor)
    bgr_b = cv2.imread(str(img2_path))
    out_subj_b = EXAMPLES_DIR / "subject2_impostor.png"
    cv2.imwrite(str(out_subj_b), bgr_b)

    print(f"Generated demo example images in: {EXAMPLES_DIR}")

if __name__ == "__main__":
    generate_demo_examples()
