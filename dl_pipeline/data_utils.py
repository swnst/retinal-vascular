"""
Dataset Utilities and File Discovery Module

Helper functions to discover and pair retinal fundus images and ground-truth vessel masks
for DRIVE, STARE, and HRF datasets.
"""

import gzip
import io
import re
from pathlib import Path
from typing import Literal
import numpy as np
from PIL import Image, ImageDraw

DatasetName = Literal["DRIVE", "STARE", "HRF"]


def load_rgb_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if str(p).endswith(".gz"):
        with gzip.open(p, "rb") as gz_file:
            with Image.open(io.BytesIO(gz_file.read())) as img_pil:
                return np.array(img_pil.convert("RGB"))
    else:
        with Image.open(p) as img_pil:
            return np.array(img_pil.convert("RGB"))


def load_binary_gt_mask(path: str | Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    p = Path(path)
    if str(p).endswith(".gz"):
        with gzip.open(p, "rb") as gz_file:
            with Image.open(io.BytesIO(gz_file.read())) as mask_pil:
                mask = np.array(mask_pil.convert("L"))
    else:
        with Image.open(p) as mask_pil:
            mask = np.array(mask_pil.convert("L"))

    if target_shape is not None and mask.shape[:2] != target_shape:
        mask_resized = Image.fromarray(mask).resize((target_shape[1], target_shape[0]), Image.NEAREST)
        mask = np.array(mask_resized)

    return (mask > 127).astype(np.uint8)


def list_drive_samples(root_dir: str | Path) -> list[tuple[str, str]]:
    """
    Pair image and mask files for the DRIVE dataset.

    Supported structures:
    - {root}/training/images/*_training.tif with {root}/training/1st_manual/*_manual1.gif
    - {root}/test/images/*_test.tif with {root}/test/1st_manual/*_manual1.gif
    - or flat structures under images/ and 1st_manual/

    Returns
    -------
    list[tuple[str, str]]
        List of (image_path, mask_path) tuples.
    """
    root = Path(root_dir)
    samples: list[tuple[str, str]] = []

    candidate_img_dirs = [
        root / "training" / "images",
        root / "test" / "images",
        root / "images",
    ]
    existing_img_dirs = [d for d in candidate_img_dirs if d.is_dir()]

    if not existing_img_dirs:
        existing_img_dirs = [root]

    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
    img_files: list[Path] = []
    for d in existing_img_dirs:
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                parts = [part.lower() for part in p.parts]
                if not any(k in parts for k in ["manual", "1st_manual", "2nd_manual", "mask", "labels"]):
                    img_files.append(p)

    for img_p in img_files:
        stem = img_p.stem
        id_match = re.search(r"(\d+)", stem)
        if not id_match:
            continue
        sample_id = id_match.group(1)

        mask_candidates = []
        search_dirs = [
            img_p.parent.parent / "1st_manual",
            img_p.parent.parent / "manual",
            img_p.parent.parent / "manual1",
            root / "training" / "1st_manual",
            root / "test" / "1st_manual",
            root / "1st_manual",
            root / "manual1",
            root / "manual",
        ]
        for s_dir in search_dirs:
            if s_dir.is_dir():
                for m_file in s_dir.glob("*"):
                    if m_file.is_file():
                        m_id_match = re.search(r"(\d+)", m_file.stem)
                        if m_id_match and m_id_match.group(1) == sample_id:
                            mask_candidates.append(m_file)

        if not mask_candidates:
            # --- Fallback Annotation Search ---
            for m_file in root.rglob("*"):
                if m_file.is_file():
                    parts = [part.lower() for part in m_file.parts]
                    fname = m_file.name.lower()
                    if ("1st_manual" in parts or "manual" in parts or "manual1" in parts or "manual" in fname) and not fname.endswith("_mask.gif"):
                        m_id_match = re.search(r"(\d+)", m_file.stem)
                        if m_id_match and m_id_match.group(1) == sample_id:
                            mask_candidates.append(m_file)

        if mask_candidates:
            samples.append((str(img_p.resolve()), str(mask_candidates[0].resolve())))

    samples.sort(key=lambda x: x[0])
    return samples


def list_stare_samples(root_dir: str | Path) -> list[tuple[str, str]]:
    """
    Pair image and mask files for the STARE dataset.
    """
    root = Path(root_dir)
    samples: list[tuple[str, str]] = []

    img_dirs = [root / "images", root]
    valid_exts = {".ppm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    img_files: list[Path] = []
    for d in img_dirs:
        if d.is_dir():
            for p in d.glob("*"):
                if p.is_file() and p.suffix.lower() in valid_exts:
                    parts = [part.lower() for part in p.parts]
                    name = p.name.lower()
                    if not any(k in parts for k in ["labels-vk", "labels-ah", "labels", "manual"]):
                        if ".vk." not in name and ".ah." not in name:
                            img_files.append(p)

    # --- Mask File Indexing ---
    mask_lookup: dict[str, list[Path]] = {}
    for m_file in root.rglob("*"):
        if m_file.is_file():
            name = m_file.name.lower()
            parts = [part.lower() for part in m_file.parts]
            if any(k in parts for k in ["labels", "labels-vk", "labels-ah", "manual", "mask"]) or ".vk." in name:
                base_id = m_file.name.split(".")[0]
                mask_lookup.setdefault(base_id, []).append(m_file)

    for img_p in img_files:
        stem = img_p.stem
        mask_candidates = mask_lookup.get(stem, [])
        if mask_candidates:
            vk_masks = [m for m in mask_candidates if "vk" in m.name.lower()]
            chosen_mask = vk_masks[0] if vk_masks else mask_candidates[0]
            samples.append((str(img_p.resolve()), str(chosen_mask.resolve())))

    samples.sort(key=lambda x: x[0])
    return samples


def list_hrf_samples(root_dir: str | Path) -> list[tuple[str, str]]:
    """
    Pair image and mask files for the HRF dataset.
    """
    root = Path(root_dir)
    samples: list[tuple[str, str]] = []

    img_dir = root / "images" if (root / "images").is_dir() else root
    valid_exts = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}

    img_files: list[Path] = []
    for p in img_dir.glob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            parts = [part.lower() for part in p.parts]
            if not any(k in parts for k in ["manual1", "manual", "mask"]):
                img_files.append(p)

    mask_lookup: dict[str, list[Path]] = {}
    for m_file in root.rglob("*"):
        if m_file.is_file():
            parts = [part.lower() for part in m_file.parts]
            if any(k in parts for k in ["manual1", "manual"]):
                mask_lookup.setdefault(m_file.stem.lower(), []).append(m_file)

    for img_p in img_files:
        stem = img_p.stem.lower()
        mask_candidates = mask_lookup.get(stem, [])
        if mask_candidates:
            samples.append((str(img_p.resolve()), str(mask_candidates[0].resolve())))

    samples.sort(key=lambda x: x[0])
    return samples


def discover_dataset_samples(
    root_dir: str | Path, dataset_name: DatasetName | str | None = None
) -> list[tuple[str, str]]:
    """
    Discover and pair images and masks across datasets.
    """
    root = Path(root_dir)
    if not root.exists():
        return []

    name = dataset_name.upper() if dataset_name else root.name.upper()

    if "DRIVE" in name:
        sub_root = root / "DRIVE" if (root / "DRIVE").is_dir() else root
        return list_drive_samples(sub_root)
    elif "STARE" in name:
        sub_root = root / "STARE" if (root / "STARE").is_dir() else root
        return list_stare_samples(sub_root)
    elif "HRF" in name:
        sub_root = root / "HRF" if (root / "HRF").is_dir() else root
        return list_hrf_samples(sub_root)
    else:
        combined: list[tuple[str, str]] = []
        for sub_name, func in [("DRIVE", list_drive_samples), ("HRF", list_hrf_samples), ("STARE", list_stare_samples)]:
            sub_p = root / sub_name
            if sub_p.is_dir():
                combined.extend(func(sub_p))
        if combined:
            return combined

        drive_res = list_drive_samples(root)
        if drive_res:
            return drive_res
        hrf_res = list_hrf_samples(root)
        if hrf_res:
            return hrf_res
        stare_res = list_stare_samples(root)
        if stare_res:
            return stare_res
        return []


def create_synthetic_drive_sample(output_dir: str | Path) -> tuple[str, str]:
    """
    Create a synthetic DRIVE sample for testing environments without raw datasets.
    """
    root = Path(output_dir)
    img_dir = root / "training" / "images"
    mask_dir = root / "training" / "1st_manual"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / "21_training.tif"
    mask_path = mask_dir / "21_manual1.gif"

    if not img_path.exists() or not mask_path.exists():
        w, h = 565, 584
        img = Image.new("RGB", (w, h), color=(10, 5, 5))
        draw = ImageDraw.Draw(img)
        draw.ellipse([30, 20, 535, 564], fill=(180, 80, 30))
        draw.ellipse([120, 250, 200, 330], fill=(240, 210, 120))
        draw.ellipse([340, 280, 380, 320], fill=(130, 40, 10))

        mask = Image.new("L", (w, h), color=0)
        mask_draw = ImageDraw.Draw(mask)

        lines = [
            [(160, 290), (220, 240), (290, 200), (380, 170), (480, 160)],
            [(160, 290), (210, 340), (280, 390), (370, 430), (460, 450)],
            [(220, 240), (230, 180), (250, 130)],
            [(280, 390), (300, 460), (330, 510)],
            [(160, 290), (100, 250), (60, 220)],
            [(160, 290), (90, 330), (50, 370)],
        ]
        for line in lines:
            mask_draw.line(line, fill=255, width=4)
            draw.line(line, fill=(100, 20, 10), width=4)

        img.save(img_path)
        mask.save(mask_path)

    return str(img_path.resolve()), str(mask_path.resolve())
