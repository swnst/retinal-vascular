"""
PyTorch Retinal Vessel Dataset Module

RetinalVesselDataset class for loading fundus images and ground truth masks:
- Multi-dataset support (DRIVE, STARE, HRF)
- Image-level train/val/test splitting
- Data augmentation via Albumentations
- Normalized tensor outputs (Image: 3x512x512, Mask: 1x512x512)
"""

import gzip
import io
import math
from pathlib import Path
from typing import Callable, Literal, Sequence
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A

from dl_pipeline.data_utils import discover_dataset_samples, DatasetName


def get_default_train_transform(img_size: int = 512) -> A.Compose:
    """
    Default Albumentations compose transform for training:
    - Resize 512x512
    - HorizontalFlip, VerticalFlip
    - RandomRotate90
    - RandomBrightnessContrast
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
        ]
    )


def get_default_val_test_transform(img_size: int = 512) -> A.Compose:
    """
    Default Albumentations compose transform for validation/testing (Resize only).
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
        ]
    )


def split_samples(
    samples: Sequence[tuple[str, str]],
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    split: Literal["train", "val", "test", "all"] = "train",
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Split list of (image, mask) pairs into train / val / test at the image level.
    """
    if split == "all":
        return list(samples)

    n_total = len(samples)
    if n_total == 0:
        return []

    r_train, r_val, r_test = split_ratios
    ratio_sum = r_train + r_val + r_test
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-3):
        r_train, r_val, r_test = r_train / ratio_sum, r_val / ratio_sum, r_test / ratio_sum

    rng = np.random.RandomState(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    if n_total == 1:
        n_train, n_val = 1, 0
    elif n_total == 2:
        n_train, n_val = 1, 1
    else:
        if r_test == 0.0:
            n_val = max(1, int(round(n_total * r_val)))
            n_train = n_total - n_val
        else:
            n_train = max(1, int(round(n_total * r_train)))
            n_val = max(1, int(round(n_total * r_val)))
            if n_train + n_val >= n_total:
                n_train = max(1, n_total - 2)
                n_val = 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    if split == "train":
        chosen_indices = train_idx
    elif split == "val":
        chosen_indices = val_idx
    elif split == "test":
        chosen_indices = test_idx if len(test_idx) > 0 else val_idx
    else:
        raise ValueError(f"Unknown split: {split}. Choose from ['train', 'val', 'test', 'all']")

    return [samples[i] for i in chosen_indices]


class RetinalVesselDataset(Dataset):
    """
    PyTorch Dataset for Retinal Blood Vessel Segmentation.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        dataset_name: DatasetName | str | None = None,
        samples: list[tuple[str, str]] | None = None,
        split: Literal["train", "val", "test", "all"] = "train",
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        img_size: int = 512,
        transform: A.Compose | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.split = split

        if samples is not None:
            all_samples = list(samples)
        elif root_dir is not None:
            all_samples = discover_dataset_samples(root_dir, dataset_name=dataset_name)
        else:
            all_samples = []

        self.samples = split_samples(
            all_samples, split_ratios=split_ratios, split=split, seed=seed
        )

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_default_train_transform(img_size=img_size)
        else:
            self.transform = get_default_val_test_transform(img_size=img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path, mask_path = self.samples[idx]

        with Image.open(img_path) as img_pil:
            image = np.array(img_pil.convert("RGB"))

        if str(mask_path).endswith(".gz"):
            with gzip.open(mask_path, "rb") as gz_file:
                with Image.open(io.BytesIO(gz_file.read())) as mask_pil:
                    mask = np.array(mask_pil.convert("L"))
        else:
            with Image.open(mask_path) as mask_pil:
                mask = np.array(mask_pil.convert("L"))

        if image.shape[:2] != mask.shape[:2]:
            mask_resized = Image.fromarray(mask).resize((image.shape[1], image.shape[0]), Image.NEAREST)
            mask = np.array(mask_resized)

        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image_norm = image.astype(np.float32) / 255.0
        mask_norm = mask.astype(np.float32)

        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_norm).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }
