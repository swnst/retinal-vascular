"""
Training Pipeline Module

Training pipeline for Retinal Vessel Segmentation:
- Combined Loss: Binary Cross Entropy (BCE) + Dice Loss
- Optimizer: Adam
- Training loop with validation evaluation
- Checkpointing and best model serialization
- Cross-platform support (Local / Google Colab / Kaggle)
"""

import os
from pathlib import Path
from typing import Any, Callable
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

import torch.nn.functional as F

from dl_pipeline.model import get_model
from dl_pipeline.dataset import RetinalVesselDataset
from dl_pipeline.data_utils import discover_dataset_samples


# --- Differentiable Morphological Skeletonization & Soft-clDice Loss ---
class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter: int = 10) -> None:
        super().__init__()
        self.num_iter = num_iter

    @staticmethod
    def _soft_erode(img: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def _soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        skel = F.relu(img - self._soft_open(img))
        for _ in range(self.num_iter - 1):
            img = self._soft_erode(img)
            delta = F.relu(img - self._soft_open(img))
            skel = skel + F.relu(delta - skel * delta)
        return skel


class SoftClDiceLoss(nn.Module):
    def __init__(self, num_iter: int = 10, smooth: float = 1e-5) -> None:
        super().__init__()
        self.soft_skel = SoftSkeletonize(num_iter=num_iter)
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        skel_pred = self.soft_skel(probs)
        skel_true = self.soft_skel(targets)

        tprec = ((skel_pred * targets).sum(dim=(1, 2, 3)) + self.smooth) / (
            skel_pred.sum(dim=(1, 2, 3)) + self.smooth
        )
        tsens = ((skel_true * probs).sum(dim=(1, 2, 3)) + self.smooth) / (
            skel_true.sum(dim=(1, 2, 3)) + self.smooth
        )
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.35,
        dice_weight: float = 0.40,
        cldice_weight: float = 0.25,
        pos_weight: float | None = None,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        p_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=p_weight)
        self.dice = DiceLoss(smooth=smooth)
        self.cldice = SoftClDiceLoss(num_iter=10, smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        if self.cldice_weight > 0.0:
            cldice_loss = self.cldice(logits, targets)
            return (
                (self.bce_weight * bce_loss)
                + (self.dice_weight * dice_loss)
                + (self.cldice_weight * cldice_loss)
            )
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


def compute_batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """
    Compute average batch Dice score from logits.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total = preds_flat.sum().item() + targets_flat.sum().item()
    return (2.0 * intersection + smooth) / (total + smooth)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Execute one training epoch.
    Returns (avg_loss, avg_dice).
    """
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        dice = compute_batch_dice(outputs, masks)

        running_loss += loss.item() * batch_size
        running_dice += dice * batch_size
        total_samples += batch_size

    avg_loss = running_loss / max(1, total_samples)
    avg_dice = running_dice / max(1, total_samples)
    return avg_loss, avg_dice


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on validation loader.
    Returns (avg_loss, avg_dice).
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, masks)
        dice = compute_batch_dice(outputs, masks)

        running_loss += loss.item() * batch_size
        running_dice += dice * batch_size
        total_samples += batch_size

    avg_loss = running_loss / max(1, total_samples)
    avg_dice = running_dice / max(1, total_samples)
    return avg_loss, avg_dice


class EarlyStopping:
    """
    Early stopping monitor on target metric (e.g., validation dice or loss).
    """

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    """
    Plot and save training and validation curves.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linestyle="--", linewidth=2)
    axes[0].set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, history["train_dice"], label="Train Dice", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_dice"], label="Val Dice", color="#d62728", linestyle="--", linewidth=2)
    axes[1].set_title("Training vs Validation Dice Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_p, dpi=300)
    plt.close(fig)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None = None,
    num_epochs: int = 30,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: str | Path = "models/best_unet_model_v3.pth",
    plot_path: str | Path = "results/training_curves.png",
    patience: int = 8,
) -> dict[str, list[float]]:
    """
    Execute full training pipeline with validation and early stopping.
    """
    device_obj = torch.device(device)
    model.to(device_obj)

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience, mode="max")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
    }

    best_val_dice = -1.0

    print(f"Starting training on device: {device_obj} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device_obj)
        val_loss, val_dice = validate(model, val_loader, criterion, device_obj)

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                },
                checkpoint_file,
            )
            saved_tag = f" [Saved Best -> Dice: {best_val_dice:.4f}]"
        else:
            saved_tag = ""

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}{saved_tag}"
        )

        if early_stopping(val_dice):
            print(f"Early stopping triggered at epoch {epoch}. Best Val Dice: {best_val_dice:.4f}")
            break

    plot_training_curves(history, plot_path)
    print(f"Training completed. Best model checkpoint saved to: {checkpoint_file}")
    print(f"Training curves saved to: {plot_path}")

    return history
