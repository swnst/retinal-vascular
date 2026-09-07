"""
Google Colab & Local Training Pipeline Script (Week 3)

สคริปต์สำหรับการรันเทรนโมเดล U-Net ResNet-34 บน Google Colab (T4 GPU) หรือ Local GPU/CPU:
1. รองรับการเทรนบน GPU (CUDA) อัตโนมัติ
2. โหลด Dataset แบบ 80/20 train/val split
3. CombinedLoss (BCEWithLogitsLoss + DiceLoss) แก้ปัญหา Class Imbalance
4. Adam optimizer (lr=1e-4) ร่วมกับ CosineAnnealingLR / ReduceLROnPlateau
5. Early Stopping ป้องกัน Overfitting
6. บันทึก Best Model ไว้ที่ models/best_unet_model.pth
7. พล็อตและบันทึกผลการเทรนที่ results/training_curves.png
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from dl_pipeline.model import get_model, count_parameters
from dl_pipeline.dataset import RetinalVesselDataset
from dl_pipeline.train import CombinedLoss, train_model


def run_training(
    data_dir: str = "data",
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    checkpoint_path: str = "models/best_unet_model.pth",
    plot_path: str = "results/training_curves.png",
    patience: int = 8,
    img_size: int = 512,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Starting Retinal Vessel Training Pipeline ===")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & DataLoader (Train/Val split 80/20)
    print(f"Loading datasets from: {data_dir}...")
    train_dataset = RetinalVesselDataset(
        root_dir=data_dir,
        split="train",
        split_ratios=(0.8, 0.2, 0.0),
        img_size=img_size,
        seed=42,
    )
    val_dataset = RetinalVesselDataset(
        root_dir=data_dir,
        split="val",
        split_ratios=(0.8, 0.2, 0.0),
        img_size=img_size,
        seed=42,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device.startswith("cuda") else 0,
        pin_memory=True if device.startswith("cuda") else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if device.startswith("cuda") else 0,
        pin_memory=True if device.startswith("cuda") else False,
    )

    # 2. Model Architecture
    print("Building U-Net (ResNet-34 Encoder)...")
    model = get_model(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    param_info = count_parameters(model)
    print(f"Total parameters: {param_info['total_params']:,}")
    print(f"Trainable parameters: {param_info['trainable_params']:,}")

    # 3. Loss & Optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Train Model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=epochs,
        device=device,
        checkpoint_path=checkpoint_path,
        plot_path=plot_path,
        patience=patience,
    )

    print("=== Training Pipeline Finished Successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Retinal Vessel Segmentation U-Net")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="models/best_unet_model.pth", help="Checkpoint path")
    parser.add_argument("--plot", type=str, default="results/training_curves.png", help="Curves plot path")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint,
        plot_path=args.plot,
        patience=args.patience,
        device=args.device,
    )
