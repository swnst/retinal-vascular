"""
Deep Learning Model Architecture Module

โมดูลนิยามโครงสร้างสถาปัตยกรรมโมเดลสำหรับ Retinal Vessel Segmentation
ตาม Technical Decisions ใน ROADMAP.md:
- Framework: PyTorch + segmentation_models_pytorch (smp)
- Architecture: U-Net with Pretrained ResNet-34 Encoder (ImageNet weights)
- Input: RGB image ขนาด 512x512x3
- Output: Single channel binary mask (logits หรือ probability ผ่าน sigmoid)

โมดูลนี้จะถูกพัฒนาอย่างสมบูรณ์ใน Week 3
"""

from typing import Literal
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    activation: str | None = None,
) -> nn.Module:
    """
    สร้างและคืนค่าสถาปัตยกรรม U-Net ด้วย segmentation_models_pytorch (smp)

    Parameters
    ----------
    encoder_name : str, optional
        ชื่อของ backbone encoder เช่น 'resnet34', 'resnet50' (default: 'resnet34')
    encoder_weights : str or None, optional
        Pretrained weights ของ encoder เช่น 'imagenet' หรือ None (default: 'imagenet')
    in_channels : int, optional
        จำนวน channel ของอินพุต เช่น 3 (RGB) หรือ 1 (Green channel) (default: 3)
    classes : int, optional
        จำนวน class ของเอาต์พุต สำหรับ binary segmentation คือ 1 (default: 1)
    activation : str or None, optional
        Activation function ท้ายโมเดล เช่น 'sigmoid' หรือ None (default: None คืน logits)

    Returns
    -------
    nn.Module
        smp.Unet instance
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    คำนวณจำนวน parameter ภายในโมเดล

    Parameters
    ----------
    model : nn.Module
        PyTorch โมเดลที่ต้องการนับพารามิเตอร์

    Returns
    -------
    dict[str, int]
        {
            "total_params": จำนวนพารามิเตอร์ทั้งหมด,
            "trainable_params": จำนวนพารามิเตอร์ที่เปิดให้อัปเดต (requires_grad=True),
            "non_trainable_params": จำนวนพารามิเตอร์ที่ freeze ไว้
        }
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def create_unet_resnet34(
    pretrained: bool = True,
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """
    Helper function สำหรับสร้าง U-Net ResNet-34 ตามข้อตกลง Week 3
    """
    weights = "imagenet" if pretrained else None
    return get_model(
        encoder_name="resnet34",
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
    )

