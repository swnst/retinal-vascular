"""
Deep Learning Model Architecture Module

Defines model architecture for Retinal Vessel Segmentation:
- Framework: PyTorch + segmentation_models_pytorch (smp)
- Architecture: U-Net with Pretrained ResNet Encoder (ImageNet weights)
- Input: RGB image (512x512x3)
- Output: Single-channel binary mask logits or probabilities
"""

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
    Construct and return U-Net architecture via segmentation_models_pytorch (smp).

    Parameters
    ----------
    encoder_name : str, optional
        Backbone encoder name, e.g., 'resnet34', 'resnet50' (default: 'resnet34').
    encoder_weights : str or None, optional
        Pretrained encoder weights, e.g., 'imagenet' or None (default: 'imagenet').
    in_channels : int, optional
        Number of input channels (default: 3).
    classes : int, optional
        Number of output classes for segmentation (default: 1).
    activation : str or None, optional
        Optional post-activation such as 'sigmoid' (default: None for raw logits).

    Returns
    -------
    nn.Module
        smp.Unet instance.
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
    Count model parameters.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to inspect.

    Returns
    -------
    dict[str, int]
        Dictionary with total, trainable, and non-trainable parameter counts.
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
    Helper function to instantiate default U-Net ResNet-34 model.
    """
    weights = "imagenet" if pretrained else None
    return get_model(
        encoder_name="resnet34",
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
    )
