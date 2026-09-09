"""
Inference and Prediction Pipeline Module

Helper functions for retinal vessel segmentation inference:
- Load model checkpoint weights (.pth / .pt)
- Preprocess and resize input image to model dimensions (512x512)
- Test-Time Augmentation (TTA) with horizontal flip averaging
- Restore predicted vessel probability maps and binary masks to original resolution
"""

from pathlib import Path
from typing import Literal
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from dl_pipeline.model import get_model


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    encoder_name: str = "resnet34",
    in_channels: int = 3,
) -> nn.Module:
    """
    Load model with saved checkpoint weights (.pth).
    """
    device_obj = torch.device(device)
    model = get_model(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    )
    checkpoint_p = Path(checkpoint_path)
    if not checkpoint_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_p}")

    ckpt = torch.load(checkpoint_p, map_location=device_obj)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("encoder") or k.startswith("decoder") for k in ckpt.keys()):
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format in {checkpoint_p}")

    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()
    return model


def predict_vessel_mask(
    model: nn.Module,
    image: np.ndarray | Image.Image | str | Path,
    device: str | torch.device = "cpu",
    threshold: float = 0.60,
    return_original_size: bool = True,
    use_tta: bool = False,
    img_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict vessel segmentation mask from a fundus image.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    image : np.ndarray | Image.Image | str | Path
        Input image (RGB or grayscale, numpy array, PIL Image, or file path).
    device : str | torch.device, optional
        Computation device (default: 'cpu').
    threshold : float, optional
        Probability binarization threshold (default: 0.60).
    return_original_size : bool, optional
        Whether to restore predicted maps to the original image resolution (default: True).
    use_tta : bool, optional
        Enable Test-Time Augmentation via horizontal flip (default: False).
    img_size : int, optional
        Model input resolution (default: 512).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (binary_mask, prob_map)
        - binary_mask: array of 0 or 1 (uint8).
        - prob_map: probability values in [0.0, 1.0] (float32).
    """
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    if isinstance(image, (str, Path)):
        pil_img = Image.open(image).convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            img_np = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_np = np.repeat(image, 3, axis=2)
        else:
            img_np = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    orig_h, orig_w = img_np.shape[:2]

    resized_img = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor_in = torch.from_numpy(resized_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        logits = model(tensor_in)
        prob = torch.sigmoid(logits)

        if use_tta:
            tensor_flipped = torch.flip(tensor_in, dims=[3])
            logits_flipped = model(tensor_flipped)
            prob_flipped = torch.sigmoid(logits_flipped)
            prob_unflipped = torch.flip(prob_flipped, dims=[3])
            prob = (prob + prob_unflipped) / 2.0

        prob_map_512 = prob.squeeze().cpu().numpy()

    if return_original_size and (orig_h != img_size or orig_w != img_size):
        prob_map = cv2.resize(prob_map_512, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        prob_map = prob_map_512

    binary_mask = (prob_map >= threshold).astype(np.uint8)

    return binary_mask, prob_map
