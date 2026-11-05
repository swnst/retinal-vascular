"""
Inference and Prediction Pipeline Module

Helper functions for retinal vessel segmentation inference:
- Load model checkpoint weights (.pth / .pt)
- Preprocess and resize input image to model dimensions (512x512)
- Test-Time Augmentation (TTA) with horizontal flip averaging
- Restore predicted vessel probability maps and binary masks to original resolution
"""

from pathlib import Path
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

    try:
        ckpt = torch.load(checkpoint_p, map_location=device_obj, weights_only=True)
    except Exception:
        ckpt = torch.load(checkpoint_p, map_location=device_obj, weights_only=False)
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


def _normalize_input_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha_raw = arr[:, :, 3:4]
        color_raw = arr[:, :, :3]
        if np.issubdtype(alpha_raw.dtype, np.floating):
            max_a = float(np.max(alpha_raw))
            alpha = np.clip(alpha_raw if max_a <= 1.05 else (alpha_raw / max(255.0, max_a)), 0.0, 1.0)
        elif alpha_raw.dtype == np.uint16:
            max_a = float(np.max(alpha_raw))
            denom_a = 65535.0 if max_a > 4095.0 else (max_a if max_a > 255.0 else 255.0)
            alpha = np.clip(alpha_raw.astype(np.float32) / denom_a, 0.0, 1.0)
        else:
            alpha = np.clip(alpha_raw.astype(np.float32) / 255.0, 0.0, 1.0)

        if np.issubdtype(color_raw.dtype, np.floating):
            min_c = float(np.min(color_raw))
            max_c = float(np.max(color_raw))
            if min_c >= -0.05 and max_c <= 1.05:
                color_float = np.clip(color_raw.astype(np.float32), 0.0, 1.0)
            elif min_c < -0.05:
                rng = max(1e-6, max_c - min_c)
                color_float = np.clip((color_raw.astype(np.float32) - min_c) / rng, 0.0, 1.0)
            elif max_c > 255.0:
                color_float = np.clip(color_raw.astype(np.float32) / max_c, 0.0, 1.0)
            else:
                color_float = np.clip(color_raw.astype(np.float32) / 255.0, 0.0, 1.0)
        elif color_raw.dtype == np.uint16:
            max_c = float(np.max(color_raw))
            denom_c = 65535.0 if max_c > 4095.0 else (max_c if max_c > 255.0 else 255.0)
            color_float = np.clip(color_raw.astype(np.float32) / denom_c, 0.0, 1.0)
        else:
            color_float = np.clip(color_raw.astype(np.float32) / 255.0, 0.0, 1.0)

        rgb_float = color_float * alpha
        return np.clip(rgb_float, 0.0, 1.0).astype(np.float32)

    if np.issubdtype(arr.dtype, np.floating):
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if min_v >= -0.05 and max_v <= 1.05:
            return np.clip(arr.astype(np.float32), 0.0, 1.0)
        elif min_v < -0.05:
            rng = max(1e-6, max_v - min_v)
            return np.clip((arr.astype(np.float32) - min_v) / rng, 0.0, 1.0)
        elif max_v > 255.0:
            return np.clip(arr.astype(np.float32) / max_v, 0.0, 1.0)
        else:
            return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
    elif arr.dtype == np.uint16:
        max_v = float(np.max(arr))
        denom = 65535.0 if max_v > 4095.0 else (max_v if max_v > 255.0 else 255.0)
        return np.clip(arr.astype(np.float32) / denom, 0.0, 1.0).astype(np.float32)
    else:
        return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)


# --- Inference Pipeline ---
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
        float_img = np.array(pil_img, dtype=np.float32) / 255.0
    elif isinstance(image, Image.Image):
        float_img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        float_img = _normalize_input_array(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    orig_h, orig_w = float_img.shape[:2]

    interp_down = cv2.INTER_AREA if max(orig_h, orig_w) > 1024 else cv2.INTER_LINEAR
    resized_img = cv2.resize(float_img, (img_size, img_size), interpolation=interp_down)
    tensor_in = torch.from_numpy(resized_img).permute(2, 0, 1).unsqueeze(0).to(device_obj)

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


predict_single_image = predict_vessel_mask

