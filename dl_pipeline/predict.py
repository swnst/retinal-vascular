"""
Inference and Prediction Pipeline Module

สคริปต์และฟังก์ชันสำหรับการทำ Inference ด้วยโมเดลที่ผ่านการเทรนแล้ว:
- โหลด checkpoint น้ำหนักโมเดล (.pth / .pt)
- รับภาพ fundus ขนาดจริง -> Preprocess & Resize สู่ 512x512 -> Forward pass
- Binarize ผลลัพธ์ด้วยเกณฑ์ threshold
- Resize vessel map ที่ทำนายได้กลับเป็นขนาดต้นฉบับเพื่อส่งต่อให้ Classical Pipeline

โมดูลนี้จะถูกพัฒนาใน Week 4 เป็นต้นไป
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
    โหลดโมเดลพร้อมน้ำหนัก checkpoint (.pth)
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
    ทำนาย Vessel Segmentation Mask จากภาพจอประสาทตา

    Parameters
    ----------
    model : nn.Module
        โมเดล U-Net ที่พร้อมทำนาย
    image : np.ndarray | Image.Image | str | Path
        ภาพอินพุต (RGB หรือ Grayscale, numpy array หรือ PIL image หรือ path)
    device : str | torch.device, optional
        อุปกรณ์ที่ใช้คำนวณ (default: 'cpu')
    threshold : float, optional
        เกณฑ์ตัดสิน Binary mask (default: 0.5)
    return_original_size : bool, optional
        ถ้า True จะขยายผลลัพธ์กลับสู่ขนาดเดิมของภาพอินพุต (default: True)
    use_tta : bool, optional
        เปิดใช้งาน Test-Time Augmentation (Horizontal Flip) (default: False)
    img_size : int, optional
        ขนาดความกว้างยาวที่จะป้อนเข้าโมเดล (default: 512)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (binary_mask, prob_map)
        - binary_mask: array ค่า 0 หรือ 1 (uint8)
        - prob_map: probability map ช่วง [0.0, 1.0] (float32)
    """
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    # จัดการประเภทของภาพอินพุต
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

    # Resize สู่ 512x512
    resized_img = cv2.resize(img_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor_in = torch.from_numpy(resized_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        logits = model(tensor_in)
        prob = torch.sigmoid(logits)

        if use_tta:
            # Test-Time Augmentation: Horizontal Flip
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

