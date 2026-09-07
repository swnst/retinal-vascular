"""
Unit tests for Week 3 & Week 4 Deliverables:
- Model instantiation, parameter counting, and forward pass
- CombinedLoss and DiceLoss calculation
- EarlyStopping logic
- Inference, TTA, and original size restoration
- Pixel metrics and AUC-ROC evaluation
"""

import numpy as np
import pytest
import torch

from dl_pipeline.model import get_model, count_parameters, create_unet_resnet34
from dl_pipeline.train import CombinedLoss, DiceLoss, EarlyStopping, compute_batch_dice
from dl_pipeline.predict import predict_vessel_mask
from evaluation.metrics import pixel_metrics, compute_auc_roc


def test_model_instantiation_and_parameters():
    """ทดสอบการสร้าง U-Net ResNet-34 และนับพารามิเตอร์"""
    model = get_model(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    assert model is not None
    assert isinstance(model, torch.nn.Module)

    param_info = count_parameters(model)
    assert "total_params" in param_info
    assert "trainable_params" in param_info
    assert param_info["total_params"] > 20_000_000  # ResNet-34 U-Net มีประมาณ ~24.4M params
    assert param_info["trainable_params"] == param_info["total_params"]


def test_model_forward_pass():
    """ทดสอบ forward pass ขนาด tensor ถูกต้อง (B, C, H, W) -> (B, 1, H, W)"""
    model = get_model(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.eval()

    # จำลอง batch ขนาด 2 ภาพ ขนาด 512x512
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1, 512, 512)
    assert not torch.isnan(output).any()


def test_dice_loss_and_combined_loss():
    """ทดสอบการคำนวณ DiceLoss และ CombinedLoss โดยไม่ error"""
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = criterion(logits, targets)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0

    # ตรวจสอบ backward pass
    loss.backward()
    assert logits.grad is not None


def test_early_stopping():
    """ทดสอบระบบ EarlyStopping ทั้งโหมด max และ min"""
    es = EarlyStopping(patience=3, mode="max")
    assert not es(0.5)
    assert not es(0.6)  # ดีขึ้น
    assert not es(0.55) # แย่ลง count=1
    assert not es(0.54) # แย่ลง count=2
    assert es(0.50)     # แย่ลง count=3 -> Early stop!


def test_predict_vessel_mask_and_resize_restoration():
    """ทดสอบ Inference pipeline: รับภาพขนาดใดก็ได้ (เช่น 600x800) และ Resize คืนขนาดเดิมได้อย่างแม่นยำ"""
    model = get_model(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.eval()

    orig_h, orig_w = 600, 800
    dummy_img = np.random.randint(0, 256, (orig_h, orig_w, 3), dtype=np.uint8)

    # ทดสอบแบบปกติ
    bin_mask, prob_map = predict_vessel_mask(
        model=model,
        image=dummy_img,
        device="cpu",
        threshold=0.5,
        return_original_size=True,
        use_tta=False,
    )

    assert bin_mask.shape == (orig_h, orig_w)
    assert prob_map.shape == (orig_h, orig_w)
    assert set(np.unique(bin_mask)).issubset({0, 1})
    assert 0.0 <= prob_map.min() and prob_map.max() <= 1.0

    # ทดสอบเปิดใช้งาน TTA
    bin_mask_tta, prob_map_tta = predict_vessel_mask(
        model=model,
        image=dummy_img,
        device="cpu",
        threshold=0.5,
        return_original_size=True,
        use_tta=True,
    )
    assert bin_mask_tta.shape == (orig_h, orig_w)
    assert prob_map_tta.shape == (orig_h, orig_w)


def test_evaluation_metrics():
    """ทดสอบการคำนวณ Metrics (Accuracy, Sensitivity, Specificity, F1/Dice, AUC-ROC)"""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[40:60, 40:60] = 1  # 400 พิกเซลบวก

    pred = np.zeros((100, 100), dtype=np.float32)
    pred[40:60, 40:60] = 0.9  # TP 400
    pred[0:10, 0:10] = 0.8    # FP 100

    metrics = pixel_metrics(pred, gt, threshold=0.5)
    assert "accuracy" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "f1" in metrics
    assert "dice" in metrics

    # TP = 400, FN = 0 -> Sensitivity = 1.0
    assert pytest.approx(metrics["sensitivity"], 1e-4) == 1.0
    assert metrics["specificity"] < 1.0  # มี FP

    auc = compute_auc_roc(pred, gt)
    assert 0.5 <= auc <= 1.0
