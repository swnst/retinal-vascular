"""
Vessel Segmentation Evaluation Pipeline

Evaluates retinal vessel segmentation model across datasets:
- DRIVE
- HRF
- STARE

Computes standard clinical segmentation metrics:
- Accuracy (ACC)
- Sensitivity / Recall (SE)
- Specificity (SP)
- F1-Score / Dice Coefficient
- Area Under ROC Curve (AUC-ROC)
"""

import sys
from pathlib import Path
from typing import Sequence

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from dl_pipeline.predict import predict_vessel_mask, load_model_checkpoint
from dl_pipeline.data_utils import discover_dataset_samples
from evaluation.metrics import pixel_metrics, compute_auc_roc


def evaluate_samples(
    model: nn.Module,
    samples: Sequence[tuple[str, str]],
    dataset_name: str,
    device: str | torch.device = "cpu",
    threshold: float = 0.60,
    use_tta: bool = False,
    max_eval_samples: int | None = None,
) -> dict[str, float]:
    """
    Evaluate sample images in a dataset and return averaged metrics.
    """
    eval_list = list(samples[:max_eval_samples]) if max_eval_samples else list(samples)
    if not eval_list:
        return {
            "dataset": dataset_name,
            "num_samples": 0,
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1_score": 0.0,
            "dice": 0.0,
            "auc_roc": 0.0,
        }

    accs, sens, specs, f1s, dices, aucs = [], [], [], [], [], []

    for img_path, mask_path in eval_list:
        bin_mask, prob_map = predict_vessel_mask(
            model=model,
            image=img_path,
            device=device,
            threshold=threshold,
            return_original_size=True,
            use_tta=use_tta,
        )

        with Image.open(mask_path) as m_pil:
            gt_mask = np.array(m_pil.convert("L"))

        if gt_mask.shape != bin_mask.shape:
            gt_resized = Image.fromarray(gt_mask).resize((bin_mask.shape[1], bin_mask.shape[0]), Image.NEAREST)
            gt_mask = np.array(gt_resized)

        metrics = pixel_metrics(bin_mask, gt_mask, threshold=threshold)
        auc = compute_auc_roc(prob_map, gt_mask)

        accs.append(metrics["accuracy"])
        sens.append(metrics["sensitivity"])
        specs.append(metrics["specificity"])
        f1s.append(metrics["f1"])
        dices.append(metrics["dice"])
        aucs.append(auc)

    return {
        "dataset": dataset_name,
        "num_samples": len(eval_list),
        "accuracy": float(np.mean(accs)),
        "sensitivity": float(np.mean(sens)),
        "specificity": float(np.mean(specs)),
        "f1_score": float(np.mean(f1s)),
        "dice": float(np.mean(dices)),
        "auc_roc": float(np.mean(aucs)),
    }


def generate_markdown_report(results: list[dict[str, float]], output_path: str | Path) -> str:
    """
    Generate Markdown summary table report and save to disk.
    """
    lines = [
        "# Retinal Vessel Segmentation Evaluation Report",
        "",
        "| Dataset | Samples | Accuracy (ACC) | Sensitivity (SE) | Specificity (SP) | F1 / Dice Score | AUC-ROC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        lines.append(
            f"| **{r['dataset']}** | {r['num_samples']} | "
            f"{r['accuracy']:.4f} | {r['sensitivity']:.4f} | "
            f"{r['specificity']:.4f} | {r['dice']:.4f} | {r['auc_roc']:.4f} |"
        )

    lines.append("")
    content = "\n".join(lines)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content, encoding="utf-8")
    return content


def run_full_evaluation(
    model_or_checkpoint: nn.Module | str | Path,
    data_dir: str | Path = "data",
    output_report_path: str | Path = "results/vessel_evaluation_report.md",
    device: str | torch.device = "cpu",
    threshold: float = 0.60,
    use_tta: bool = False,
    max_eval_samples: int | None = None,
) -> list[dict[str, float]]:
    """
    Run evaluation across DRIVE, HRF, and STARE datasets.
    """
    if isinstance(model_or_checkpoint, (str, Path)):
        model = load_model_checkpoint(model_or_checkpoint, device=device)
    else:
        model = model_or_checkpoint

    data_p = Path(data_dir)
    results = []

    for ds_name in ["DRIVE", "HRF", "STARE"]:
        ds_path = data_p / ds_name
        if not ds_path.exists():
            continue
        samples = discover_dataset_samples(ds_path, dataset_name=ds_name)
        if not samples:
            continue

        print(f"Evaluating {ds_name} ({len(samples)} samples)...")
        res = evaluate_samples(
            model=model,
            samples=samples,
            dataset_name=ds_name,
            device=device,
            threshold=threshold,
            use_tta=use_tta,
            max_eval_samples=max_eval_samples,
        )
        results.append(res)

    generate_markdown_report(results, output_report_path)
    print(f"Evaluation report saved to: {output_report_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Vessel Segmentation Model")
    parser.add_argument("--checkpoint", type=str, default="models/best_unet_model_v2.pth")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/vessel_evaluation_report.md")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    run_full_evaluation(
        model_or_checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_report_path=args.output,
        device=args.device,
        use_tta=args.tta,
        threshold=args.threshold,
    )
