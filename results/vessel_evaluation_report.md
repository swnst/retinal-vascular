# Retinal Vessel Segmentation Evaluation Report

## 1. Held-out Validation Set (Zero Data Leakage)

| Evaluation Split | Samples | Accuracy (ACC) | Sensitivity (SE) | Specificity (SP) | F1 / Dice Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out Val Set (20%)** | 17 | 0.9551 | 0.6905 | 0.9835 | 0.7433 | 0.9572 |

## 2. Per-Dataset Evaluation (Inside-FOV Benchmark)

| Dataset | Samples | Accuracy (ACC) | Sensitivity (SE) | Specificity (SP) | F1 / Dice Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DRIVE** | 20 | 0.9649 | 0.7647 | 0.9844 | 0.7945 | 0.9790 |
| **HRF** | 45 | 0.9566 | 0.6738 | 0.9844 | 0.7344 | 0.9512 |
| **STARE** | 20 | 0.9573 | 0.7533 | 0.9816 | 0.7920 | 0.9663 |
