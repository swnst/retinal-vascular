# Retinal Vessel Segmentation Evaluation Report

## 1. Held-out Validation Set (Zero Data Leakage)

| Evaluation Split | Samples | Accuracy (ACC) | Sensitivity (SE) | Specificity (SP) | F1 / Dice Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-out Val Set (20%)** | 17 | 0.9515 (+/- 0.0076) | 0.7240 (+/- 0.0407) | 0.9782 (+/- 0.0056) | 0.7554 (+/- 0.0466) | 0.9562 (+/- 0.0145) |

## 2. Per-Dataset Evaluation (Inside-FOV Benchmark)

| Dataset | Samples | Accuracy (ACC) | Sensitivity (SE) | Specificity (SP) | F1 / Dice Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DRIVE** | 20 | 0.9473 (+/- 0.0074) | 0.7503 (+/- 0.0360) | 0.9756 (+/- 0.0061) | 0.7809 (+/- 0.0178) | 0.9710 (+/- 0.0060) |
| **HRF** | 45 | 0.9541 (+/- 0.0057) | 0.7245 (+/- 0.0435) | 0.9770 (+/- 0.0064) | 0.7384 (+/- 0.0381) | 0.9506 (+/- 0.0136) |
| **STARE** | 20 | 0.9502 (+/- 0.0110) | 0.7500 (+/- 0.0373) | 0.9791 (+/- 0.0049) | 0.7915 (+/- 0.0340) | 0.9654 (+/- 0.0100) |
