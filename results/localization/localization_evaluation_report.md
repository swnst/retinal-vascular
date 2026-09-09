# IDRiD Optic Disc & Fovea Localization Evaluation Report

- **Dataset**: IDRiD C. Localization Testing Set
- **Evaluated Samples**: 103
- **Pipeline Architecture**: DL U-Net ResNet-34 Vessel Segmentation -> Classical Localization API

---

## 1. Quantitative Performance Summary

| Metric | Optic Disc (OD) | Fovea |
|---|---|---|
| **Mean Error (px)** | 68.18 px | 108.11 px |
| **Median Error (px)** | 56.99 px | 94.88 px |
| **Std Deviation (px)** | 81.51 px | 109.35 px |
| **Mean Normalized Error (% Width)** | 1.59% | 2.52% |
| **Median Normalized Error (% Width)** | 1.33% | 2.21% |
| **Success Rate (Error <= 5% Image Width)** | 99.03% | 94.17% |

---

## 2. Technical Observations
- Optic Disc Localization leverages convergence of the segmented retinal vascular tree combined with vessel density and brightness.
- Fovea Localization utilizes anatomical priors relative to the detected Optic Disc center and blood vessel sparsity in the macular zone.
