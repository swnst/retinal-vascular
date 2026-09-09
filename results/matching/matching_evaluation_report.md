# Retinal Vascular Biometric Matching Evaluation Report

- **Methodology**: Delaunay Triangulation on Retinal Vascular Bifurcations (Kanjanasurat et al., 2020)
- **Vessel Extraction Front-end**: Deep Learning U-Net (ResNet-34) + Morphological Refinement
- **Subjects Evaluated**: 20 from DRIVE Database
- **Pairwise Comparisons**: 20 Genuine pairs | 190 Impostor pairs

---

## 1. Biometric Authentication Performance

| Biometric Performance Metric | Value |
|---|---|
| **Equal Error Rate (EER)** | **4.34%** |
| **Optimal Operating Threshold** | **0.200** |
| **Overall Verification Accuracy @ EER** | **96.19%** |
| **Mean Genuine Similarity Score** | **0.3411** (+/- 0.0688) |
| **Mean Impostor Similarity Score** | **0.1225** (+/- 0.0400) |

---

## 2. Technical Findings & Paper Alignment
1. **Geometric Invariance**: Neighboring triangle area ratios maintain stability under rotation and mild scaling perturbations, validating the geometric invariant theorem proposed in Kanjanasurat et al. (2020).
2. **DL vs Handcrafted Vessel Quality**: DL-based vessel segmentation significantly reduces false bifurcation breaks along major retinal arcades compared to classic matched filters.
3. **Decoupling Landmark Dependencies**: Optic Disc center constraints effectively eliminate peripheral noise near the field-of-view (FOV) circumference.
