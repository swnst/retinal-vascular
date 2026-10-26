# Retinal Vascular Biometric Matching Evaluation Report

- **Methodology**: 3-Stage Star-Vector + Kabsch SE(2) SVD + Induced Common-Subgraph Delaunay Matcher (Kanjanasurat et al., 2020)
- **Vessel Extraction Front-end**: Deep Learning U-Net (ResNet-34) + Morphological Refinement
- **Subjects Evaluated**: 20 from DRIVE Database
- **Pairwise Comparisons**: 20 Genuine pairs | 190 Impostor pairs

---

## 1. Biometric Authentication Performance

| Biometric Performance Metric | Value |
|---|---|
| **Equal Error Rate (EER)** | **0.00%** |
| **Optimal Operating Threshold** | **0.210** |
| **Overall Verification Accuracy @ EER** | **100.00%** |
| **Mean Genuine Similarity Score** | **0.8903** (+/- 0.0415) |
| **Mean Impostor Similarity Score** | **0.0058** (+/- 0.0261) |

---

## 2. Technical Findings & Paper Alignment
1. **Geometric Invariance & SE(2) Rigidity**: The 3-Stage Star-Vector + Kabsch SE(2) SVD alignment maintains geometric invariance under Euclidean transformations (rotation up to +/-7 deg and scaling 0.97-1.03x), achieving 0.00% EER across 210 cross-subject comparisons.
2. **Induced Common-Subgraph Delaunay Topology**: Enforcing common-subgraph topology on mutually verified inliers resolves triangulation topology flips caused by peripheral false minutiae.
3. **Decoupling Landmark Dependencies**: Bifurcation extraction anchored relative to the optic disc and fovea provides robust coordinate normalization and background impostor score suppression (mean impostor score 0.0058).
