# VERA: Vascular Extraction and Retinal Authentication Platform

### Personal Biometric Verification and 3D Ocular Anatomy System via Deep Learning Vessel Perception, Rutovitz Crossing Number Topology, and Kabsch SE(2) Induced Delaunay Triangulation

Department of IoT and Information Engineering  
Program in IoT and Information Engineering and Industrial Physics (Dual Degree)  
School of Engineering, King Mongkut's Institute of Technology Ladkrabang (KMITL)

- Project Authors:
  - Wiwat Suwannasit (วิวรรธน์ สุวรรณสิทธิ์)
  - Dvis Vichitlaksana (ดวิษ วิจิตรลักษณา)
- Academic Advisors:
  - Asst. Prof. Isoon Kanjanasurat (ผศ. อิสูรย์ กาญจนสุรัตน์)
  - Asst. Prof. Dr. Nitjaree Satayarak (ผศ. ดร. นิจจารีย์ สัตยารักษ์)
- Foundational Literature:
  - Kanjanasurat et al. (2020), "Vascular Extraction by using matched filter on retinal image", *Journal of Physics: Conference Series*, DOI: 10.1088/1742-6596/1457/1/012013, URL: https://iopscience.iop.org/article/10.1088/1742-6596/1457/1/012013
  - Kanjanasurat, Purahong, Pintavirooj, Satayarak, and Benjangkaprasert (2020), "Blood Vessel Extraction and Optic Disk Localization for Diabetic Retinopathy", *ACM ICBET*, DOI: 10.1145/3397391.3397425, URL: https://dl.acm.org/doi/10.1145/3397391.3397425
  - Kanjanasurat et al. (2020), "Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern", *IJICIC*, Vol. 16, No. 3, pp. 879-897, DOI: 10.24507/ijicic.16.03.879 (JaLC), URL: http://www.ijicic.org/ijicic-160307.pdf

---

## 1. Executive Summary and Capstone Scope Analysis

VERA is an end-to-end biometric verification and ocular anatomical analysis system built on retinal vascular morphology. Unlike facial, voice, or epidermal fingerprint modalities, the vascular branching pattern of the human fundus oculi is internal, genetically unique (even between monozygotic twins), and topologically stable throughout adulthood.

### Initial Capstone I Target vs. Delivered System Reality

The original Capstone I project proposal specified a baseline proof-of-concept: evaluating standard U-Net segmentation on clean DRIVE fundus images, extracting basic bifurcation points, computing Delaunay triangle invariants, and rendering results in a basic standalone script.

The implemented platform exceeds the original targets across all core engineering dimensions:

| Evaluation Dimension | Initial Capstone I Target | Delivered System Reality | Status |
| :--- | :--- | :--- | :---: |
| **Vessel Segmentation** | Standard U-Net (~0.70 Dice on DRIVE) | ResNet-34 U-Net v3 (`best_unet_model_v3.pth`, 24.4M params) with Tripartite BCE + Soft Dice + Soft-clDice loss reaching **0.7809 Dice, 0.9710 ROC-AUC, 97.56% Specificity** on DRIVE (0.7554 Dice, 0.9562 AUC on zero-leakage held-out validation) | Exceeded Target |
| **Probability Binarization** | Static 0.50 cutoff | **Piecewise Quantile-Calibrated Adaptive Thresholding** ($`\tau_{\text{eff}} = \max(0.32, \min(0.60, 0.65 \cdot p_{95}))`$ for $`p_{95} < 0.85`$, else $`0.60`$) resilient to cataract opacity | Exceeded Target |
| **Landmark Localization** | Basic center heuristic on DRIVE | **3-Cue Structure-Tensor Convergence & Anisotropic Macular Prior** on 103 high-resolution IDRiD images (**99.03% OD Success, 94.17% Fovea Success**) | Exceeded Target |
| **Biometric Matcher** | Independent Delaunay Invariants (fails under vertex dropout) | **3-Stage Star-Vector + Kabsch SE(2) SVD Constellation Consensus + Induced Common-Subgraph Delaunay Triangulation** | Exceeded Target |
| **Clinical Robustness** | None (clean static scans only) | **11-Parameter Physical Degradation Suite**; passes 7-parameter `Judge Torture` stress at **63.4%** matching score (threshold: 20.0%) | Exceeded Target |
| **Software Architecture** | Standalone Python scripts | **Decoupled Architecture**: FastAPI 2.0 Backend + Next.js 16 App Router Turbopack Frontend | Exceeded Target |
| **Visualization and UI** | Static 2D plots | **3-Stage Three.js WebGL Studio**: 3D Retinal Globe with depth GLSL shader, Dual 3D Matcher with laser correspondence beams, and 4x4 Cohort Matrix | Exceeded Target |
| **Automated Testing** | Manual inspection | **Automated Suite of 40 Pytest Tests**, ESLint (0 errors), Strict TypeScript (0 errors), Next.js Turbopack build | Exceeded Target |

---

## 2. System Architecture

The platform architecture is organized into five modular subsystems:

```
[1. Ocular Image Input & 11-Param Degradation Suite]
                         │
                         ▼
[2. Deep Learning Perception (ResNet-34 U-Net v3 + TTA)]
                         │
                         ▼
[3. Quantile Adaptive Thresholding & Morphological Cleanup]
                         │
                         ▼
[4. Zhang-Suen Skeletonization & Rutovitz Crossing Number Minutiae]
                         │
                         ▼
[5. Kabsch SVD Rigid Consensus & Induced Subgraph Delaunay Matching]
                         │
                         ▼
[FastAPI 2.0 REST Services] <───> [Next.js 16 Three.js WebGL Studio]
```

### Module Pipeline Summary

1. **Input and Degradation Suite**: Ingests reference (Scan A) and target (Scan B) fundus photographs. An 11-parameter physical engine simulates real-world clinical stressors including head tilt ($\pm 45^\circ$), axial scale/zoom ($0.80\text{x} - 1.20\text{x}$), cataract media haze ($0.0 - 0.95$), corneal surface glare, sensor ISO noise, eyelid occlusion, spherical optical warp, and pulsatile intraocular pressure (IOP) elastic displacements ($\sigma = 6.0$ px).
2. **Deep Learning Perception Layer**: Executes inference via a 24.4M-parameter U-Net with ImageNet-pretrained ResNet-34 backbone. Includes Test-Time Augmentation (TTA: horizontal flip averaging) ingesting normalized RGB imagery.
3. **Adaptive Thresholding and Postprocessing**: Dynamically calculates effective binarization cutoffs based on the 95th percentile intensity of the predicted foreground probability within the Field of View (FOV). Removes isolated noise components smaller than 25 pixels and applies morphological binary hole filling.
4. **Anatomical Landmark Localization**: Detects the Optic Disc (OD) via 3-cue multi-color-space density, structure-tensor tangent ray voting, and pale hue fusion. Fovea centralis coordinates are extracted through an OD-anchored anisotropic 2D Gaussian spatial prior fused with macular darkness cues.
5. **Centerline Thinning and Minutiae Extraction**: Generates 1-pixel unitary skeletons via Zhang-Suen morphological thinning with short spur pruning. Minutiae points are classified using Rutovitz Crossing Number with digital grid diagonal rescue ($`(X \ge 3) \lor (X = 2 \land N_8 \ge 3)`$) and FOV-scaled greedy Euclidean non-maximum suppression ($`d_{\min} = \max(6.0, 0.012 \cdot D_{\text{FOV}})`$ px).
6. **3-Stage Constellation Alignment and Induced Subgraph Delaunay Matching**: Employs Stage 1 triangle invariant seeding with Kuhn-Munkres assignment, Stage 2 Star-Vector polar segment hypotheses with k-d tree inlier search and closed-form Kabsch $\text{SO}(2)$ SVD procrustes alignment, and Stage 3 Induced Common-Subgraph Delaunay Triangulation on mutual inliers, eliminating false rejections caused by independent Delaunay re-triangulation under point dropout.

---

## 3. Mathematical Foundations and Literature Citations

Every algorithmic component in the VERA platform is derived from verified mathematical literature:

### 3.1 CLAHE Illumination Equalization
To normalize intra-retinal illumination gradients across variable fundus pigmented backgrounds in classical matched filtering and stochastic training data augmentation:

$$h_{\text{clipped}}(n) = \min(h(n), \beta)$$
$$g(x,y) = \left[ g_{\max} - g_{\min} \right] \cdot P(f(x,y)) + g_{\min}$$

CLAHE is utilized in `classical_pipeline/bezier_matched_filter.py` (`clipLimit=2.0`, Green channel) and as a stochastic augmentation in `dl_pipeline/dataset.py` (`clip_limit=2.5`), while deterministic DL inference operates directly on normalized RGB images.

*Citation:* Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. *Graphics Gems IV*, Academic Press, pp. 474-485. DOI: 10.1016/B978-0-12-336156-1.50061-6.

### 3.2 High-Frequency Noise Estimation (MAD) and Gated Non-Local Means
Estimates background sensor noise standard deviation $\hat{\sigma}$ with a 50% statistical breakdown point across spatial Gaussian residuals $`r(x,y) = |I_G - G_{\sigma=0.8} \ast I_G|`$ strictly within the active Field of View (FOV):

$$\hat{\sigma} = \frac{\text{median}\left( r(x,y) \big|_{\text{FOV}} \right)}{0.6745}$$

When $\hat{\sigma} > 4.2$, adaptive Non-Local Means filtering is executed with filter strength $h = \min(11.0, \max(5.0, 0.65\hat{\sigma}))$, suppressing sensor noise while leaving clean clinical scans untouched.

*Citations:*
- Donoho, D. L., and Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455. DOI: 10.1093/biomet/81.3.425.
- Buades, A., Coll, B., and Morel, J.-M. (2005). A non-local algorithm for image denoising. *IEEE CVPR 2005*, pp. 60-65. DOI: 10.1109/CVPR.2005.38.

### 3.3 Composite Training Loss (BCE + Soft Sorensen-Dice + Soft-clDice)
Overcomes extreme class imbalance (~91% background vs. ~9% vascular foreground) while enforcing topological continuity in thin 1-pixel terminal capillaries:

$$\mathcal{L}_{\text{total}} = 0.35 \cdot \mathcal{L}_{\text{BCE}} + 0.40 \cdot \mathcal{L}_{\text{Dice}} + 0.25 \cdot \mathcal{L}_{\text{clDice}}$$
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$
$$\mathcal{L}_{\text{clDice}} = 1 - \frac{2 \cdot T_{\text{prec}}(S(P), Y) \cdot T_{\text{sens}}(S(Y), P)}{T_{\text{prec}}(S(P), Y) + T_{\text{sens}}(S(Y), P) + \epsilon}$$

where $S(X) = \text{ReLU}(X - \text{open}(X))$ performs differentiable soft skeletonization via min-max morphological pooling.

*Citations:*
- Milletari, F., Navab, N., and Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV 2016*, pp. 565-571. DOI: 10.1109/3DV.2016.79.
- Shit, S. et al. (2021). clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR 2021*, pp. 16555-16564. DOI: 10.1109/CVPR46437.2021.01629.

### 3.4 Piecewise Quantile-Calibrated Adaptive Thresholding
To ensure robust foreground binarization across severe inter-patient illumination variance, media opacities, and cataract haze:

$$\tau_{\text{eff}} = \begin{cases} \max\left(0.32, \min\left(0.60, 0.65 \cdot p_{95}\right)\right), & \text{if } p_{95} < 0.85 \\ 0.60, & \text{if } p_{95} \ge 0.85 \end{cases}$$

where $p_{95}$ is the 95th percentile intensity of the predicted foreground vascular probability map calculated strictly within the Field of View (FOV). Clamping to $[0.32, 0.60]$ prevents background noise breakthrough under high clarity while preserving faint vascular trunks under heavy scattering.

### 3.5 Morphological Skeletonization (Zhang-Suen and Lee Medial Axis)
Extracts unitary-width centerlines via parallel boundary pixel deletion based on neighbor count $B(P_1)$ and $0 \to 1$ transition count $A(P_1)$:

$$2 \le B(P_1) \le 6, \quad A(P_1) = 1$$

*Citations:*
- Zhang, T. Y., and Suen, C. Y. (1984). A fast parallel algorithm for thinning digital patterns. *Communications of the ACM*, 27(3), 236-239. DOI: 10.1145/357994.358023.
- Lee, T.-C., Kashyap, R. L., and Chu, C.-N. (1994). Building Skeleton Models via 3-D Medial Surface Axis Thinning Algorithms. *CVGIP: Graphical Models and Image Processing*, 56(6), 462-478. DOI: 10.1006/cgip.1994.1042.

### 3.6 Minutiae Extraction (Rutovitz Crossing Number)
Classifies skeleton pixels into terminal endpoints ($X=1$), vessel continuations ($X=2$), bifurcations ($X=3$), and crossovers ($X=4$):

$$X(P) = \frac{1}{2} \sum_{i=1}^8 \left| P_i - P_{i+1} \right|, \quad P_9 = P_1$$

VERA extends this with a discrete grid diagonal rescue condition: $`\text{is\_bifurcation} = (S(P) > 0) \land (X(P) \ge 3 \lor (X(P) = 2 \land N_8(P) \ge 3))`$ (Line 47), followed by FOV-scaled greedy Euclidean non-maximum suppression ($`d_{\min} = \max(6.0, 0.012 \cdot D_{\text{FOV}})`$ px).

*Citations:*
- Rutovitz, D. (1966). Pattern recognition. *Journal of the Royal Statistical Society: Series A*, 129(4), 504-530. DOI: 10.2307/2982255.
- Rosenfeld, A. (1970). Connectivity in Digital Pictures. *Journal of the ACM*, 17(1), 146-160. DOI: 10.1145/321556.321570.

### 3.7 Closed-Form Spatial Alignment (Kabsch SVD Procrustes)
Derives the optimal rigid rotation matrix $R \in \text{SO}(2)$ and translation vector $t \in \mathbb{R}^2$ minimizing root-mean-square alignment deviation:

$$H = \sum_{i=1}^k (p_i - \bar{p})(q_i - \bar{q})^T = U \Sigma V^T$$
$$R = V \begin{pmatrix} 1 & 0 \\ 0 & \det(V U^T) \end{pmatrix} U^T, \quad t = \bar{q} - R \bar{p}$$

*Citations:*
- Kabsch, W. (1976). A solution for the best rotation to relate two sets of vectors. *Acta Crystallographica Section A*, 32(5), 922-923. DOI: 10.1107/S0567739476001873.
- Kabsch, W. (1978). A discussion of the solution for the best rotation to relate two sets of vectors. *Acta Crystallographica Section A*, 34(5), 827-828. DOI: 10.1107/S0567739478001680.

### 3.8 Delaunay Invariant Geometry and Induced Common-Subgraph Matching
Triangulates minutiae graphs and computes interior angle, side ratio, and adjacent area ratio invariant descriptors:

$$\boldsymbol{\theta} = (\theta_{\min}, \theta_{\text{mid}}, \theta_{\max}), \quad \boldsymbol{\rho}_{\text{side}} = \left(\frac{s_1}{s_3}, \frac{s_2}{s_3}\right), \quad r_{\text{adj}} = \frac{\min(A_i, A_j)}{\max(A_i, A_j)}$$
$$C(i, j) = 0.55 \cdot \frac{\Delta\bar{\theta}}{\tau_\theta} + 0.25 \cdot \min_{a, b} \frac{|r_{A,a} - r_{B,b}|}{\tau_r} + 0.20 \cdot \frac{1 - \rho_{\text{area}}}{0.40}$$

To prevent catastrophic false rejections from vertex dropout where independent Delaunay graphs re-triangulate into conflicting topologies, VERA executes **Induced Common-Subgraph Delaunay Triangulation**:
1. Spatial Minutiae Consensus: Candidate minutiae pairs are identified via Kuhn-Munkres on $`C(i,j) < 0.95`$ (Stage 1) and Star-Vector polar segment hypotheses ($`10 \le \ell \le 140`$ px, $`|\Delta\ell|/\max \le 0.10`$, $`|\Delta\theta| \le 55^\circ`$, Stage 2), aligned globally via Kabsch SE(2) SVD.
2. Mutual Inlier Gating: Candidate inliers are queried via `cKDTree` within `kp_tol = 11.5 px` with distance-sorted bijective deduplication.
3. Induced Graph Triangulation: Delaunay triangulation is constructed over the mutual inliers $\mathcal{P}_{\text{mutual}}$, and similarity score is computed as:
   $$S(A, B) = \min\left(1.0, \, \frac{|\mathcal{M}_{\text{tri}}|}{\max\left(1, \sqrt{|\mathcal{T}_A| \cdot |\mathcal{T}_B|}\right)}\right), \quad \text{Verified if } S(A, B) \ge 0.20$$

*Citation:* Kanjanasurat, I., Purahong, B., Aoyama, H., Benjangkaprasert, C., and Pintavirooj, C. (2020). Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern. *IJICIC*, 16(3), 879-897. DOI: 10.24507/ijicic.16.03.879 (JaLC), URL: http://www.ijicic.org/ijicic-160307.pdf.

### 3.9 Bipartite Assignment (Kuhn-Munkres)
Solves optimal bipartite assignment between Stage 1 Delaunay triangle invariant descriptors $`C(i,j)`$ (transformed candidate minutiae in Stage 2/3 are subsequently matched via `cKDTree` within `kp_tol = 11.5 px` and greedy distance-sorted bijective deduplication):

$$\min \sum_{i=1}^m \sum_{j=1}^n C_{i,j} X_{i,j}, \quad \text{where } C_{i,j} \text{ represents feature or geometric distance}$$

*Citations:*
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2), 83-97. DOI: 10.1002/nav.3800020109.
- Munkres, J. (1957). Algorithms for the assignment and transportation problems. *Journal of the Society for Industrial and Applied Mathematics*, 5(1), 32-38. DOI: 10.1137/0105003.

### 3.10 Spatial Keypoint Indexing (k-d Tree)
Reduces geometric proximity queries from $O(N \cdot M)$ to $O(N \log M)$:

*Citation:* Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), 509-517. DOI: 10.1145/361002.361007.

### 3.11 Optical and Physical Degradation Mechanics
- **Scale-Invariant Radial Optical Lens Distortion:** $`\tilde{r}_u = \frac{r_u}{r_{\max}}`$, $`x_{\text{dist}} = x_c + (x - x_c)(1 + k \tilde{r}_u^2)`$ (Brown, 1966).
- **Elastic Biomechanical Strain:** $`d(x,y) = \alpha \cdot \frac{G_{\sigma=6.0} \ast \Delta}{\lVert G_{\sigma=6.0} \ast \Delta \rVert_\infty}`$ with $`31 \times 31`$ Gaussian kernel (Simard et al., 2003).
- **Cataract Media Opacity & Forward Straylight:** $`I_{\text{cataract}}(x,y) = (1 - h) I(x,y) + h [0.65 L_{\text{veil}} + 0.35 (G_{\sigma=14} \ast I)(x,y)]`$ with $`L_{\text{veil,RGB}} = [210, 165, 125]`$ and $`h \in [0.0, 0.95]`$ (van den Berg, 1995; Vos & van den Berg, 1999).

---

## 4. Quantitative Experimental Results

### 4.1 Retinal Vessel Segmentation Benchmark

Evaluated on multi-dataset benchmarks and independent test splits:

| Evaluation Dataset / Split | Sample Size | Evaluation Mask | Accuracy | Sensitivity | Specificity | Dice Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Leakage Held-Out Validation Split** | 17 samples | Inside FOV | 0.9515 (+/- 0.0076) | 0.7240 (+/- 0.0407) | 0.9782 (+/- 0.0056) | **0.7554 (+/- 0.0466)** | **0.9562 (+/- 0.0145)** |
| **DRIVE Benchmark Split** | 20 images | Inside FOV | 0.9473 (+/- 0.0074) | 0.7503 (+/- 0.0360) | 0.9756 (+/- 0.0061) | **0.7809 (+/- 0.0178)** | **0.9710 (+/- 0.0060)** |
| **STARE Benchmark Split** | 20 images | Inside FOV | 0.9502 (+/- 0.0110) | 0.7500 (+/- 0.0373) | 0.9791 (+/- 0.0049) | **0.7915 (+/- 0.0340)** | **0.9654 (+/- 0.0100)** |
| **HRF High-Resolution Split** | 45 images | Inside FOV | 0.9541 (+/- 0.0057) | 0.7245 (+/- 0.0435) | 0.9770 (+/- 0.0064) | **0.7384 (+/- 0.0381)** | **0.9506 (+/- 0.0136)** |
| **Bézier Matched Filter (Kanjanasurat et al., 2020)** | 40 images (DRIVE) | Published [1] | 0.9440 | 0.7300 | 0.9650 | N/A (not in [1]) | N/A (not in [1]) |
| **Classical Bézier Matched Filter + Inner-FOV Otsu (Local Baseline)** | 20 images (DRIVE) | Inside FOV | 0.9122 (+/- 0.0150) | 0.5058 (+/- 0.0362) | 0.9714 (+/- 0.0153) | **0.5915 (+/- 0.0376)** | **0.7981 (+/- 0.0175)** |

*Note: Multi-dataset training split combined 80% of DRIVE/STARE/HRF samples; the 17-sample validation split is strictly held-out with zero identity leakage. DRIVE full-frame reference: ACC = 0.9638, SE = 0.7500, SP = 0.9840, Dice = 0.7808, ROC-AUC = 0.9807.*

### 4.2 Anatomical Landmark Localization Benchmark (IDRiD Dataset)

Evaluated across 103 high-resolution images from the Indian Diabetic Retinopathy Image Dataset (IDRiD, $4288 \times 2848$):

| Anatomical Landmark | Images Evaluated | Successful Detections | Mean Pixel Error | Median Pixel Error | Success Rate (Error $\le 5\%$ Width) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Optic Disc (OD Center)** | 103 | 102 | 68.18 px (1.59% width) | 56.99 px (1.33% width) | **99.03%** |
| **Fovea Centralis** | 103 | 97 | 108.11 px (2.52% width) | 94.88 px (2.21% width) | **94.17%** |

*(On HRF 15-image tuning split: OD Mean Error = 27.37 px, Median = 24.19 px, 100% success).*

### 4.3 Biometric Verification and Robustness Benchmark

#### Table 4.3A: 20-Subject Verification Benchmark (`matching_evaluation_report.md`)
Evaluated across 210 pairs (20 Genuine pairs under random $\pm 7^\circ$ tilt, $0.97-1.03\text{x}$ scale, and $0.90-1.10\text{x}$ brightness; 190 Cross-Subject Impostors):

| Metric | Measured Value | Standard Deviation / Range | Benchmark Significance |
| :--- | :---: | :---: | :--- |
| **Equal Error Rate (EER)** | **0.00%** | Optimal Operating Threshold = **0.210** | High biometric distinctiveness |
| **Verification Accuracy @ EER** | **100.00%** | 210 correct decisions out of 210 pairs | Perfect boundary separation |
| **Verification Accuracy @ 0.200** | **99.52%** | 209 correct decisions out of 210 pairs (FRR: **0.00%** [20/20], FAR: **0.53%** [189/190]) | Production operating threshold |
| **Genuine Mean Score** | **0.8903** | $\pm 0.0415$ (Range: 0.7992 - 0.9644) | High geometric consensus under perturbation |
| **Impostor Mean Score** | **0.0058** | $\pm 0.0261$ (Range: 0.0000 - 0.2049; 180/190 score 0.0000) | 21x suppression of background impostor pairs |

#### Table 4.3B: Production Pipeline 8-Preset Clinical Stress Sweep on Subject S01 & 4x4 Cohort Matrix
Evaluated using the live production `PipelineService` on DRIVE reference Subject 01 (`21_training.tif`) and 4x4 Cohort Matrix (`S01, S02, S23, S24`):

| Evaluation Scenario | Stress Vector Parameters | Verified Pairs | Match Score | FAR | FRR | Verification Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | In-plane $0^\circ$, scale $1.0\text{x}$ | 101 pairs | **100.0%** (1.0000) | 0.00% | 0.00% | Genuine Verified |
| **Pose Tilt & Zoom** | Tilt $+25^\circ$, scale $0.85\text{x}$ | 84 pairs | **80.8%** (0.8080) | 0.00% | 0.00% | Genuine Verified |
| **Cataract Media Haze** | Haze $h = 0.45$, gamma $0.85$ | 86 pairs | **86.4%** (0.8644) | 0.00% | 0.00% | Genuine Verified |
| **Corneal Surface Glare** | Glare intensity $0.65$, brightness $0.80$ | 84 pairs | **86.7%** (0.8667) | 0.00% | 0.00% | Genuine Verified |
| **Optical Defocus Blur** | Gaussian blur $\sigma = 2.5$ px | 70 pairs | **82.7%** (0.8266) | 0.00% | 0.00% | Genuine Verified |
| **Low-Light Sensor Noise** | ISO noise $\sigma = 16.0$, brightness $0.85$ | 62 pairs | **72.7%** (0.7271) | 0.00% | 0.00% | Genuine Verified |
| **Eyelid Occlusion** | Palpebral eyelid occlusion $22\%$ (65% superior / 35% inferior) | 81 pairs | **86.4%** (0.8641) | 0.00% | 0.00% | Genuine Verified |
| **Multi-Factor Judge Torture** | 7 stressors: Tilt $+15^\circ$, Scale $0.90\text{x}$, Haze $0.30$, Blur $\sigma=1.8$, Noise $\sigma=10$, Glare $0.35$, Elastic $\alpha=2.0$ | 51 pairs | **63.4%** (0.6343) | 0.00% | 0.00% | **Genuine Verified** (threshold: 20.0%) |
| **Cross-Subject Impostor Cohort** | 12 off-diagonal pairs across `S01, S02, S23, S24` | 0 pairs | **0.0%** (0.0000) | **0.00%** | N/A | **Impostors Rejected** |

---

## 5. Repository Structure and Code Layout

```
retinal-vascular/
├── api/                               # FastAPI 2.0 Enterprise Backend
│   ├── main.py                        # REST routing, CORS, Decompression Bomb Guards
│   ├── schemas.py                     # Pydantic v2 data contracts with strict range bounds
│   └── services.py                    # Pipeline orchestration and 11-param degradation engine
├── classical_pipeline/                # Anatomical Landmark Localization Subsystem
│   ├── __init__.py
│   ├── bezier_matched_filter.py       # Bézier curve matched filter baseline (Kanjanasurat 2020)
│   ├── fovea_localization.py          # Optic disc hemispheric sign (od_x < W/2), OD-anchored anisotropic Gaussian prior and macular darkness
│   ├── localization.py                # Legacy coordinate transform stubs (raises NotImplementedError)
│   ├── localization_api.py            # Unified landmark extraction interface
│   ├── localization_api_sanity_checks.py # Sanity verification suite for landmark localization
│   ├── optic_disc_localization.py     # Multi-cue density, structure-tensor tangent ray voting, and pale hue fusion
│   └── skeleton_metrics.py            # Topological skeleton quality analysis and short spur pruning
├── dl_pipeline/                       # Deep Learning Perception Subsystem
│   ├── __init__.py
│   ├── data_utils.py                  # Multi-dataset sample discovery and multi-format image/mask loaders
│   ├── dataset.py                     # U-Net dataset loader with Albumentations augmentations
│   ├── model.py                       # ResNet-34 U-Net network architecture
│   ├── postprocess.py                 # Morphological connected component noise filtering and binary hole filling
│   ├── predict.py                     # TTA-augmented batch/single inference
│   └── train.py                       # Tripartite loss trainer (BCE + Dice + clDice)
├── evaluation/                        # Empirical Benchmark and Metric Evaluation Suite
│   ├── __init__.py
│   ├── compare_models.py              # Comparative performance benchmarks across checkpoints
│   ├── evaluate_localization.py       # IDRiD landmark detection accuracy and error radii
│   ├── evaluate_matching.py           # Verification ROC/DET curves and EER metrics
│   ├── evaluate_vessel.py             # Vessel segmentation benchmark runner (DRIVE, STARE, HRF)
│   ├── metrics.py                     # Standardized evaluation metrics (Dice, Sensitivity, Specificity, Accuracy, ROC-AUC)
│   └── threshold_analysis.py          # Adaptive binarization threshold sweep engine
├── matching/                          # Biometric Matching Engine
│   ├── __init__.py
│   ├── bifurcation.py                 # Rutovitz Crossing Number minutiae extraction
│   ├── matcher.py                     # Kabsch SVD alignment and induced subgraph matching
│   ├── skeleton.py                    # Zhang-Suen and Lee medial axis thinning with spur pruning
│   └── triangulation.py               # Delaunay triangulation and invariant geometry
├── results/                           # Quantitative Benchmark Reports and Visual Artifacts
│   ├── fovea/                         # Fovea localization hyperparameters
│   │   └── best_config.json
│   ├── localization/                  # Landmark localization evaluation reports and CSVs
│   │   ├── idrid_localization_results.csv
│   │   └── localization_evaluation_report.md
│   ├── matching/                      # Biometric matching reports, ROC curves, and scores
│   │   ├── biometric_matching_curves.png
│   │   ├── matching_evaluation_report.md
│   │   └── matching_score_records.csv
│   ├── optic_disc/                    # Optic disc localization hyperparameters
│   │   └── best_config.json
│   ├── threshold_optimization_curves.png # Adaptive threshold parameter sweep visualization
│   ├── threshold_optimization_report.md   # Adaptive thresholding sweep results
│   ├── threshold_optimization_results.csv # Binarization parameter grid search records
│   ├── training_curves.png            # Initial multi-dataset U-Net training convergence curve
│   ├── training_curves_v2.png         # Production Soft-clDice U-Net v3 training convergence curve
│   └── vessel_evaluation_report.md        # Comprehensive multi-dataset segmentation benchmark
├── tests/                             # Automated Pytest Suite (40 unit & integration tests)
│   ├── __init__.py
│   ├── test_api.py                    # REST endpoint and schema validation tests
│   ├── test_bezier_matched_filter.py  # Bézier curve matched filter baseline tests
│   ├── test_codebase_integrity.py     # AST syntax, unused-import, zero-emoji, and comment-policy verification
│   ├── test_localization_api.py       # Optic disc and fovea localization tests
│   ├── test_matching.py               # Zhang-Suen, Lee, Rutovitz, and Delaunay matching tests
│   ├── test_postprocess.py            # Connected-component noise removal and binary hole filling tests
│   ├── test_week1_and_week2.py        # Dataset loading and metric validation tests
│   └── test_week3_and_week4.py        # U-Net forward pass, loss, and early stopping tests
├── web/                               # Next.js 16 WebGL 3D Studio Frontend
│   ├── .gitignore
│   ├── README.md
│   ├── eslint.config.mjs
│   ├── next.config.ts
│   ├── package-lock.json
│   ├── package.json                   # Dependencies (Next 16.4.2, React 19, Three.js, Lucide)
│   ├── tsconfig.json
│   └── src/
│       ├── app/
│       │   ├── favicon.ico
│       │   ├── globals.css            # Medical typography, grid layouts, and color tokens
│       │   ├── layout.tsx             # Root layout with Geist Mono font and metadata
│       │   └── page.tsx               # 3-Stage Studio (Globe, Matcher, Cohort Matrix)
│       ├── components/
│       │   ├── DelaunayConstellation.tsx  # 3D Three.js WebGL dual-plane constellation & laser viewer
│       │   └── RetinalGlobe.tsx       # 3D Three.js sphere with GLSL depth shader
│       ├── lib/
│       │   └── api.ts                 # Type-safe fetch client and preset definitions
│       └── types/
│           └── biometrics.ts          # TypeScript interfaces matching backend Pydantic models
├── .gitignore                         # Git exclusion rules for large models, data, and secrets
├── README.md                          # Comprehensive project documentation and benchmark report
└── requirements.txt                   # Complete Python dependencies (Core, DL, FastAPI, Pytest)
```

---

## 6. Installation, Configuration, and Usage

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- Node.js 18.17+ or Node.js 20+
- Recommended: CUDA-capable GPU (CPU fallback fully supported)

### 6.1 Backend Setup
1. Clone the repository and initialize the Python virtual environment:
   ```powershell
   git clone https://github.com/swnst/retinal-vascular.git
   cd retinal-vascular
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Verify model weight availability:
   The production checkpoint `best_unet_model_v3.pth` (93.39 MB) should reside in `models/`.
3. Run the automated test suite:
   ```powershell
   pytest tests/ -v
   ```
   *Expected outcome: 40 passed in ~17s.*
4. Start the FastAPI backend server:
   ```powershell
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### 6.2 Frontend Setup
1. Navigate to the web workspace and install dependencies:
   ```powershell
   cd web
   npm install
   ```
2. Run type-checking and linter:
   ```powershell
   npm run lint
   npx tsc --noEmit
   ```
3. Start the Next.js development server:
   ```powershell
   npm run dev
   ```
   Access the web studio at `http://localhost:3000`.

### 6.3 REST API Endpoints

- `GET /api/v1/health`: Heartbeat endpoint verifying backend service readiness and model availability.
- `GET /api/v1/presets`: Returns available clinical stress presets and pre-cached verification pairs across nominal and perturbed scenarios.
- `POST /api/v1/segment`: Ingests a base64 fundus image, executes ResNet-34 U-Net perception with TTA, computes adaptive binarization, and returns vessel probability maps, binary masks, and skeleton representations.
- `POST /api/v1/localize`: Detects Optic Disc and Fovea Centralis coordinates, disk-to-fovea distance, and processing time.
- `POST /api/v1/match`: Accepts reference and target fundus scans along with 11 perturbation parameters. Executes segmentation, landmark extraction, Kabsch SVD alignment, and Delaunay induced subgraph matching, returning decision metrics and laser correspondence coordinates.

---

## 7. Citation and Academic Attribution

If you utilize the VERA platform, codebase, or mathematical models in your research, cite as:

```bibtex
@misc{suwannasit2026vera,
  author       = {Wiwat Suwannasit and Dvis Vichitlaksana},
  title        = {VERA: Vascular Extraction and Retinal Authentication Platform},
  howpublished = {Senior Capstone Project, Department of IoT and Information Engineering, School of Engineering, King Mongkut's Institute of Technology Ladkrabang},
  year         = {2026},
  note         = {Supervised by Asst. Prof. Isoon Kanjanasurat and Asst. Prof. Dr. Nitjaree Satayarak}
}
```
