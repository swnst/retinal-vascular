# VERA: Vascular Extraction and Retinal Authentication Platform

### Personal Biometric Verification and 3D Ocular Anatomy System via Deep Learning Vessel Segmentation, Rutovitz Crossing Number Topology, and Kabsch SE(2) Induced Delaunay Triangulation

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
  - Kanjanasurat et al. (2020), *International Journal of Innovative Computing, Information and Control* (IJICIC), DOI: 10.24507/ijicic.16.03.879
  - Satayarak and Kanjanasurat (2018), *Proceedings of the 2018 International Conference on Biomedical Engineering and Technology* (ICBET), DOI: 10.1145/3239438.3239446

---

## 1. Executive Summary and Capstone Scope Analysis

VERA is an end-to-end biometric verification and ocular anatomical analysis system built on retinal vascular morphology. Unlike facial, voice, or epidermal fingerprint modalities, the vascular branching pattern of the human fundus oculi is internal, genetically unique (even between monozygotic twins), and topologically stable throughout adulthood.

### Initial Capstone I Target vs. Delivered System Reality

The original Capstone I project proposal specified a baseline proof-of-concept: evaluating standard U-Net segmentation on clean DRIVE fundus images, extracting basic bifurcation points, computing Delaunay triangle invariants, and rendering results in a basic standalone script.

The implemented platform exceeds the original targets across all core engineering dimensions:

| Evaluation Dimension | Initial Capstone I Target | Delivered System Reality | Status |
| :--- | :--- | :--- | :---: |
| **Vessel Segmentation** | Standard U-Net (~0.70 Dice on DRIVE) | Dual-Stream Self-Consistent U-Net (`best_unet_model_v3.pth`) with ResNet-34 encoder reaching **0.7848 Dice, 0.9652 AUC, 98.66% Specificity** | Exceeded Target |
| **Probability Binarization** | Static 0.50 cutoff | **Quantile-Calibrated Adaptive Thresholding** ( $\tau_{\text{eff}} = \max(0.32, \min(0.60, 0.65 \cdot p_{95})$ ) resilient to cataract opacity | Exceeded Target |
| **Landmark Localization** | Basic center heuristic on DRIVE | Multi-stage Hough and Vessel Convergence on 103 high-resolution IDRiD images (**99.03% OD Success, 94.17% Fovea Success**) | Exceeded Target |
| **Biometric Matcher** | Independent Delaunay Invariants (fails under vertex dropout) | **3-Stage Kabsch SE(2) SVD Constellation Consensus + Induced Common-Subgraph Delaunay Triangulation** | Exceeded Target |
| **Clinical Robustness** | None (clean static scans only) | **11-Parameter Physical Degradation Suite**; passes 7-parameter `Judge Torture` stress at **63.4%** matching score (threshold: 20.0%) | Exceeded Target |
| **Software Architecture** | Standalone Python scripts | **Decoupled Architecture**: FastAPI 2.0 Backend + Next.js 16 App Router Turbopack Frontend | Exceeded Target |
| **Visualization and UI** | Static 2D plots | **3-Stage Three.js WebGL Studio**: 3D Retinal Globe with depth GLSL shader, Dual 3D Matcher with laser correspondence beams, and 4x4 Cohort Matrix | Exceeded Target |
| **Automated Testing** | Manual inspection | **Automated Suite of 29 Pytest Tests**, ESLint (0 errors), Strict TypeScript (0 errors), Next.js Turbopack build | Exceeded Target |

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

1. **Input and Degradation Suite**: Ingests reference (Scan A) and target (Scan B) fundus photographs. An 11-parameter physical engine simulates real-world clinical stressors including head tilt ($\pm 45^\circ$), axial scale/zoom ($0.80\text{x} - 1.20\text{x}$), cataract media haze ($0.0 - 0.70$), corneal surface glare, sensor ISO noise, eyelid occlusion, spherical optical warp, and pulsatile intraocular pressure (IOP) elastic displacements.
2. **Deep Learning Perception Layer**: Executes inference via a 24.4M-parameter U-Net with ImageNet-pretrained ResNet-34 backbone. Includes Test-Time Augmentation (TTA: horizontal flip averaging) and CLAHE contrast equalization.
3. **Adaptive Thresholding and Postprocessing**: Dynamically calculates effective binarization cutoffs based on the 95th percentile intensity of the predicted foreground probability within the Field of View (FOV). Removes isolated noise components smaller than 25 pixels and applies morphological binary hole filling.
4. **Anatomical Landmark Localization**: Detects the Optic Disc (OD) via Green-channel intensity and circular Hough transformation. Fovea centralis coordinates are extracted through parabolic intensity minimum search guided by anatomical distance priors (2.0 to 2.5 OD diameters temporal to the disc).
5. **Centerline Thinning and Minutiae Extraction**: Generates 1-pixel unitary skeletons via Zhang-Suen morphological thinning. Minutiae points (bifurcations and crossovers) are classified using the Rutovitz Crossing Number algorithm.
6. **Kabsch SE(2) SVD Alignment and Induced Subgraph Delaunay Matching**: Resolves global spatial rotation and translation between scans using Singular Value Decomposition on candidate minutiae pairs. Common inlier vertices form an induced shared subgraph, eliminating false rejections caused by independent Delaunay re-triangulation under point dropout.

---

## 3. Mathematical Foundations and Literature Citations

Every algorithmic component in the VERA platform is derived from verified mathematical literature:

### 3.1 CLAHE Illumination Equalization
To normalize intra-retinal illumination gradients across variable fundus pigmented backgrounds:

$$h_{\text{clipped}}(n) = \min(h(n), \beta)$$
$$g(x,y) = \left[ g_{\max} - g_{\min} \right] \cdot P(f(x,y)) + g_{\min}$$

*Citation:* Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. *Graphics Gems IV*, Academic Press, pp. 474-485. DOI: 10.1016/B978-0-12-336156-1.50061-6.

### 3.2 High-Frequency Noise Estimation (MAD)
Estimates background sensor noise standard deviation $\hat{\sigma}$ with a 50% statistical breakdown point:

$$\hat{\sigma} = \frac{\text{median}\left( \left| y_i - \text{median}(y) \right| \right)}{0.6745}$$

*Citation:* Donoho, D. L., and Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455. DOI: 10.1093/biomet/81.3.425.

### 3.3 Composite Training Loss (BCE + Soft Sorensen-Dice + Soft-clDice)
Overcomes extreme class imbalance (~91% background vs. ~9% vascular foreground) while enforcing topological continuity in thin 1-pixel terminal capillaries:

$$\mathcal{L}_{\text{total}} = 0.35 \cdot \mathcal{L}_{\text{BCE}} + 0.40 \cdot \mathcal{L}_{\text{Dice}} + 0.25 \cdot \mathcal{L}_{\text{clDice}}$$
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$
$$\mathcal{L}_{\text{clDice}} = 1 - \frac{2 \cdot T_{\text{prec}}(S(P), Y) \cdot T_{\text{sens}}(S(Y), P)}{T_{\text{prec}}(S(P), Y) + T_{\text{sens}}(S(Y), P) + \epsilon}$$

where $S(X) = \text{ReLU}(X - \text{open}(X))$ performs differentiable soft skeletonization via min-max morphological pooling.

*Citations:*
- Milletari, F., Navab, N., and Ahmadi, S. A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. *3DV 2016*, pp. 565-571. DOI: 10.1109/3DV.2016.79.
- Shit, S. et al. (2021). clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. *CVPR 2021*, pp. 16560-16569. DOI: 10.1109/CVPR46437.2021.01629.

### 3.4 Morphological Skeletonization (Zhang-Suen and Lee Medial Axis)
Extracts unitary-width centerlines via parallel boundary pixel deletion based on neighbor count $B(P_1)$ and $0 \to 1$ transition count $A(P_1)$:

$$2 \le B(P_1) \le 6, \quad A(P_1) = 1$$

*Citations:*
- Zhang, T. Y., and Suen, C. Y. (1984). A fast parallel algorithm for thinning digital patterns. *Communications of the ACM*, 27(3), 236-239. DOI: 10.1145/357994.358023.
- Lee, T. C., Kashyap, R. L., and Chu, C. N. (1994). Building skeleton models via 3-D medial surface/axis thinning algorithms. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 16(4), 351-360. DOI: 10.1109/34.277590.

### 3.5 Minutiae Extraction (Rutovitz Crossing Number)
Classifies skeleton pixels into terminal endpoints ($X=1$), vessel continuations ($X=2$), bifurcations ($X=3$), and crossovers ($X=4$):

$$X(P) = \frac{1}{2} \sum_{i=1}^8 \left| P_i - P_{i+1} \right|, \quad P_9 = P_1$$

*Citation:* Rutovitz, D. (1966). Pattern recognition. *Journal of the Royal Statistical Society: Series A*, 129(4), 504-530. DOI: 10.2307/2982247.

### 3.6 Closed-Form Spatial Alignment (Kabsch SVD Procrustes)
Derives the optimal rigid rotation matrix $R \in \text{SO}(2)$ and translation vector $t \in \mathbb{R}^2$ minimizing root-mean-square alignment deviation:

$$H = \sum_{i=1}^k (p_i - \bar{p})(q_i - \bar{q})^T = U \Sigma V^T$$
$$R = V \begin{pmatrix} 1 & 0 \\ 0 & \det(V U^T) \end{pmatrix} U^T, \quad t = \bar{q} - R \bar{p}$$

*Citation:* Kabsch, W. (1976). A solution for the best rotation to relate two sets of vectors. *Acta Crystallographica Section A*, 32(5), 922-923. DOI: 10.1107/S0567739476001873.

### 3.7 Delaunay Invariant Geometry and Induced Common-Subgraph Matching
Triangulates minutiae graphs and computes interior angle and side ratio invariant descriptors:

$$\boldsymbol{\theta} = (\theta_{\min}, \theta_{\text{mid}}, \theta_{\max}), \quad \boldsymbol{\rho} = \left(\frac{a}{b}, \frac{b}{c}, \frac{a}{c}\right)$$
$$C(i, j) = 0.55 \cdot \frac{\Delta\bar{\theta}}{\tau_\theta} + 0.25 \cdot \min_{a, b} \frac{|r_{A,a} - r_{B,b}|}{\tau_r} + 0.20 \cdot \frac{1 - \rho_{\text{area}}}{0.40}$$

To prevent catastrophic false rejections from vertex dropout where independent Delaunay graphs re-triangulate into conflicting topologies, VERA executes **Induced Common-Subgraph Delaunay Triangulation**:
1. Spatial Minutiae Consensus: Bifurcation points from Scan A and Scan B are aligned globally into a unified frame using Kabsch SE(2) SVD consensus.
2. Mutual Inlier Gating: Global bipartite assignment identifies the mutual inlier point subset $\mathcal{P}\_{\text{mutual}} = \mathcal{P}\_A \cap \mathcal{P}\_B$.
3. Induced Graph Triangulation: Delaunay triangulation is constructed over the shared subset $\mathcal{P}_{\text{mutual}}$, guaranteeing identical topological connectivity between both scans.

*Citation:* Kanjanasurat, I., Purahong, B., Aoyama, H., Benjangkaprasert, C., and Pintavirooj, C. (2020). Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern. *IJICIC*, 16(3), 879-894. DOI: 10.24507/ijicic.16.03.879.

### 3.8 Bipartite Assignment (Kuhn-Munkres)
Finds optimal bijective correspondence between transformed candidate minutiae and target minutiae:

$$\min \sum_{i=1}^m \sum_{j=1}^n C_{i,j} X_{i,j}, \quad \text{where } C_{i,j} = \lVert R p_i + t - q_j \rVert_2$$

*Citation:* Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2), 83-97. DOI: 10.1002/nav.3800020109.

### 3.9 Spatial Keypoint Indexing (k-d Tree)
Reduces geometric proximity queries from $O(N \cdot M)$ to $O(N \log M)$:

*Citation:* Bentley, J. L. (1975). Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), 509-517. DOI: 10.1145/361002.361007.

### 3.10 Optical and Physical Degradation Mechanics
- **Scale-Invariant Radial Optical Lens Distortion:** $\tilde{r}\_{u} = \frac{r\_{u}}{r\_{\text{max}}}$, $x\_{\text{dist}} = x\_c + (x - x\_c)(1 + k \tilde{r}\_u^2)$ (Brown, 1966).
- **Elastic Biomechanical Strain:** $d(x,y) = \alpha \cdot \frac{G_\sigma * \Delta}{\lVert G_\sigma * \Delta \rVert_\infty}$ (Simard et al., 2003).
- **Cataract Media Opacity & Forward Straylight:** $I\_{\text{cataract}}(x,y) = (1 - h) I(x,y) + h [0.65 L\_{\text{veil}} + 0.35 (G\_{\sigma=14} * I)(x,y)]$ with $L\_{\text{veil}} = [210, 165, 125]$ (van den Berg, 1995; Vos & van den Berg, 1999).

For complete derivations, variable mappings, and extended theoretical proofs, refer to the project technical documentation.

---

## 4. Quantitative Experimental Results

### 4.1 Retinal Vessel Segmentation Benchmark (DRIVE Test Set)

Evaluated on the independent DRIVE test benchmark (20 high-resolution fundus images, manual expert consensus ground truth):

| Model / Architecture | Backbone | Input Channels | TTA | Dice Score | ROC-AUC | Sensitivity | Specificity | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline U-Net (Ronneberger 2015) | Plain Conv | 3 (RGB) | No | 0.7241 | 0.9412 | 0.7105 | 0.9782 | 0.9420 |
| Attention U-Net | ResNet-18 | 3 (RGB) | No | 0.7518 | 0.9540 | 0.7420 | 0.9810 | 0.9498 |
| VERA U-Net v1 | ResNet-34 | 3 (RGB) | No | 0.7682 | 0.9589 | 0.7584 | 0.9835 | 0.9521 |
| VERA U-Net v2 | ResNet-34 | 3 (RGB) | Yes | 0.7765 | 0.9618 | 0.7650 | 0.9850 | 0.9542 |
| **VERA Production Model v3** | **ResNet-34** | **3 (RGB + CLAHE)** | **Yes** | **0.7848** | **0.9652** | **0.7712** | **0.9866** | **0.9568** |

### 4.2 Anatomical Landmark Localization Benchmark (IDRiD Dataset)

Evaluated across 103 high-resolution images from the Indian Diabetic Retinopathy Image Dataset (IDRiD):

| Anatomical Landmark | Target Count | Successful Detections | Mean Pixel Error | Radial Accuracy ($\le 1.0 \text{ OD diameter}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Optic Disc (OD Center)** | 103 | 102 | 14.8 px | **99.03%** |
| **Fovea Centralis** | 103 | 97 | 26.3 px | **94.17%** |

### 4.3 Biometric Verification and Robustness Benchmark

Evaluated using genuine same-subject pairs, impostor cross-subject pairs, and calibrated clinical perturbation vectors:

| Evaluation Scenario | Stress Conditions | Sample Size | Mean Match Score | False Accept Rate (FAR) | False Reject Rate (FRR) | Verification Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Nominal Same-Person** | Clean reference vs. identical scan | 40 pairs | 100.0% | 0.00% | 0.00% | Genuine Verified |
| **Moderate Head Tilt** | $\Delta \theta = \pm 15^\circ$ | 40 pairs | 92.4% | 0.00% | 0.00% | Genuine Verified |
| **Axial Zoom / Scale** | Scale factor $0.85\text{x} - 1.15\text{x}$ | 40 pairs | 88.7% | 0.00% | 0.00% | Genuine Verified |
| **Cataract Media Haze** | Haze coefficient $h = 0.45$ | 40 pairs | 79.2% | 0.00% | 0.00% | Genuine Verified |
| **Pulsatile IOP Elastic Strain** | Elastic amplitude $\alpha = 10.0$ | 40 pairs | 74.6% | 0.00% | 0.00% | Genuine Verified |
| **Full Judge Torture Suite** | 7 concurrent stressors (Tilt, Scale, Haze, Glare, ISO, Occlusion, Elastic) | 40 pairs | **63.4%** | 0.00% | 0.00% | **Genuine Verified** (threshold: 20.0%) |
| **Cross-Subject Impostor** | Unrelated individuals (Subject 01 vs. Subject 02) | 120 pairs | **0.0% - 1.8%** | **0.00%** | N/A | **Impostor Rejected** |

---

## 5. Repository Structure and Code Layout

```
retinal-vascular/
├── api/                               # FastAPI 2.0 Enterprise Backend
│   ├── main.py                        # REST routing, CORS, Decompression Bomb Guards
│   ├── schemas.py                     # Pydantic v2 data contracts with strict range bounds
│   └── services.py                    # Pipeline orchestration and 11-param degradation engine
├── classical_pipeline/                # Anatomical Landmark Localization Subsystem
│   ├── fovea_localization.py          # FAZ parabolic search and directional vessel search
│   ├── localization.py                # Coordinate transforms and anatomical landmark heuristics
│   ├── localization_api.py            # Unified landmark extraction interface
│   ├── optic_disc_localization.py     # Circular Hough and Green-channel luminance detection
│   └── skeleton_metrics.py            # Geometric vessel caliber and density metrics
├── dl_pipeline/                       # Deep Learning Perception Subsystem
│   ├── data_utils.py                  # Multi-dataset sample discovery (DRIVE, STARE, HRF, CHASE)
│   ├── dataset.py                     # Dual-stream dataset loader with Albumentations
│   ├── model.py                       # ResNet-34 U-Net network architecture
│   ├── postprocess.py                 # Quantile adaptive thresholding and morphological cleanup
│   ├── predict.py                     # TTA-augmented batch/single inference
│   └── train.py                       # Tripartite loss trainer (BCE + Dice + clDice)
├── evaluation/                        # Empirical Benchmark and Metric Evaluation Suite
│   ├── compare_models.py              # Comparative performance benchmarks across checkpoints
│   ├── evaluate_localization.py       # IDRiD landmark detection accuracy and error radii
│   ├── evaluate_matching.py           # Verification ROC/DET curves and EER metrics
│   ├── evaluate_vessel.py             # Vessel segmentation benchmark runner (DRIVE, STARE, HRF)
│   ├── metrics.py                     # Standardized metrics (Dice, IoU, AUC, F1, Specificity)
│   └── threshold_analysis.py          # Adaptive binarization threshold sweep engine
├── matching/                          # Biometric Matching Engine
│   ├── bifurcation.py                 # Rutovitz Crossing Number minutiae extraction
│   ├── matcher.py                     # Kabsch SVD alignment and induced subgraph matching
│   ├── skeleton.py                    # Zhang-Suen and Lee medial axis thinning with spur pruning
│   └── triangulation.py               # Delaunay triangulation and invariant geometry
├── models/                            # Local Model Weights (Excluded from Git via .gitignore)
│   └── best_unet_model_v3.pth         # Active production checkpoint (93.39 MB, ResNet-34)
├── notebooks/                         # Google Colab & Cloud GPU Training Workflows
│   └── train_colab_v2.ipynb           # Cloud training notebook for ResNet-34 U-Net
├── results/                           # Quantitative Benchmark Reports and Visual Artifacts
│   ├── localization/                  # Landmark localization evaluation reports and CSVs
│   │   ├── idrid_localization_results.csv
│   │   └── localization_evaluation_report.md
│   ├── matching/                      # Biometric matching reports, ROC curves, and scores
│   │   ├── biometric_matching_curves.png
│   │   ├── matching_evaluation_report.md
│   │   └── matching_score_records.csv
│   ├── threshold_optimization_report.md   # Adaptive thresholding sweep results
│   ├── threshold_optimization_results.csv # Binarization parameter grid search records
│   └── vessel_evaluation_report.md        # Comprehensive multi-dataset segmentation benchmark
├── tests/                             # Automated Pytest Suite (29 unit & integration tests)
│   ├── test_api.py                    # REST endpoint and schema validation tests
│   ├── test_localization_api.py       # Optic disc and fovea localization tests
│   ├── test_matching.py               # Zhang-Suen, Lee, Rutovitz, and Delaunay matching tests
│   ├── test_postprocess.py            # Morphological cleanup and adaptive binarization tests
│   ├── test_week1_and_week2.py        # Dataset loading and metric validation tests
│   └── test_week3_and_week4.py        # U-Net forward pass, loss, and early stopping tests
├── web/                               # Next.js 16 WebGL 3D Studio Frontend
│   ├── package.json                   # Dependencies (Next 16.4.2, React 19, Three.js, Lucide)
│   ├── src/
│   │   ├── app/
│   │   │   ├── globals.css            # Medical typography, grid layouts, and color tokens
│   │   │   ├── layout.tsx             # Root layout with Geist Mono font and metadata
│   │   │   └── page.tsx               # 3-Stage Studio (Globe, Matcher, Cohort Matrix)
│   │   ├── components/
│   │   │   ├── DelaunayConstellation.tsx  # 2D Canvas keypoint and triangulation overlay
│   │   │   └── RetinalGlobe.tsx       # 3D Three.js sphere with GLSL depth shader
│   │   ├── lib/
│   │   │   └── api.ts                 # Type-safe fetch client and preset definitions
│   │   └── types/
│   │       └── biometrics.ts          # TypeScript interfaces matching backend Pydantic models
├── .gitignore                         # Git exclusion rules for large models, data, and secrets
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
   *Expected outcome: 29 passed in ~16s.*
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

- `POST /api/segment`: Ingests a base64 fundus image, executes ResNet-34 U-Net perception with TTA, computes adaptive binarization, and returns vessel probability maps, binary masks, and skeleton representations.
- `POST /api/localize`: Detects Optic Disc and Fovea Centralis coordinates, radius, and anatomical confidence scores.
- `POST /api/match`: Accepts reference and target fundus scans along with 11 perturbation parameters. Executes segmentation, landmark extraction, Kabsch SVD alignment, and Delaunay induced subgraph matching, returning decision metrics and laser correspondence coordinates.
- `GET /api/presets`: Returns pre-cached genuine and impostor clinical test pairs across nominal and perturbed scenarios.

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
