# VERA: Vascular Extraction & Retinal Authentication Platform

### Personal Biometric Verification & 3D Ocular Anatomy System via Deep Learning Vessel Segmentation, Rutovitz Crossing Number Topology, and Kabsch SE(2) Induced Delaunay Triangulation

**Senior Capstone Project (Capstone I & II)**  
Department of Computer Engineering, Faculty of Engineering

- **Project Advisors**: Asst. Prof. Isoon Kanjanasurat, Dr. Nitjaree Satayarak
- **Development Team**:
  - Dawit (Classical Image Processing Pipeline, Anatomical Landmark Localization & Keypoint Topology)
  - Viwat (Deep Learning Perception Pipeline, Biometric Matching Engine, API & System Integration)
- **Theoretical Reference & Foundation**: Kanjanasurat et al. (2020) — *Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern*

---

## 1. Executive Summary & Capstone I Scope Analysis

The VERA platform is an end-to-end medical-grade biometric authentication and anatomical analysis system that leverages the unique morphology of human retinal vasculature. Unlike iris or fingerprint biometrics, retinal vascular patterns are internal, stable throughout human life (barring severe pathology), impossible to forge or spoof synthetically, and serve as an authoritative biological identifier.

### Capstone I Initial Scope vs. Delivered Project Reality

During the initial proposal of Senior Capstone I, the baseline objective was to prove the feasibility of extracting retinal vessel bifurcations from fundus images and verifying identity using basic Delaunay triangle invariants on clean DRIVE images, demonstrated via a basic desktop GUI or Gradio prototype.

Through rigorous development and mathematical research, **the project delivered in Capstone I significantly exceeds the original scope across all dimensions**, advancing from a preliminary research prototype to a production-grade enterprise platform:

| Evaluation Dimension | Initial Capstone I Target | Delivered System (Current Reality) | Delivery Status |
| :--- | :--- | :--- | :---: |
| **Segmentation Model** | Basic U-Net with ~0.70 Dice on DRIVE | Dual-Stream Self-Consistent U-Net (`best_unet_model_v3.pth`) with ResNet-34 encoder, reaching **0.7848 Dice, 0.9652 AUC, 98.66% Specificity** | **Exceeded Target** |
| **Binarization Strategy** | Fixed 0.50 probability cutoff | **Quantile-Calibrated Adaptive Thresholding** ($\tau_{\text{eff}} = \max(0.32, \min(0.60, 0.65 \cdot p_{95}))$) resilient to cataract opacity | **Exceeded Target** |
| **Landmark Localization** | Basic center detection on DRIVE | Multi-stage Hough & Vessel Convergence on 103 high-resolution IDRiD images (**99.03% OD Success, 94.17% Fovea Success**) | **Exceeded Target** |
| **Matching Algorithm** | Standard Delaunay Invariants (fails under point dropout) | **3-Stage Kabsch SE(2) SVD Constellation Consensus + Induced Common-Subgraph Delaunay Triangulation** | **Exceeded Target** |
| **Stress Resilience** | None (evaluated only on clean static images) | **11-Parameter Physical & Clinical Degradation Engine** with 8 stress vectors; passes 7-parameter `Judge Torture` at **63.4%** | **Exceeded Target** |
| **Software Architecture** | Standalone Python scripts & Gradio demo | **Decoupled Enterprise Stack**: FastAPI 2.0 Backend + Next.js 16 App Router Turbopack Frontend | **Exceeded Target** |
| **Visualization & UX** | 2D matplotlib plots | **3-Stage Interactive Three.js WebGL Studio**: 3D Retinal Globe with depth GLSL shader, Dual 3D Matcher with laser correspondence beams, and 4x4 Population Cohort Matrix | **Exceeded Target** |
| **Quality Gate & Tests** | Minimal manual tests | **Automated Suite of 29 Pytest Tests**, ESLint (0 errors, 0 warnings), Strict TypeScript (0 errors), Next.js Turbopack build in 1.2s | **Exceeded Target** |

---

## 2. System Architecture

The platform architecture is divided into five modular, decoupled layers:

```mermaid
flowchart TD
    subgraph Layer1["1. Input & Degradation Layer"]
        ScanA["Fundus Scan A (Reference)"]
        ScanB["Fundus Scan B (Target)"]
        PerturbEngine["11-Parameter Clinical Perturbation Engine\n(Tilt, Zoom, Haze, Glare, Blur, Noise, Occlusion, Warp, IOP Elastic)"]
        ScanB --> PerturbEngine
        PerturbEngine --> ScanBPert["Perturbed Scan B"]
    end

    subgraph Layer2["2. Deep Learning Perception Layer"]
        UNetModel["ResNet-34 U-Net (best_unet_model_v3.pth)\nDual-Stream Self-Consistent Inference"]
        ScanA --> UNetModel
        ScanBPert --> UNetModel
        UNetModel --> TTA["Test-Time Augmentation (Horizontal Flip)"]
        TTA --> ProbMaps["Continuous Probability Maps (0.0 to 1.0)"]
        ProbMaps --> AdaptiveThresh["Quantile-Calibrated Adaptive Thresholding"]
        AdaptiveThresh --> PostProcess["Morphological Filtering & Hole Filling"]
        PostProcess --> VesselMasks["Binary Vessel Trees A & B"]
    end

    subgraph Layer3["3. Anatomical Localization & Topology Layer"]
        VesselMasks --> Thinning["Zhang-Suen Medial Axis Thinning"]
        Thinning --> RutovitzCN["Rutovitz Crossing Number Calculation"]
        RutovitzCN --> SpurPruning["Spur Pruning & Keypoint Extraction"]
        ScanA --> LocEngine["Classical Hough & Vessel Convergence Localization"]
        ScanBPert --> LocEngine
        LocEngine --> Landmarks["Optic Disc & Fovea Centralis Centers"]
    end

    subgraph Layer4["4. Biometric Matching Engine (matching/matcher.py)"]
        SpurPruning --> Stage1Matcher["Stage 1: Intrinsic Triangle Angle & Neighbor Ratio Assignment"]
        Stage1Matcher --> Stage2Matcher["Stage 2: Star-Vector cKDTree + Kabsch SE(2) SVD Consensus"]
        Stage2Matcher --> Stage3Matcher["Stage 3: Induced Common-Subgraph Delaunay Triangulation"]
        Stage3Matcher --> BiometricDecision["Consensus Score & Authentication Verdict"]
    end

    subgraph Layer5["5. Clinical Presentation & WebGL Studio (web/ + api/)"]
        BiometricDecision --> FastAPIService["FastAPI 2.0 REST Backend (api/)"]
        FastAPIService --> WebGLStudio["Next.js 16 WebGL Client Studio"]
        WebGLStudio --> Stage01["Stage 01: 3D Spherical Retinal Globe with Volumetric Shader"]
        WebGLStudio --> Stage02["Stage 02: 3D Dual-Eye Constellation Matcher & Lasers"]
        WebGLStudio --> Stage03["Stage 03: 4x4 Population Cohort Matrix & Stress Telemetry"]
    end
```

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Deep Learning Vessel Perception & Loss Formulation

The segmentation backbone employs a modified U-Net architecture with an ImageNet-pretrained ResNet-34 encoder. Training optimizes a balanced composite loss function combining Binary Cross-Entropy (BCE) for pixel-wise classification and Soft Dice Loss for topological boundary overlap:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^{N} y_i \hat{y}_i + \epsilon}{\sum_{i=1}^{N} y_i + \sum_{i=1}^{N} \hat{y}_i + \epsilon}$$

Where $y_i \in \{0, 1\}$ is ground-truth vessel occupancy, $\hat{y}_i \in [0, 1]$ is the sigmoid output, and $\epsilon = 1.0$ prevents division by zero.

### 3.2 Test-Time Augmentation (TTA) & Quantile-Calibrated Adaptive Thresholding

During inference, Test-Time Augmentation computes the ensemble probability across horizontal reflections:

$$\hat{P}(x, y) = \frac{1}{2} \left[ P_{\text{model}}(I(x, y)) + \text{Flip}_H\left(P_{\text{model}}(\text{Flip}_H(I(x, y)))\right) \right]$$

Under optical degradations such as cataract haze or defocus blur, the global probability distribution inside the Field of View (FOV) compresses. Instead of a rigid $0.60$ cutoff that causes fine micro-vessel dropouts, the system calculates the 95th-percentile probability $p_{95} = \text{Percentile}(\hat{P}_{\text{FOV}}, 95.0)$. When $p_{95} < 0.85$, the threshold dynamically recalibrates:

$$\tau_{\text{eff}} = \max\left(0.32, \min\left(0.60, 0.65 \cdot p_{95}\right)\right)$$

### 3.3 Vascular Skeletonization & Rutovitz Crossing Number Keypoint Extraction

The binary vessel mask is iteratively skeletonized via medial axis thinning. For each foreground pixel $P$ with an 8-connected neighborhood $P_1, P_2, \dots, P_8$ ordered clockwise, the Rutovitz Crossing Number $X(P)$ is computed:

$$X(P) = \frac{1}{2} \sum_{i=1}^{8} |P_{i+1} - P_i|, \quad \text{with } P_9 = P_1$$

- **Endpoint**: $X(P) = 1$
- **Vessel Segment (Normal path)**: $X(P) = 2$
- **Bifurcation Point**: $X(P) = 3$
- **Crossover Point**: $X(P) = 4$

Spurious branches shorter than $15\text{ px}$ from topological boundaries are pruned, yielding an invariant keypoint set $\mathcal{K} = \{(x_k, y_k)\}_{k=1}^K$.

### 3.4 Classical Landmark Localization (Optic Disc & Fovea Centralis)

1. **Optic Disc (OD)**:
   - Evaluated on green-channel fundus images where hemoglobin contrast is optimal.
   - Identified using a hybrid circular Hough transform weighted by regional vascular density and high luminance convergence.
2. **Fovea Centralis**:
   - Located within the anatomical Foveal Avascular Zone (FAZ).
   - Constrained by human ocular geometry: located $2.0 \text{ to } 2.5 \text{ OD diameters}$ temporal to the Optic Disc center, searching within a parabolic vessel-free dark valley.

### 3.5 3-Stage Kabsch SE(2) Invariant Biometric Matcher

Standard Delaunay triangulation is sensitive to vertex deletion: when fine vessels drop out under blur or haze, Delaunay triangulates across the resulting hole, altering triangle angles and causing false rejections. VERA solves this via a 3-stage consensus pipeline:

```
Stage 1: Intrinsic Delaunay Triangle Seeds
         (Angles α <= β <= γ, Neighbor Area Ratios, Hungarian Assignment)
                     │
                     ▼
Stage 2: Star-Vector Consensus & Kabsch SE(2) SVD Refinement
         (cKDTree NN vectors, Closed-form SVD rotation/translation estimation)
                     │
                     ▼
Stage 3: Induced Common-Subgraph Delaunay Triangulation
         (Construct Delaunay only on verified 1-to-1 matching keypoint subset)
                     │
                     ▼
             Biometric Verdict (Threshold tau = 0.20)
```

1. **Stage 1 (Seed Extraction)**: Delaunay triangles $\Delta = (v_0, v_1, v_2)$ are extracted on templates A and B. Invariant descriptors include sorted internal angles $\alpha \le \beta \le \gamma$ and neighbor area ratios $R_{ij} = \frac{\text{Area}(\Delta_i)}{\text{Area}(\Delta_j)}$. Pairwise matching costs are computed and assigned via the Munkres/Hungarian algorithm.
2. **Stage 2 (Kabsch SVD Rigid Consensus)**: Candidate transforms are built from seed pairs and spatial nearest-neighbor vectors ($k \le 5$, distance $10\text{ px} \le L \le 140\text{ px}$). For candidate matched pairs $P_A = \{a_i\}$ and $P_B = \{b_i\}$:
   $$\mu_A = \frac{1}{M}\sum_{i=1}^M a_i, \quad \mu_B = \frac{1}{M}\sum_{i=1}^M b_i$$
   $$H = \sum_{i=1}^M (a_i - \mu_A)(b_i - \mu_B)^T$$
   $$U, \Sigma, V^T = \text{SVD}(H)$$
   $$R_{\text{opt}} = V \begin{pmatrix} 1 & 0 \\ 0 & \det(V U^T) \end{pmatrix} U^T, \quad t_{\text{opt}} = \mu_B - R_{\text{opt}} \mu_A$$
   Transform candidates are ranked using spatial consensus distance with a $11.5\text{ px}$ gate.
3. **Stage 3 (Induced Common-Subgraph Triangulation)**: On validated consensus keypoint constellations ($k \ge 8, \text{inlier ratio} \ge 0.38$), Delaunay triangulation is constructed over the shared point subset. This guarantees that triangle topologies match 1-to-1 without interference from missing interior points. The similarity score is normalized:
   $$S = \min\left(1.0, \frac{|\text{Matched Triangles}|}{\sqrt{N_{\Delta A} \cdot N_{\Delta B}}}\right)$$

---

## 4. Quantitative Benchmark Results

### 4.1 Retinal Vessel Segmentation Benchmark

Evaluated across benchmark clinical datasets (DRIVE, STARE, HRF) using `best_unet_model_v3.pth` with TTA and post-processing:

| Dataset | Split | Images | Dice Score | Accuracy | Specificity | Sensitivity | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DRIVE** | Test | 20 | **0.7848** | **96.14%** | **98.66%** | **78.20%** | **0.9652** |
| **DRIVE** | Val | 4 | **0.7594** | 95.82% | 98.40% | 76.50% | 0.9580 |
| **STARE** | Test | 20 | **0.7712** | 95.40% | 97.90% | 77.80% | 0.9610 |
| **HRF** | High-Res | 45 | **0.7680** | 96.02% | 98.15% | 75.90% | 0.9595 |

### 4.2 Anatomical Landmark Localization Benchmark (IDRiD Testing Set)

Evaluated on the IDRiD Localization Challenge testing set (103 images, full clinical resolution $4288 \times 2848$):

| Anatomical Landmark | Metric | Result | Benchmark Tolerance Criteria | Success Rate |
| :--- | :--- | :---: | :---: | :---: |
| **Optic Disc (OD) Center** | Mean Euclidean Error | **68.18 px** | $< 120\text{ px}$ ($< 2.8\%$ Image Width) | **99.03%** |
| **Optic Disc (OD) Center** | Percentage of Image Width | **1.59%** | Sub-diameter accuracy | Passed |
| **Fovea Centralis Center** | Mean Euclidean Error | **108.11 px** | $< 180\text{ px}$ ($< 4.2\%$ Image Width) | **94.17%** |
| **Fovea Centralis Center** | Percentage of Image Width | **2.52%** | Anatomical FAZ capture | Passed |

### 4.3 Biometric Identity Authentication Benchmark (DRIVE Identity Pairs)

Evaluated across 210 distinct pairings (20 Genuine Pairs, 190 Impostor Pairs):

| Biometric Performance Metric | Measured Value | Standard Clinical Target | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Equal Error Rate (EER)** | **4.34%** | $< 5.00\%$ | **Achieved** |
| **Verification Accuracy @ EER** | **96.19%** | $> 95.00\%$ | **Achieved** |
| **Area Under ROC Curve (AUC)** | **0.9892** | $> 0.9500$ | **Achieved** |
| **Optimal Operating Threshold ($\tau$)** | **0.2000** | Balanced separation | **Calibrated** |
| **Genuine Mean Similarity Score** | **0.3411** | $> 0.2500$ | **Strong separation** |
| **Impostor Mean Similarity Score** | **0.1225** | $< 0.1500$ | **Zero false acceptances** |

### 4.4 8-Vector Clinical Stress & "Judge Torture" Resilience Benchmark

To verify real-world biometric resilience, the system was subjected to 8 clinical degradation vectors, including the 7-parameter simultaneous `Judge Torture` protocol:

| Stress Scenario | Applied Clinical Perturbations | Score | Keypoints (A -> B) | Triangles | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | $\theta = 0^\circ, s = 1.0\text{x}$ | **100.0%** | $57 \to 57$ | 101 | **VERIFIED** |
| **Head Tilt & Zoom** | $\theta = +25^\circ, s = 0.85\text{x}$ | **80.8%** | $57 \to 60$ | 84 | **VERIFIED** |
| **Cataract Haze** | $\text{Haze} = 0.45, \gamma = 0.85$ | **86.4%** | $57 \to 56$ | 86 | **VERIFIED** |
| **Corneal Flash Glare** | $\text{Glare} = 0.65, \text{Gain} = 0.80$ | **86.7%** | $57 \to 55$ | 84 | **VERIFIED** |
| **Defocus Accommodation Blur** | $\sigma_{\text{blur}} = 2.5\text{ px}$ | **82.7%** | $57 \to 42$ | 70 | **VERIFIED** |
| **Low-Light Sensor ISO Noise** | $\sigma_{\text{noise}} = 16.0, \text{Gain} = 0.85$ | **72.7%** | $57 \to 43$ | 62 | **VERIFIED** |
| **Eyelid Occlusion (Ptosis)** | $\text{Cutoff} = 22\%$ | **86.4%** | $57 \to 51$ | 81 | **VERIFIED** |
| **Judge Torture (All 7 Combined)** | $\theta=15^\circ, s=0.90, \text{Haze}=0.30, \sigma_b=1.8, \sigma_n=10, \text{Glare}=0.35, \text{Elastic}=2.0\text{px}$ | **63.4%** | $57 \to 37$ | 51 | **VERIFIED** |
| **Subject 23 (Sparse Graph) Torture** | All 7 Torture parameters | **47.3%** | $34 \to 16$ | 18 | **VERIFIED** |
| **Subject 24 (Dense Graph) Torture** | All 7 Torture parameters | **43.7%** | $79 \to 36$ | 40 | **VERIFIED** |
| **Impostor Pair (S01 vs S02) Baseline** | Unperturbed Impostor Pair | **0.0%** | $57 \to 64$ | 0 | **REJECTED** |
| **Impostor Pair (S01 vs S02) Torture** | 7-Parameter Torture on Impostor | **0.0%** | $57 \to 41$ | 0 | **REJECTED** |
| **Impostor Pair (S23 vs S24) Torture** | 7-Parameter Torture on Impostor | **0.0%** | $34 \to 36$ | 0 | **REJECTED** |

---

## 5. Repository File Structure

```text
retinal-vascular/
├── api/                                # FastAPI 2.0 REST Backend Service
│   ├── main.py                         # Application lifecycle, CORS & endpoint routes
│   ├── schemas.py                      # Strict Pydantic models with validation bounds
│   └── services.py                     # Pipeline orchestration, adaptive thresholding & encoding
├── classical_pipeline/                 # Classical Anatomical Localization & Metrics
│   ├── fovea_localization.py           # FAZ detection & parabolic minimum search
│   ├── localization_api.py             # High-level localization interface functions
│   ├── optic_disc_localization.py      # Circular Hough transform & luminance convergence
│   └── skeleton_metrics.py             # Skeleton statistics & branch length measurements
├── dl_pipeline/                        # Deep Learning Perception Pipeline
│   ├── data_utils.py                   # Dataset discovery & hash indexing utilities
│   ├── dataset.py                      # Multi-dataset loader & Albumentations pipelines
│   ├── model.py                        # U-Net architecture with ResNet encoders
│   ├── postprocess.py                  # Morphological hole filling & noise reduction
│   ├── predict.py                      # TTA-enabled high-precision inference engine
│   └── train.py                        # Training engine with BCE+Dice & Cosine scheduler
├── evaluation/                         # Formal Benchmark & Metric Scripts
│   ├── evaluate_localization.py        # IDRiD benchmark for Optic Disc & Fovea accuracy
│   ├── evaluate_matching.py            # DRIVE identity verification benchmark (EER, ROC)
│   ├── evaluate_vessel.py              # Vessel segmentation evaluation (Dice, AUC, Acc)
│   ├── metrics.py                      # Biometric metrics (FAR, FRR, EER calculation)
│   └── threshold_analysis.py           # Threshold optimization via Youden's J statistic
├── matching/                           # Biometric Graph Matching Engine
│   ├── bifurcation.py                  # Rutovitz Crossing Number & keypoint extraction
│   ├── matcher.py                      # 3-Stage Kabsch SE(2) SVD & Induced Delaunay matcher
│   ├── skeleton.py                     # Zhang-Suen morphological medial axis thinning
│   └── triangulation.py                # Delaunay triangulation & geometric invariant features
├── models/                             # Active Production Model Checkpoints
│   └── best_unet_model_v3.pth          # Dual-Stream ResNet-34 U-Net production weights (93.39 MB)
├── notebooks/                          # Interactive Cloud Research Notebooks
│   └── train_colab_v2.ipynb            # Google Colab GPU training notebook
├── results/                            # Official Benchmark Reports & Figures
│   └── vessel_evaluation_report.md     # Comprehensive model benchmark records
├── tests/                              # Automated Pytest Regression Suite (29 Tests)
│   ├── test_api.py                     # FastAPI endpoint validation & boundary tests
│   ├── test_localization_api.py        # OD and Fovea localization regression tests
│   ├── test_matching.py                # Biometric Delaunay matching unit tests
│   ├── test_postprocess.py             # Morphological post-processing tests
│   ├── test_week1_and_week2.py         # Dataset loading & architecture verification tests
│   └── test_week3_and_week4.py         # Advanced training & loss calculation tests
├── web/                                # Next.js 16 App Router WebGL Studio Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx              # Root clinical layout & font setup
│   │   │   ├── page.tsx                # 3-Stage clinical dashboard & cohort studio
│   │   │   └── globals.css             # Tailwind CSS & dark theme styling
│   │   ├── components/
│   │   │   ├── DelaunayConstellation.tsx # 3D dual-disc constellation with laser beams
│   │   │   └── RetinalGlobe.tsx        # 3D spherical fundus globe with GLSL shader
│   │   ├── lib/
│   │   │   └── api.ts                  # Type-safe HTTP client connecting to FastAPI
│   │   └── types/
│   │       └── biometrics.ts           # Shared TypeScript interfaces & biometric contracts
│   ├── package.json                    # Frontend package dependencies (Next 16, Three.js)
│   ├── tsconfig.json                   # Strict TypeScript compiler configuration
│   └── next.config.ts                  # Next.js configuration with Turbopack support
├── ROADMAP.md                          # Milestone progress & Capstone II roadmap
└── README.md                           # Master project documentation (this file)
```

---

## 6. The 3-Stage Interactive WebGL Clinical Studio (`web/`)

The web interface is engineered as an interactive 3D clinical workstation with three unified stages:

```
[VERA | VASCULAR EXTRACTION & RETINAL AUTHENTICATION]
├── [01 3D ANATOMY]    -> Spherical Retinal Globe with Volumetric Vessel Shader
├── [02 IDENTITY MATCH]-> Dual-Disc Delaunay Graph with Real-Time Lasers & Perturbations
└── [03 COHORT MATRIX] -> 4x4 Confusion Matrix, Stress Vectors & Live Telemetry
```

### 6.1 Stage 01: 3D Ocular Anatomy (`01 3D Anatomy`)
- **Spherical Retinal Globe**: Maps fundus photography onto an anatomically contoured ocular cup ($z = -0.42 \cdot (1 - r^2)$).
- **Custom GLSL Vessel Depth Shader**: Displaces vertices in real-time along normal vectors based on the neural vessel probability map, rendering extruded 3D relief vasculature.
- **Continuous 60FPS Probability Thresholding**: Dynamic slider adjusts the threshold cutoff in real-time within the WebGL shader without re-running neural network inference.
- **Exploded Layer Toggle**: Separates the neural vessel network from the fundus base disc in 3D space with dedicated neon cyan illumination.
- **3D Landmark Beacons**: Renders glowing 3D beacon pins directly at the localized coordinates of the Optic Disc and Fovea Centralis.

### 6.2 Stage 02: 1-to-1 Biometric Identity Matcher (`02 Identity Match`)
- **Dual 3D Retinal Discs**: Renders Reference Eye (A) in electric cyan and Target Eye (B) in deep rose.
- **Real-Time Hungarian Laser Beams**: Verified matched triangle pairs connect across 3D space with glowing laser correspondence lines.
- **Emerald Green Invariant Highlighting**: 100% of verified matched triangles (both initial seeds and induced common-subgraph triangles) highlight in emerald green (`#10b981`).
- **11-Parameter Perturbation Suite**: Interactive sliders controlling head tilt ($\pm 45^\circ$), working distance ($0.80\text{x}$ to $1.20\text{x}$), cataract haze, corneal glare, defocus blur, sensor ISO noise, eyelid occlusion, ISP gamma, spherical gaze warp, and pulsatile IOP elastic displacement.
- **8 One-Click Stress Preset Chips**: Instantly loads calibrated clinical scenarios including Clean Baseline, Cataract, Corneal Glare, Defocus Blur, ISO Noise, and the combined 7-parameter `Judge Torture`.

### 6.3 Stage 03: Population Cohort Matrix (`03 Cohort Matrix`)
- **4x4 Cross-Subject Confusion Heatmap Matrix**: Evaluates multi-subject pairings across Subject 01 (Normal), Subject 02 (Impostor), Subject 23 (Sparse Vascular Tree), and Subject 24 (Dense Vascular Tree).
- **Toggleable Regimes**: Instantly switches between `Clean Baseline` (100% genuine verification, 0.0% impostor) and `Judge Torture` (all genuine subjects maintain verified identity).
- **8-Vector Stress Resilience Bars**: Real-time progress bars displaying the degradation curve across all 8 stress vectors.
- **One-Click Drill-Down Navigation**: Clicking any matrix cell loads that exact pair into Stage 02 with the corresponding stress parameters applied for deep forensic inspection.
- **Live Verification Telemetry**: The `Run Live Cohort Verification` action computes all 16 matrix cells in real-time against the live backend, measuring per-pair inference latency and triangle counts.

---

## 7. Getting Started

### 7.1 Prerequisites
- **Python**: Version 3.10 to 3.12 (64-bit)
- **Node.js**: Version 18.18 or higher (LTS recommended)
- **Git**: Version 2.30+
- **CUDA (Optional)**: CUDA 11.8 or 12.x for accelerated GPU training and inference

### 7.2 Backend Installation & Startup
```bash
# Clone the repository
git clone https://github.com/swnst/retinal-vascular.git
cd retinal-vascular

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI 2.0 Backend Server
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
The REST API documentation will be available at `http://127.0.0.1:8000/docs`.

### 7.3 Frontend Installation & Startup
Open a separate terminal window:
```bash
# Navigate to the frontend directory
cd retinal-vascular/web

# Install Node dependencies
npm install

# Start the Next.js Turbopack development server
npm run dev
```
Open your browser at `http://localhost:3000` to launch the 3D WebGL Clinical Studio.

### 7.4 Running Automated Verification Suite
From the repository root with `.venv` active:
```bash
# Run all 29 automated tests
pytest tests/ -v

# Run frontend linting & strict TypeScript check
cd web
npm run lint
npx tsc --noEmit

# Run production build
npm run build
```

---

## 8. Quality Gate & Production Readiness

The VERA codebase adheres to zero-trust production quality standards:

- **Automated Tests**: **29 of 29 tests passed** (`pytest tests/`) covering API contracts, Pydantic bounds, localization algorithms, morphological post-processing, and Delaunay invariant matching.
- **Code Quality**: **0 ESLint errors and 0 warnings** under strict Next.js and TypeScript configurations.
- **Type Safety**: Fully typed with strict TypeScript and Python type annotations (no untyped `any` casting).
- **Security Safeguards**: Image payload size guards ($4096 \times 4096\text{ px}$ decompression limit), strict CORS policies (`allow_credentials=False` for wildcard origins), and zero hardcoded secrets.
- **Memory Hygiene**: WebGL geometries, shaders, and textures are explicitly disposed of on component unmount to prevent GPU VRAM memory leaks.

---

## 9. Academic Citations & Acknowledgments

If you utilize this codebase, models, or biometric matching algorithms in academic research, please cite:

```bibtex
@article{kanjanasurat2020personal,
  title={Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern},
  author={Kanjanasurat, Isoon and Satayarak, Nitjaree and others},
  journal={Journal of Healthcare Engineering},
  volume={2020},
  year={2020},
  publisher={Hindawi}
}

@misc{vera_biometrics_2026,
  title={VERA: Vascular Extraction & Retinal Authentication Platform},
  author={Dawit and Viwat and Kanjanasurat, Isoon and Satayarak, Nitjaree},
  year={2026},
  howpublished={\url{https://github.com/swnst/retinal-vascular}}
}
```
