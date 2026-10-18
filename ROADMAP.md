# Retinal Vascular Identification: Engineering and Research Roadmap

## Senior Capstone Project (Capstone I and Capstone II)
Department of IoT and Information Engineering  
Program in IoT and Information Engineering and Industrial Physics (Dual Degree)  
School of Engineering, King Mongkut's Institute of Technology Ladkrabang (KMITL)

- Project Authors:
  - Wiwat Suwannasit (วิวรรธน์ สุวรรณสิทธิ์)
  - Dvis Vichitlaksana (ดวิษ วิจิตรลักษณา)
- Academic Advisors:
  - Asst. Prof. Isoon Kanjanasurat (ผศ. อิสูรย์ กาญจนสุรัตน์)
  - Asst. Prof. Dr. Nitjaree Satayarak (ผศ. ดร. นิจจารีย์ สัตยารักษ์)
- Theoretical Reference:
  - Kanjanasurat et al. (2020), "Vascular Extraction by using matched filter on retinal image", *Journal of Physics: Conference Series*, DOI: 10.1088/1742-6596/1457/1/012013
  - Kanjanasurat, Purahong, Pintavirooj, Satayarak, and Benjangkaprasert (2020), "Blood Vessel Extraction and Optic Disk Localization for Diabetic Retinopathy", *ACM ICBET*, DOI: 10.1145/3397391.3397425
  - Kanjanasurat et al. (2020), "Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern", *IJICIC*, Vol. 16, No. 3, pp. 879-897, URL: http://www.ijicic.org/ijicic-160307.pdf

---

## 1. Technical Baseline and Architecture Decisions

- **Perception Framework**: PyTorch 2.x, `segmentation_models_pytorch`, Albumentations
- **Segmentation Backbone**: U-Net architecture with ImageNet-pretrained ResNet-34 encoder (`best_unet_model_v3.pth`)
- **Optimization Objective**: Balanced Composite Loss (0.50 BCEWithLogitsLoss + 0.50 Soft DiceLoss)
- **Inference Pipeline**:
  - Test-Time Augmentation (TTA): Horizontal flip averaging
  - **Quantile-Calibrated Adaptive Thresholding**: $\tau_{\text{eff}} = \max(0.32, \min(0.60, 0.65 \cdot p_{95}))$ calculated dynamically per scan Field of View (FOV)
  - Morphological Cleanup: Connected-component noise removal ($< 25$ px) and binary hole filling
- **Anatomical Landmark Localization Subsystem**:
  - Optic Disc (OD): Circular Hough Transform coupled with Green-channel luminance profiling and major vessel convergence
  - Fovea Centralis: Foveal Avascular Zone (FAZ) parabolic intensity search bounded by anatomical distance priors from OD (2.0 to 2.5 OD diameters)
- **Biometric Topology and Invariant Matching**:
  - Medial Axis Thinning: Zhang-Suen morphological skeletonization
  - Minutiae Classification: Rutovitz Crossing Number ($X=3$ for bifurcations, $X=4$ for crossovers) with spur filtering
  - Verification Engine: **3-Stage Kabsch SE(2) SVD Constellation Consensus + Induced Common-Subgraph Delaunay Triangulation**
- **11-Parameter Physical Degradation Suite**:
  - In-plane tilt ($\pm 45^\circ$), axial scale/zoom ($0.80\text{x} - 1.20\text{x}$), illumination shift, defocus blur, cataract media haze, corneal surface glare, CMOS ISO noise, eyelid occlusion, gamma correction, Brown-Conrady radial lens warp, and Simard pulsatile IOP elastic displacements
- **Software Stack**:
  - Backend: FastAPI 2.0 with Pydantic v2 data models, explicit value constraints, decompression bomb defenses, and CORS middleware
  - Frontend: Next.js 16 App Router (Turbopack) with Three.js WebGL GPU Studio (3D Retinal Globe, Dual Laser Matcher, 4x4 Cohort Matrix)

---

## 2. Milestone Execution and Delivery Status (Capstone I)

### Baseline Proposal vs. Delivered Reality

In the original Capstone I proposal, the primary objective was a proof-of-concept demonstrating basic U-Net vessel segmentation, bifurcation extraction, and Delaunay triangle invariant verification on static DRIVE images, packaged as a single-file prototype.

The actual delivered implementation exceeds the initial proposal across every system dimension:

```
Initial Capstone I Target:
[Basic U-Net] ──> [Bifurcation Points] ──> [Delaunay Invariants] ──> [Single Script Prototype]

Delivered Production Platform:
[U-Net v3 + TTA] ──> [Hough + FAZ Localization] ──> [Kabsch SVD Matcher] ──> [FastAPI 2.0 REST] ──> [Next.js 16 3D Studio]
       │                         │                               │                         │                          │
 (0.7848 Dice)           (99.03% OD Success)           (Judge Torture 63.4%)         (33 Pytest Passed)          (3 Connected Stages)
```

### Phase 1: Environment, Dataset Pipeline, and Architecture (Completed)
- Established clean decoupled project structure separating `classical_pipeline`, `dl_pipeline`, `matching`, `api`, and `web`.
- Implemented robust multi-dataset loader supporting DRIVE (40 images), STARE (20 images), and high-resolution HRF (45 images) with deterministic hashing.
- Constructed dual-stream photometric and spatial data augmentation pipeline via Albumentations.

### Phase 2: Deep Learning Vessel Perception Backbone (Completed)
- Implemented ResNet-34 U-Net model with composite BCE + Soft Dice loss.
- Evaluated checkpoint progression: v1 (0.7682 Dice) to v2 (0.7765 Dice) to production model v3 (**0.7848 Dice, 0.9652 ROC-AUC, 98.66% Specificity** on DRIVE test set).
- Developed Quantile-Calibrated Adaptive Thresholding, replacing brittle static cutoffs with distribution-aware binarization.
- Integrated Test-Time Augmentation (TTA) and morphological artifact suppression.

### Phase 3: Anatomical Landmark Localization (Completed)
- Developed Optic Disc localization utilizing Green-channel luminance and circular Hough transform.
- Implemented Fovea Centralis detection combining FAZ parabolic intensity minimum search with vessel convergence vectors.
- Validated accuracy on 103 high-resolution images from the IDRiD benchmark: **99.03% OD success** (mean error 14.8 px) and **94.17% Fovea success** (mean error 26.3 px).

### Phase 4: Biometric Minutiae and Matching Engine (Completed)
- Implemented Zhang-Suen morphological skeletonization and Rutovitz Crossing Number minutiae extraction.
- Engineered **3-Stage Kabsch SE(2) SVD Constellation Consensus + Induced Common-Subgraph Delaunay Triangulation**.
- Resolved the fundamental vulnerability of independent Delaunay re-triangulation under vertex dropout.
- Achieved **63.4% matching score** under 7 concurrent clinical stressors (`Judge Torture` preset), safely exceeding the 20.0% verification threshold while maintaining 0.00% False Acceptance Rate on impostor pairs.

### Phase 5: Enterprise Backend and Degradation Suite (Completed)
- Developed FastAPI 2.0 backend with strict Pydantic v2 request contracts, type validation, and input sanitization.
- Built 11-parameter physical degradation engine modeling optical, atmospheric, sensor, and biomechanical clinical stress vectors.
- Implemented automated test suite (`pytest tests/`) achieving **33 passing tests out of 33** with zero failures.

### Phase 6: Three.js WebGL 3D Studio Frontend (Completed)
- Built responsive Next.js 16 App Router application with Turbopack compilation.
- Stage 01: Interactive 3D Retinal Globe with custom GLSL depth shader and anatomical landmark focus.
- Stage 02: Dual 3D Retinal Matcher with real-time 3D laser correspondence beams and 2D Delaunay constellation overlay.
- Stage 03: 4x4 Population Cohort Matrix evaluating all 16 pairwise identity combinations across nominal and stress regimes.
- Rolled back UI branding to clean clinical typography, eliminating non-standard graphic elements.

---

## 3. Capstone II Research and Development Horizons

Building on the verified engineering platform delivered in Capstone I, Capstone II focuses on four advanced research and deployment tracks:

### Track 1: Graph Neural Networks (GNN) and Topological Deep Descriptors
- **Limitation of Current Heuristic**: While Kabsch SVD alignment and Delaunay triangle invariant matching are mathematically grounded, they rely on handcrafted geometric rules that do not capture deep neighborhood vascular context.
- **Capstone II Objective**:
  - Formulate the retinal vascular skeleton as an attributed spatial graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, X_v, X_e)$, where vertices represent bifurcations/crossovers and edges represent vessel segments with caliber, curvature, and length attributes.
  - Implement a Message Passing Neural Network (MPNN) or Point Transformer architecture to generate permutation-invariant, rotation-invariant vascular embeddings.
  - Train the network with metric learning objectives (SupCon / InfoNCE) to directly minimize cross-entropy between genuine embeddings while maximizing distance to impostors.

### Track 2: Cross-Database Domain Adaptation and Generalization
- **Limitation of Current Model**: High segmentation performance on DRIVE (0.7848 Dice) is subject to domain shift when applied to fundus cameras with different optical transfer functions or sensor sensor characteristics (e.g., Zeiss FF450 vs. Topcon TRC-50DX vs. Canon CR-2).
- **Capstone II Objective**:
  - Implement unsupervised domain adaptation (UDA) utilizing adversarial domain discriminators or Fourier domain adaptation (FDA) to align feature distributions across diverse datasets: DRIVE, STARE, HRF, CHASE_DB1, and IDRiD.
  - Evaluate zero-shot generalization across ultra-widefield Scanning Laser Ophthalmoscopy (UWF-SLO) images from the Optos platform.

### Track 3: Embedded Edge AI and Handheld Ophthalmoscope Deployment
- **Limitation of Current Platform**: The active system operates on desktop workstation environments requiring a dedicated Python server and WebGL client.
- **Capstone II Objective**:
  - Quantize the ResNet-34 U-Net model from FP32 to INT8 via TensorRT and ONNX Runtime, targeting sub-50ms inference latency per frame.
  - Deploy the complete segmentation, localization, and matching pipeline onto embedded edge hardware (NVIDIA Jetson Orin Nano / Raspberry Pi 5 with Hailo-8 AI accelerator) for integration into low-cost portable fundus cameras.

### Track 4: Multi-Modal Ophthalmic Identity and Systemic Biomarker Fusion
- **Extended Utility**: The human retina provides a non-invasive direct optical window into the microcirculation of the body.
- **Capstone II Objective**:
  - Integrate automated clinical diagnostic metrics into the verification workflow: Arteriolar-to-Venular Ratio (AVR), vessel tortuosity index, and vascular fractal dimension.
  - Implement multi-task learning where the network outputs both biometric identity verification and risk scores for systemic cardiovascular conditions (hypertension, diabetic microangiopathy, arteriosclerosis).

---

## 4. Technical Debt, Risk Analysis, and Mitigation Strategy

| Risk Factor | Severity | Current Status in Capstone I | Capstone II Mitigation Strategy |
| :--- | :---: | :--- | :--- |
| **Model Generalization on Low-Quality Scans** | Medium | Managed via quantile adaptive binarization ($\tau_{\text{eff}}$) | Incorporate synthetic blur and cataract haze into training augmentations via self-supervised contrastive learning |
| **Bifurcation Dropout under Extreme Blur** | Medium | Mitigated via Kabsch SVD induced common subgraph | Deploy edge-aware multi-scale vessel enhancement filters prior to skeletonization |
| **Inference Latency on High-Res Images (HRF)** | Low | Full-resolution processing takes ~1.2s on GPU | Implement progressive tiling inference or dynamic resolution downsampling with coordinate re-projection |
| **Optical Axis Misalignment (3D Tilt)** | Low | Handled up to $\pm 45^\circ$ in-plane and spherical warp | Develop 3D spherical mesh registration directly in the WebGL engine to model non-planar eye rotations |

---

## 5. Capstone Timeline and Milestone Schedule

```
Capstone I (Completed):
  Week 01-04: Repository Setup, Multi-Dataset Pipeline, Dataset Discovery
  Week 05-08: ResNet-34 U-Net Perception, Composite Loss, TTA, Adaptive Binarization
  Week 09-11: Landmark Localization (OD & Fovea), Minutiae Extraction, Pytest Suite
  Week 12-14: Kabsch SVD Matcher, 11-Param Degradation Suite, 3D WebGL Studio
  Week 15-16: System Hardening, Academic Compendium, Peer-Reviewed Verification

Capstone II (Upcoming Horizons):
  Month 01-02: Graph Neural Network (GNN) Formulation & Topological Embedding
  Month 03-04: Cross-Dataset Domain Adaptation (HRF, CHASE_DB1, STARE, UWF-SLO)
  Month 05-06: INT8 Model Quantization & NVIDIA Jetson Edge Deployment
  Month 07-08: Multi-Modal Ophthalmic Fusion (AVR, Tortuosity, Systemic Biomarkers)
```
