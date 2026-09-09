# Retinal Vascular Identification — Personal Biometric Authentication System

Senior Capstone Project: Personal Identification and Verification System using Retinal Vascular Pattern Biometrics.

- **Advisors**: Asst. Prof. Isoon Kanjanasurat, Dr. Nitjaree Satayarak
- **Development Team**:
  - Dawit (Classical Image Processing Pipeline & Keypoint Matching)
  - Viwat (Deep Learning Pipeline: Vessel Segmentation & Feature Extraction)
- **Reference & Extension**: Kanjanasurat et al. (2020) — *Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern*

---

## 1. System Architecture

The system provides an end-to-end identity verification pipeline from retinal fundus imaging to biometric matching:
1. **Vessel Segmentation (Deep Learning)**: U-Net with ResNet-34 Encoder for precise vascular segmentation, enhanced with Test-Time Augmentation (TTA) and morphological post-processing.
2. **Anatomical Landmarks Localization (Classical Pipeline)**: Identifies center coordinates of Optic Disc and Fovea landmarks through vessel convergence and intensity analysis.
3. **Keypoint Extraction (Topology)**: Extracts vessel bifurcations and crossovers via Rutovitz Crossing Number.
4. **Delaunay Triangulation & Invariant Matching**: Constructs Delaunay triangular network centered around the Optic Disc and computes multi-attribute geometric invariants (area ratios and internal angles) for rotation/scale-invariant authentication.

---

## 2. Repository Structure

```text
retinal-vascular/
├── classical_pipeline/         # Landmark localization algorithms
│   ├── localization_api.py     # Unified API integration (find_optic_disk, find_fovea)
│   ├── optic_disc_localization.py
│   └── fovea_localization.py
├── dl_pipeline/                # Deep Learning vessel segmentation pipeline
│   ├── dataset.py              # PyTorch Dataset supporting Augmentation & Multi-dataset loading
│   ├── model.py                # U-Net Architecture (ResNet-34 / ResNet-50)
│   ├── train.py                # Training Engine (BCE + Dice Loss)
│   ├── predict.py              # High-Precision Inference with TTA
│   └── postprocess.py          # Morphological Filtering & Noise Removal
├── matching/                   # Biometric Identity Matching Engine
│   ├── skeleton.py             # Vessel Thinning & Skeletonization
│   ├── bifurcation.py          # Bifurcation & Crossover extraction via Crossing Number
│   ├── triangulation.py        # Delaunay Triangulation & Geometric Invariants
│   └── matcher.py              # Template Extraction & Invariant Matching Engine
├── evaluation/                 # Quantitative Evaluation and Benchmarking
│   ├── evaluate_vessel.py      # Vessel Segmentation Evaluation (Dice, AUC, Acc, Sens, Spec)
│   ├── evaluate_localization.py# IDRiD Landmark Benchmark for OD & Fovea coordinates
│   ├── evaluate_matching.py    # DRIVE Biometric Benchmark (EER, FAR, FRR, ROC)
│   └── threshold_analysis.py   # Optimal Threshold Determination via Youden's J Index
├── demo/                       # Interactive Web Application
│   ├── app.py                  # Gradio 3-Tab Interface (Segmentation, Landmarks, Matching)
│   └── requirements.txt
├── notebooks/                  # Training Pipelines & Colab Notebooks
│   ├── train_pipeline.py       # Standalone training script
│   └── train_colab_v2.ipynb    # Google Colab Ready Notebook
├── results/                    # Experimental reports, curves, and visualization figures
├── tests/                      # Automated Unit Test Suite (20 passing tests)
├── ROADMAP.md                  # Development roadmap and benchmark tracking
└── README.md
```

---

## 3. Quantitative Benchmark Results (Capstone I)

### 3.1 Retinal Vessel Segmentation
- **Validation Dice Score**: **0.7324** (Threshold = 0.60 + Test-Time Augmentation)
- **Specificity**: **98.66%** (Clean background suppression without false positives)
- **DRIVE Test Set**: Accuracy = **96.14%**, AUC = **0.9652**, Dice = **0.7848**

### 3.2 Anatomical Landmark Localization (IDRiD Testing Set - 103 images)
- **Optic Disc Localization**: Mean Error = **68.18 px** (1.59% Image Width), **Success Rate = 99.03%**
- **Fovea Localization**: Mean Error = **108.11 px** (2.52% Image Width), **Success Rate = 94.17%**

### 3.3 Biometric Identity Authentication (DRIVE Matching Benchmark)
- **Equal Error Rate (EER)**: **4.34%** (Extremely low error rate)
- **Verification Accuracy @ EER**: **96.19%**
- **Optimal Operating Threshold**: **0.200**
- **Similarity Scores**: Genuine mean = **0.3411** vs Impostor mean = **0.1225**

---

## 4. Getting Started

### 4.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/swnst/retinal-vascular.git
cd retinal-vascular

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install gradio
```

### 4.2 Run Automated Unit Tests
```bash
pytest tests/ -v
```

### 4.3 Launch Interactive Web Demo
Start the Web Application for demonstration:
```bash
python demo/app.py
```
Open your browser at `http://127.0.0.1:7860` to access the 3-tab interactive demonstration interface (Segmentation, Landmarks, Biometrics).
