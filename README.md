# Retinal Vascular Identification — ระบบยืนยันตัวบุคคลจากลวดลายหลอดเลือดจอตา

โครงงานระดับปริญญานิพนธ์ (Senior Capstone Project): ระบบยืนยันตัวบุคคลจากชีวมิติลวดลายหลอดเลือดบนจอประสาทตา (Retinal Vascular Pattern Biometrics)

- **อาจารย์ที่ปรึกษา**: อ. อิสูรย์ กาญจนสุรัตน์ (Isoon Kanjanasurat), ดร. นิจจารีย์ สัตยารักษ์ (Nitjaree Satayarak)
- **คณะผู้พัฒนา**:
  - ดวิษ (Classical Image Processing Pipeline & Keypoint Matching)
  - วิวรรธน์ (Deep Learning Pipeline: Vessel Segmentation & Feature Extraction)
- **อ้างอิงและต่อยอดจาก**: Kanjanasurat et al. (2020) — *Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern*

---

## 1. ภาพรวมระบบ (System Architecture)

ระบบทำการยืนยันตัวบุคคลแบบ End-to-End ตั้งแต่การรับภาพถ่ายจอประสาทตาจนถึงการจับคู่ชีวมิติ:
1. **Vessel Segmentation (Deep Learning)**: U-Net with ResNet-34 Encoder ทำนายลวดลายหลอดเลือดอย่างแม่นยำ พร้อม Test-Time Augmentation (TTA) และ Morphological Post-processing
2. **Anatomical Landmarks Localization (Classical Pipeline)**: ตรวจหาพิกัดกึ่งกลางจานประสาทตา (Optic Disc) และจุดรับภาพชัด (Fovea) ด้วยการวิเคราะห์การรวมตัวของหลอดเลือดและความสว่าง
3. **Keypoint Extraction (Topology)**: สกัดจุดแยก (Bifurcations) และจุดตัด (Crossovers) ด้วย Rutovitz Crossing Number
4. **Delaunay Triangulation & Invariant Matching**: สร้างโครงข่ายสามเหลี่ยม Delaunay รอบ Optic Disc และคำนวณ Multi-attribute Geometric Invariants (Area Ratios + Internal Angles) เพื่อใช้จับคู่ยืนยันตัวตน

---

## 2. โครงสร้างโปรเจกต์ (Repository Structure)

```text
retinal-vascular/
├── classical_pipeline/         # อัลกอริทึมหาตำแหน่งจุดสำคัญ (ดวิษ)
│   ├── localization_api.py     # Unified API สำหรับเชื่อมต่อ (find_optic_disk, find_fovea)
│   ├── optic_disc_localization.py
│   └── fovea_localization.py
├── dl_pipeline/                # Deep Learning Pipeline (วิวรรธน์)
│   ├── dataset.py              # PyTorch Dataset รองรับ Augmentation & Multi-dataset
│   ├── model.py                # U-Net Architecture (ResNet-34 / ResNet-50)
│   ├── train.py                # Training Engine (BCE + Dice Loss)
│   ├── predict.py              # High-Precision Inference พร้อม TTA
│   └── postprocess.py          # Morphological Filtering กำจัด Noise
├── matching/                   # ระบบชีวมิติ Identity Matching (วิวรรธน์)
│   ├── skeleton.py             # Vessel Thinning / Skeletonization
│   ├── bifurcation.py          # ตรวจหา Bifurcations & Crossovers ผ่าน Crossing Number
│   ├── triangulation.py        # Delaunay Triangulation & Geometric Invariants
│   └── matcher.py              # Template Extraction & Invariant Matching Engine
├── evaluation/                 # การทดสอบและประเมินผลเชิงปริมาณ
│   ├── evaluate_vessel.py      # วัดผล Vessel Segmentation (Dice, AUC, Acc, Sens, Spec)
│   ├── evaluate_localization.py# Benchmark IDRiD วัดความคลาดเคลื่อนพิกัด OD & Fovea
│   ├── evaluate_matching.py    # Benchmark DRIVE คำนวณ EER, FAR, FRR, ROC
│   └── threshold_analysis.py   # วิเคราะห์ Optimal Threshold ด้วย Youden's J Index
├── demo/                       # Interactive Web Application
│   ├── app.py                  # Gradio 3-Tab Interface (Segmentation, Landmarks, Matching)
│   └── requirements.txt
├── notebooks/                  # Training Pipelines & Colab Notebooks
│   ├── train_pipeline.py       # Standalone training script
│   └── train_colab_v2.ipynb    # Google Colab Ready Notebook
├── results/                    # รายงานผลการทดลองและกราฟสถิติ (Reports, Curves, Plots)
├── tests/                      # Unit Tests ครอบคลุมทุกโมดูล (20 passing tests)
├── ROADMAP.md                  # แผนการดำเนินงานและผลลัพธ์เชิงตัวเลข
└── README.md
```

---

## 3. ผลการทดลองเชิงปริมาณ (Capstone I Benchmark Results)

### 3.1 การแบ่งส่วนหลอดเลือด (Retinal Vessel Segmentation)
- **Validation Dice Score**: **0.7324** (Threshold = 0.60 + Test-Time Augmentation)
- **Specificity**: **98.66%** (พื้นหลังสะอาด ปราศจาก False Positive Noise)
- **DRIVE Test Set**: Accuracy = **96.14%**, AUC = **0.9652**, Dice = **0.7848**

### 3.2 การระบุตำแหน่งจุดอ้างอิงทางกายวิภาค (IDRiD Testing Set - 103 ภาพ)
- **Optic Disc Localization**: Mean Error = **68.18 px** (1.59% Image Width), **Success Rate = 99.03%**
- **Fovea Localization**: Mean Error = **108.11 px** (2.52% Image Width), **Success Rate = 94.17%**

### 3.3 การยืนยันตัวบุคคลด้วยชีวมิติ (DRIVE Biometric Matching)
- **Equal Error Rate (EER)**: **4.34%** (ความผิดพลาดต่ำมาก)
- **Verification Accuracy @ EER**: **96.19%**
- **Optimal Operating Threshold**: **0.200**
- **คะแนนความเหมือน Genuine**: เฉลี่ย **0.3411** vs **Impostor**: เฉลี่ย **0.1225**

---

## 4. วิธีการติดตั้งและเริ่มต้นใช้งาน (Getting Started)

### 4.1 การเตรียม Environment
```bash
# Clone repository
git clone https://github.com/swnst/retinal-vascular.git
cd retinal-vascular

# สร้างและเปิดใช้งาน venv
python -m venv .venv
.venv\Scripts\activate

# ติดตั้ง Dependencies
pip install -r requirements.txt
pip install gradio
```

### 4.2 การรันชุดทดสอบ (Automated Unit Tests)
```bash
pytest tests/ -v
```

### 4.3 การเปิดใช้งาน Interactive Web Demo
เปิด Web App สำหรับทดลองใช้งานและนำเสนอผลงาน:
```bash
python demo/app.py
```
เปิดบราวเซอร์ไปที่ `http://127.0.0.1:7860` เพื่อเข้าสู่ระบบเดโม 3 แท็บ (Segmentation, Landmarks, Biometrics)
