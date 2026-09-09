# Retinal Vascular Identification — Project Roadmap (Capstone I & II)

## โปรเจกต์จบ: ระบบยืนยันตัวบุคคลจากลวดลายหลอดเลือดจอประสาทตา
- **อาจารย์ที่ปรึกษา**: อ. อิสูรย์ กาญจนสุรัตน์ (Isoon Kanjanasurat), ดร. นิจจารีย์ สัตยารักษ์ (Nitjaree Satayarak)
- **คณะผู้พัฒนา**:
  - ดวิษ (Classical Image Processing Pipeline & Keypoint Matching)
  - วิวรรธน์ (Deep Learning Pipeline, Biometrics Matching & System Integration)
- **อ้างอิงและต่อยอดจาก**: Kanjanasurat et al. (2020) — *Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern*

---

## 1. Technical Decisions & System Specifications

- **Framework**: PyTorch + `segmentation_models_pytorch` (smp) + Albumentations
- **Architecture**: U-Net with Pretrained ResNet-34 / ResNet-50 Encoder
- **Loss Function**: Combined Loss (BCEWithLogitsLoss 0.5 + DiceLoss 0.5)
- **Post-processing & Inference**:
  - Probability Threshold: 0.60 (Validated via ROC & Youden's J Analysis)
  - Test-Time Augmentation (TTA): Horizontal Flip Averaging (+0.56% Dice boost)
  - Morphological Filtering: Small component noise removal (< 25px) + Hole filling
- **Landmark Localization**:
  - Optic Disc: Vessel tree convergence + green channel brightness prior
  - Fovea: Macular vessel sparsity + anatomical distance prior relative to OD
- **Biometric Matching**:
  - Feature Points: Vascular Bifurcations & Crossovers via Rutovitz Crossing Number
  - Graph Topology: Delaunay Triangulation restricted to peri-papillary zone
  - Invariant Descriptors: Adjacent triangle area ratios + internal angle preservation
- **Datasets**:
  - Vessel Segmentation: DRIVE (40) + STARE (20) + HRF (45)
  - Localization Benchmark: IDRiD C. Localization Testing Set (103)
  - Biometric Matching Benchmark: DRIVE Identity Pairs (20 Genuine, 190 Impostor)

---

## 2. Milestone Progress & Execution Summary (Capstone I)

### Phase 1: Environment & Dataset Pipeline (Completed)
- โครงสร้างโปรเจกต์แบบ Clean Architecture พร้อม Strict Type Hinting
- Data utilities จับคู่ภาพและ Mask (DRIVE, STARE, HRF) ด้วย Hash Indexing
- Dataset Pipeline และ Albumentations data augmentation

### Phase 2: Deep Learning Vessel Segmentation (Completed)
- โมเดล U-Net ResNet-34 (Pretrained ImageNet)
- เทรนโมเดลบน Google Colab (Tesla T4 GPU)
- ผลการประเมิน: DRIVE Dice = 0.7848, AUC = 0.9652, Accuracy = 96.14%

### Phase 3: Model Improvement & Hyperparameter Optimization (Completed)
- Threshold Optimization: ขยับจาก 0.50 เป็น 0.60 เพิ่ม Specificity เป็น 98.66%
- Test-Time Augmentation (TTA) เพิ่ม Dice บน Validation set เป็น 0.7324
- พัฒนา `dl_pipeline/postprocess.py` ลบจุดรบกวนขยะรอบจอประสาทตา

### Phase 4: Classical Pipeline Integration & Formal Localization Benchmark (Completed)
- นำเข้าโมดูล `classical_pipeline/` (ดวิษ) และตั้งสัญญา Interface แบบสมบูรณ์
- ประเมินผลบน IDRiD Testing Set (103 ภาพความละเอียด 4288x2848):
  - **Optic Disc (OD) Success Rate**: **99.03%** (Mean Error: 68.18 px หรือ 1.59% ของความกว้างภาพ)
  - **Fovea Success Rate**: **94.17%** (Mean Error: 108.11 px หรือ 2.52% ของความกว้างภาพ)

### Phase 5: Biometric Identity Matching Engine (Completed)
- พัฒนาโมดูล `matching/`:
  - `skeleton.py`: Zhang-Suen / morphological thinning
  - `bifurcation.py`: Rutovitz Crossing Number ตรวจหาจุดแตกแขนงและจุดตัด
  - `triangulation.py`: Delaunay Triangulation + Multi-attribute Geometric Invariants
  - `matcher.py`: Invariant matching and identity verification
- ผลการประเมินบน DRIVE Benchmark:
  - **Equal Error Rate (EER)**: **4.34%**
  - **Verification Accuracy @ EER**: **96.19%**
  - **Optimal Threshold**: **0.200**

### Phase 6: Interactive Demo Application (Completed)
- พัฒนา Web Application ผ่าน Gradio (`demo/app.py`):
  - Tab 1: Vessel Segmentation (Overlay + Heatmap)
  - Tab 2: Landmark Detection (OD + Fovea Coordinates)
  - Tab 3: Biometric Identity Matching (Delaunay Graph + Similarity Verification)

---

## 3. Capstone II Future Directions (เทอมหน้า)

1. **Learned Feature Descriptors**: พัฒนา Contrastive Learning / Siamese Network หรือ GNN บน Vessel Graph แทน Handcrafted Invariant Ratios
2. **Advanced Segmentation Backbone**: อัปเกรดเป็น Attention U-Net / Swin-UNet และทดสอบ Patch-based 1024x1024
3. **Cross-Database Domain Adaptation**: ทดสอบข้าม Dataset หลากหลายเชื้อชาติ (เช่น CHASE_DB1, ARIA)
4. **Hardware Deployment**: ย่อส่วนโมเดลและพัฒนา Edge Device หรือระบบ Mobile Authentication
