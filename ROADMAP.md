# Retinal Vascular Identification — Project Roadmap (Capstone I & II)

## โครงงานวิศวกรรมคอมพิวเตอร์: ระบบยืนยันตัวบุคคลจากลวดลายหลอดเลือดจอประสาทตาและแบบจำลองกายวิภาค 3 มิติ

- **อาจารย์ที่ปรึกษา**: อ. อิสูรย์ กาญจนสุรัตน์ (Asst. Prof. Isoon Kanjanasurat), ดร. นิจจารีย์ สัตยารักษ์ (Dr. Nitjaree Satayarak)
- **คณะผู้พัฒนา**:
  - ดวิษ (Classical Image Processing Pipeline, Anatomical Landmark Localization & Keypoint Topology)
  - วิวรรธน์ (Deep Learning Perception Pipeline, Biometric Matching Engine, API Architecture & System Integration)
- **อ้างอิงและต่อยอดจาก**: Kanjanasurat et al. (2020) — *Personal Identification Using a Delaunay Triangle and Optic Disc Retinal Vascular Pattern*

---

## 1. Technical Decisions & System Architecture

- **Deep Learning Framework**: PyTorch 2.x + `segmentation_models_pytorch` (smp) + Albumentations
- **Segmentation Backbone**: U-Net พร้อม ResNet-34 ImageNet-pretrained Encoder (`best_unet_model_v3.pth`)
- **Loss Function**: Balanced Composite Loss (BCEWithLogitsLoss 0.5 + Soft DiceLoss 0.5)
- **Inference & Binarization Strategy**:
  - Test-Time Augmentation (TTA): Horizontal Flip Averaging
  - **Quantile-Calibrated Adaptive Thresholding**: $\tau_{\text{eff}} = \max(0.32, \min(0.60, 0.65 \cdot p_{95}))$ คำนวณแบบ Dynamic ตามความสว่างของ FOV ทนทานต่อฝ้ากระจกตาและภาพเบลอ
  - Morphological Filtering: ขจัด Noise ขนาดเล็กกว่า 25px และ Morphological Hole Filling
- **Anatomical Landmark Localization**:
  - Optic Disc (OD): Circular Hough Transform ผสาน Green-channel Luminance และ Vessel Tree Convergence
  - Fovea Centralis: ตรวจหา Foveal Avascular Zone (FAZ) ผ่าน Parabolic Minimum Search ร่วมกับระยะกายวิภาคอ้างอิงจาก OD (2.0 - 2.5 OD diameters)
- **Biometric Topology & Invariant Matching**:
  - Thinning: Zhang-Suen Morphological Medial Axis Skeletonization
  - Feature Extraction: Rutovitz Crossing Number ($X(P) = 3$ สำหรับ Bifurcation, $X(P) = 4$ สำหรับ Crossover) ร่วมกับ Spur Pruning
  - Matching Engine: **3-Stage Kabsch SE(2) SVD Constellation Consensus + Induced Common-Subgraph Delaunay Triangulation** แก้ปัญหา Independent Retriangulation เมื่อจุดแตกแขนงหายไป
- **11-Parameter Clinical Perturbation Engine**:
  - Head Tilt ($\pm 45^\circ$), Scale/Zoom ($0.80\text{x} - 1.20\text{x}$), Brightness, Defocus Blur, Cataract Haze, Corneal Glare, CMOS ISO Noise, Eyelid Occlusion, ISP Gamma, Off-axis Spherical Warp, และ Pulsatile IOP Elastic Displacements
- **Software Architecture**:
  - Backend: FastAPI 2.0 พร้อม Pydantic V2 Schemas, Strict Range Bounds, Decompression Bomb Guards, และ CORS Middleware
  - Frontend: Next.js 16 App Router (Turbopack) + Three.js WebGL GPU Shader Studio (Stage 01 Globe, Stage 02 Dual Matcher, Stage 03 Cohort Matrix)

---

## 2. Milestone Execution & Achievement Summary (Capstone I)

> [!NOTE]
> **การประเมินเทียบกับเป้าหมายเดิมของ Capstone I**:
> ในข้อเสนอโครงงาน Capstone I เดิม ตั้งเป้าหมายไว้เพียงแค่การพิสูจน์ความเป็นไปได้ (Proof of Concept) ของการตัดเส้นเลือด U-Net พื้นฐาน, การดึงจุด Bifurcation, การสร้างสามเหลี่ยม Delaunay อย่างง่ายบนภาพ DRIVE ที่สะอาด และการสร้าง Prototype Demo ผ่าน Gradio สคริปต์เดี่ยว
>
> **สิ่งที่ทำสำเร็จจริงใน Capstone I (ปัจจุบัน)** ได้ก้าวข้ามเป้าหมายเดิมไปไกลมากในทุกมิติ โดยส่งมอบแพลตฟอร์มระดับ Production-Grade เต็มรูปแบบ ทั้งเอนจินคณิตศาสตร์ขั้นสูง 3-Stage Kabsch SVD Matcher, เอนจินจำลองความท้าทายทางคลินิก 11 พารามิเตอร์, สถาปัตยกรรม Decoupled Enterprise API, หน้าเว็บ 3D WebGL Studio 3 สเตจ และชุดทดสอบอัตโนมัติครบ 29 ข้อ

```
เป้าหมายเดิม Capstone I:
[U-Net พื้นฐาน] ──> [จุด Bifurcation] ──> [Delaunay Invariants] ──> [Gradio สคริปต์เดี่ยว]

สิ่งที่ทำสำเร็จจริง (Delivered Reality):
[U-Net v3 + TTA] ──> [Hough + FAZ Localization] ──> [3-Stage Kabsch SVD Matcher] ──> [FastAPI 2.0 REST] ──> [Next.js 16 WebGL 3D Studio]
       │                         │                               │                         │                            │
  (0.7848 Dice)           (99.03% OD Success)            (Judge Torture 63.4%)         (29 Pytest Passed)            (3 Connected Stages)
```

### Phase 1: Environment, Dataset Pipeline & Clean Architecture (Completed)
- วางโครงสร้างโปรเจกต์แบบ Clean Architecture แยก Concerns ชัดเจน (`classical_pipeline`, `dl_pipeline`, `matching`, `api`, `web`)
- ระบบ Dataset Loader รองรับ DRIVE (40 ภาพ), STARE (20 ภาพ), HRF (45 ภาพความละเอียดสูง) พร้อม Hash-based Pairing
- ท่อส่งข้อมูล Data Augmentation ด้วย Albumentations แบบ Dual-Stream (Spatial + Photometric)

### Phase 2: Deep Learning Vessel Segmentation Backbone (Completed)
- พัฒนาโมเดล U-Net ResNet-34 พร้อมโมดูล CLAHE Preprocessing
- พัฒนาระบบ Loss แบบผสมผสาน Binary Cross Entropy และ Soft Dice Loss
- ผลลัพธ์บน DRIVE Test Set: **Dice = 0.7848, AUC = 0.9652, Accuracy = 96.14%, Specificity = 98.66%**

### Phase 3: Morphological Post-Processing & Adaptive Thresholding (Completed)
- พัฒนา `dl_pipeline/postprocess.py`: กำจัดเกาะสัญญาณรบกวนนอกแนวหลอดเลือด และเติมเต็มช่องว่างภายในแนวเส้นเลือด
- พัฒนา **Quantile-Calibrated Adaptive Probability Thresholding**: ปรับค่า Threshold ตามเปอร์เซ็นไทล์ที่ 95 ของ FOV ป้องกันปัญหาเส้นเลือดขาดหายจากต้อกระจกและภาพมัว

### Phase 4: Classical Anatomical Landmark Localization Benchmark (Completed)
- บูรณาการอัลกอริทึมตรวจหา Optic Disc และ Fovea Centralis จาก `classical_pipeline/`
- ประเมินผลบนชุดทดสอบมาตรฐานสากล **IDRiD Testing Set (103 ภาพความละเอียดสูง 4288x2848)**:
  - **Optic Disc Localization Success Rate**: **99.03%** (Mean Error: 68.18 px หรือ 1.59% ของความกว้างภาพ)
  - **Fovea Centralis Localization Success Rate**: **94.17%** (Mean Error: 108.11 px หรือ 2.52% ของความกว้างภาพ)

### Phase 5: Biometric Keypoint Extraction & Skeletonization (Completed)
- พัฒนา `matching/skeleton.py` ด้วย Zhang-Suen Medial Axis Thinning
- พัฒนา `matching/bifurcation.py` ด้วย Rutovitz Crossing Number สกัดจุด Bifurcation และ Crossover พร้อม Spur Pruning กรองกิ่งสั้นหลอก (< 15px)

### Phase 6: 3-Stage Kabsch SE(2) SVD & Induced Delaunay Matcher (Completed — Beyond Scope)
- ค้นพบและแก้ไขปัญหา Root Cause ของ Delaunay Triangulation แบบเดิม: เมื่อจุดปลายหลอดเลือดหายไป Delaunay เดิมจะ Retriangulate ข้ามรูโหว่ ทำให้มุมสามเหลี่ยมเปลี่ยนและยืนยันตัวตนล้มเหลว
- พัฒนา **3-Stage Kabsch SE(2) SVD Constellation Consensus + Induced Common-Subgraph Delaunay Triangulation** ใน `matching/matcher.py`:
  - Stage 1: คัดเลือกคู่สามเหลี่ยมตั้งต้น (Seed Pairs) ด้วย Hungarian Bipartite Matching
  - Stage 2: คำนวณ Star-Vector Consensus และ SVD Rotation/Translation แบบ Closed-form
  - Stage 3: สร้างสามเหลี่ยม Delaunay เฉพาะบนกลุ่มจุด Inliers ที่จับคู่กันได้สมบูรณ์ (Induced Common Subgraph)
- ผลลัพธ์บน DRIVE Biometric Benchmark (210 คู่):
  - **Equal Error Rate (EER)**: **4.34%**
  - **Verification Accuracy @ EER**: **96.19%**
  - **Optimal Threshold**: **0.200**

### Phase 7: 11-Parameter Perturbation Suite & Stress Resilience (Completed — Beyond Scope)
- พัฒนาเอนจินจำลองความท้าทายทางคลินิก 11 ตัวแปรใน `api/services.py`
- ทดสอบความทนทานผ่าน 8 เวกเตอร์ความเครียด:
  - Clean Baseline: **100.0%**
  - Head Tilt (+25°) & Zoom (0.85x): **80.8%**
  - Cataract Haze (45%): **86.4%**
  - Corneal Glare (65%): **86.7%**
  - Defocus Blur (2.5px): **82.7%**
  - ISO Noise (16.0): **72.7%**
  - Eyelid Occlusion (22%): **86.4%**
  - **Judge Torture (7 พารามิเตอร์พร้อมกัน)**: **63.4% (PASS)**
  - ทุกคู่ Impostor ได้ **0.0% (REJECT)** ไร้ข้อผิดพลาด False Acceptance แม้แต่กรณีเดียว

### Phase 8: Production FastAPI REST Backend (`api/`) (Completed — Beyond Scope)
- พัฒนา Service Layer แยกขาดจาก Core Logic
- กำหนด Schema ด้วย Pydantic V2 พร้อมตรวจสอบ Validation Bounds
- ระบบความปลอดภัย: ป้องกัน Decompression Bomb (จำกัดขนาดภาพไม่เกิน 4096x4096) และปรับ CORS เป็น `allow_credentials=False` ตามมาตรฐาน Wildcard
- Endpoints ครบวงจร: `/api/v1/health`, `/presets`, `/segment`, `/localize`, `/match`

### Phase 9: 3-Stage Interactive WebGL Clinical Studio (`web/`) (Completed — Beyond Scope)
- พัฒนา Next.js 16 (App Router + Turbopack) ผสาน Three.js และ Custom GLSL Shaders:
  - **Stage 01 (3D Ocular Anatomy)**: ลูกตาจำลอง 3 มิติพร้อม Shader ดึงความลึกของเส้นเลือดตามผล U-Net, ปรับ Threshold แบบเรียลไทม์ 60FPS, โหมดระเบิดเลเยอร์ (Exploded Layers) และหมุดเรืองแสงชี้ตำแหน่ง Optic Disc กับ Fovea
  - **Stage 02 (1-to-1 Biometric Matcher)**: โมเดลดวงตาคู่เปรียบเทียบสองข้าง, ลำแสงเลเซอร์ Hungarian สดเชื่อมโยงจุดที่จับคู่ได้, เส้นขอบสามเหลี่ยมสีเขียวมรกต และแผงสไลเดอร์ 11 พารามิเตอร์
  - **Stage 03 (Population Cohort Matrix)**: ตาราง Confusion Matrix 4x4 เปรียบเทียบ 4 บุคคล (S01 ปกติ, S02 ตัวปลอม, S23 หลอดเลือดเบาบาง, S24 หลอดเลือดหนาแน่น), สลับโหมด Baseline vs Judge Torture, กราฟแท่งความทนทาน 8 เวกเตอร์, ปุ่ม One-Click Drill-Down กลับสู่ Stage 02 และปุ่มทดสอบสดครบ 16 ช่อง

### Phase 10: Automated Quality Gate & Repository Hygiene (Completed — Beyond Scope)
- ชุดทดสอบอัตโนมัติ `pytest tests/`: ผ่านครบ **29/29 ข้อ (100% Passed)** ใน 15.83 วินาที
- ตรวจสอบ ESLint: ผ่าน **0 errors, 0 warnings**
- ตรวจสอบ Type Safety: ผ่าน `npx tsc --noEmit` **0 errors**
- คอมไพล์โปรดักชัน: `npm run build` ผ่านสมบูรณ์ใน 1.2 วินาที
- ลบสคริปต์เก่าและโมเดลที่ไม่ใช้งาน (`demo/`, `notebooks/` เก่า, `best_unet_model.pth` v1/v2) คืนพื้นที่ ~186.8 MB

---

## 3. Capstone II Future Directions (แผนงานเทอมถัดไป)

จากรากฐานที่พัฒนาเกินเป้าหมายใน Capstone I ทำให้ Capstone II สามารถมุ่งเน้นการวิจัยเชิงลึกและนวัตกรรมระดับสูงได้อย่างเต็มศักยภาพ:

1. **Graph Neural Network (GNN) & Deep Graph Descriptors**:
   - พัฒนา Topological Graph Neural Network (เช่น Graph Attention Network หรือ Graph Transformer) เพื่อเรียนรู้ Local Subgraph Embedding บนโครงสร้างเส้นเลือด แทนการพึ่งพา Handcrafted Invariant Ratios เพียงอย่างเดียว
2. **Cross-Database Domain Adaptation & Generalization**:
   - ทดสอบและปรับจูนโมเดลข้ามฐานข้อมูลประชากรหลากหลายเชื้อชาติ (เช่น CHASE_DB1, ARIA, Messidor) เพื่อศึกษาผลกระทบของสีผิวและระดับความผิดปกติของจอประสาทตาต่อความแม่นยำของระบบชีวมิติ
3. **Hardware Deployment & Embedded Edge Acceleration**:
   - แปลงโมเดลเป็นรูปแบบ TensorRT / ONNX Runtime
   - เชื่อมต่อกับกล้องถ่ายภาพจอประสาทตาแบบพกพา (Handheld Fundus Camera) เพื่อประมวลผลบนบอร์ดประมวลผลขอบ (NVIDIA Jetson หรืออุปกรณ์มือถือ) แบบ Real-time
4. **Multi-Modal Ocular Biometrics Fusion**:
   - ผสานการยืนยันตัวตนระหว่างลวดลายหลอดเลือดจอประสาทตา (Retinal Vasculature) ร่วมกับลักษณะทางกายภาพของม่านตา (Iris) หรือลวดลายโครงสร้างชั้นประสาทตา (OCT En-face) เพื่อสร้างระบบ Multi-factor Ophthalmic Authentication ที่มีความปลอดภัยสูงสุด
