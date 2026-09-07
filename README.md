# Retinal Vascular Identification — ระบบยืนยันตัวบุคคลจากลวดลายหลอดเลือดจอตา

โครงงานระดับปริญญานิพนธ์ (Senior Capstone Project): ระบบยืนยันตัวบุคคลจากชีวมิติลวดลายหลอดเลือดบนจอประสาทตา (Retinal Vascular Pattern Biometrics)

- **อาจารย์ที่ปรึกษา**: อ. อิสูรย์ กาญจนสุรัตน์ (Isoon Kanjanasurat), ดร. นิจจารีย์ สัตยารักษ์ (Nitjaree Satayarak)
- **คณะผู้พัฒนา**:
  - ดวิษ (Classical Image Processing Pipeline & Keypoint Matching)
  - วิวรรธน์ (Deep Learning Pipeline: Vessel Segmentation & Feature Extraction)
- **อ้างอิงและต่อยอดจาก**: Kanjanasurat et al. (2020) — *Delaunay Triangle + Optic Disc Retinal Vascular Pattern*

---

## 1. ภาพรวมโปรเจกต์ (Project Overview)

ระบบนี้มีเป้าหมายในการระบุและยืนยันตัวบุคคลด้วยลักษณะเฉพาะของหลอดเลือดบนจอตา (Retinal Blood Vessels) โดยผสมผสานทั้งแนวทาง Classical Computer Vision และ Deep Learning (U-Net with Pretrained ResNet-34 Encoder) เพื่อ:
1. ทำการแบ่งส่วนหลอดเลือด (Vessel Segmentation) จากภาพถ่ายจอประสาทตา (Fundus Images)
2. หาตำแหน่งจุดอ้างอิงสำคัญ ได้แก่ จานประสาทตา (Optic Disk - OD) และจุดรับภาพชัด (Fovea)
3. สกัดจุดแยกและจุดเชื่อมต่อของหลอดเลือด (Bifurcations & Crossover Points)
4. สร้าง Topological Graph ด้วย Delaunay Triangulation เพื่อใช้จับคู่และยืนยันตัวตน (Authentication / Identification)

---

## 2. โครงสร้างโปรเจกต์ (Repository Structure)

```text
retinal-vascular/
├── classical_pipeline/         # งานฝั่ง Classical Image Processing (ดวิษ)
│   ├── __init__.py
│   └── localization.py         # ฟังก์ชันระบุพิกัด Optic Disk และ Fovea
├── dl_pipeline/                # งานฝั่ง Deep Learning Pipeline (วิวรรธน์)
│   ├── __init__.py
│   ├── data_utils.py           # Helper สำหรับ scan/parse file path ของ DRIVE, STARE, HRF
│   ├── dataset.py              # PyTorch Dataset รองรับ Augmentation & Multi-dataset
│   ├── model.py                # โมเดล Segmentation (U-Net + ResNet-34)
│   ├── train.py                # Training script (BCE + Dice Loss)
│   └── predict.py              # Inference script สกัด vessel map
├── evaluation/                 # การประเมินผลและการวัดประสิทธิภาพ
│   ├── __init__.py
│   └── metrics.py              # Pixel metrics (Acc, Sens, Spec) & Localization Euclidean distance
├── notebooks/                  # สคริปต์ทดสอบและ Sanity check
│   ├── 00_sanity_check.py      # ตรวจสอบการโหลดภาพและแสดง Green channel
│   └── 01_data_check.py        # ตรวจสอบ DataLoader, Dataset และ Data Augmentation
├── data/                       # จัดเก็บ Dataset (DRIVE, STARE, HRF) — Gitignored
├── ROADMAP.md                  # แผนการดำเนินงาน 7 สัปดาห์
├── requirements.txt            # รายการ Dependencies
└── README.md
```

---

## 3. โครงสร้างชุดข้อมูล (Dataset Directory Structure)

สร้างโฟลเดอร์ `data/` ไว้ที่ root directory ตามรูปแบบดังนี้:

```text
data/
├── DRIVE/
│   ├── training/
│   │   ├── images/          # เช่น 21_training.tif
│   │   └── 1st_manual/      # เช่น 21_manual1.gif
│   └── test/
│       ├── images/          # เช่น 01_test.tif
│       └── 1st_manual/      # เช่น 01_manual1.gif
├── STARE/
│   ├── images/              # เช่น im0001.ppm
│   └── labels-vk/           # หรือ 1st_manual/ เช่น im0001.vk.ppm
└── HRF/
    ├── images/              # เช่น 01_dr.jpg, 01_g.jpg, 01_h.jpg
    └── manual1/             # เช่น 01_dr.tif, 01_g.tif, 01_h.tif
```

---

## 4. การติดตั้งและเตรียมสภาพแวดล้อม (Installation & Setup)

### 4.1 แนะนำการใช้ Virtual Environment

```bash
# Clone repository
git clone https://github.com/swnst/retinal-vascular.git
cd retinal-vascular

# สร้างและเปิดใช้งาน Virtual Environment
python -m venv .venv

# บน Windows PowerShell
.venv\Scripts\Activate.ps1

# บน Linux / macOS
source .venv/bin/activate
```

### 4.2 ติดตั้ง Dependencies

```bash
# ติดตั้ง Library สำหรับ Classical Pipeline และ Evaluation
pip install -r requirements.txt

# สำหรับงาน Deep Learning (PyTorch + Albumentations + SMP)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install albumentations segmentation-models-pytorch
```

---

## 5. วิธีการรันและตรวจสอบระบบ (How to Run)

### 5.1 ตรวจสอบเบื้องต้น (Sanity Check - Week 1)
ตรวจสอบการอ่านไฟล์ภาพจาก dataset DRIVE และแสดงผลแยก Green channel:
```bash
python notebooks/00_sanity_check.py
```
ผลลัพธ์จะถูกบันทึกเป็นรูปภาพ `notebooks/sanity_check_output.png`

### 5.2 ตรวจสอบ Data Pipeline & Augmentation (Week 2)
ตรวจสอบการทำงานของ PyTorch Dataset, Data Split (Train/Val/Test) และ Albumentations:
```bash
python notebooks/01_data_check.py
```
ผลลัพธ์ตัวอย่าง 4 คู่ (Image + Mask) หลังผ่าน Augmentation จะถูกบันทึกเป็น `notebooks/data_check_output.png`

---

## 6. ข้อตกลง Interface ระหว่างสองสายงาน (Interface Agreement)

ฟังก์ชันเชื่อมโยงระหว่างโมเดล Deep Learning (สกัด Vessel) กับ Classical Pipeline (หาจุดสำคัญ):

```python
def find_optic_disk(vessel_map: np.ndarray, green_image: np.ndarray) -> tuple[int, int]:
    # vessel_map: binary 2D array ขนาดเท่าภาพต้นฉบับ (0/1, 1=vessel)
    # green_image: green channel ขนาดเดียวกัน
    # return: (x, y) พิกัดจุดศูนย์กลาง optic disk

def find_fovea(vessel_map: np.ndarray, optic_disk_xy: tuple[int, int]) -> tuple[int, int]:
    # vessel_map: binary 2D array ขนาดเท่าภาพต้นฉบับ (0/1, 1=vessel)
    # optic_disk_xy: (x, y) พิกัดจุดศูนย์กลาง optic disk
    # return: (x, y) พิกัดจุดศูนย์กลาง fovea
```
