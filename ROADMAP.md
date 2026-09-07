# Retinal Vascular Identification — Project Roadmap

## โปรเจกต์จบ: ระบบยืนยันตัวบุคคลจากลวดลายหลอดเลือดจอตา
- อาจารย์ที่ปรึกษา: Isoon Kanjanasurat, Nitjaree Satayarak
- ทีม: ดวิษ (Classical Pipeline) + วิวรรธน์ (Deep Learning Pipeline)
- ต่อยอดจาก: Kanjanasurat et al. (2020) — Delaunay Triangle + Optic Disc Retinal Vascular Pattern

## Technical Decisions (เคาะแล้ว)
- **Framework**: PyTorch + `segmentation_models_pytorch` (smp)
- **Architecture**: U-Net with Pretrained ResNet-34 Encoder
- **Image sizing**: Resize 512×512 (ไม่ใช้ Patch-based ในรอบแรก)
- **Loss function**: BCE + Dice Loss (Combined)
- **Training hardware**: Google Colab / Kaggle (Free GPU)
- **Datasets**: DRIVE + STARE + HRF (เทรน/ประเมิน vessel), HRF (ประเมิน OD/Fovea)

## Interface Agreement (ใช้ร่วมกัน 2 สาย)
```python
# ดวิษเขียน, วิวรรธน์เอาไปใช้
def find_optic_disk(vessel_map: np.ndarray, green_image: np.ndarray) -> tuple[int, int]:
    # vessel_map: binary 2D array ขนาดเท่าภาพต้นฉบับ (0/1, 1=vessel)
    # green_image: green channel ขนาดเดียวกัน
    # return: (x, y) พิกัดจุดศูนย์กลาง optic disk

def find_fovea(vessel_map: np.ndarray, optic_disk_xy: tuple[int, int]) -> tuple[int, int]:
    # return: (x, y) พิกัดจุดศูนย์กลาง fovea
```
**สำคัญ**: vessel_map ที่ส่งเข้าฟังก์ชันต้อง resize กลับเป็นขนาดต้นฉบับก่อน

## 7-Week Roadmap (ฝั่ง DL — วิวรรธน์)

### Week 1: Project Setup
- สร้าง repo structure, .gitignore, requirements.txt
- สร้าง interface stubs (find_optic_disk, find_fovea)
- สร้าง evaluation/metrics.py (Accuracy, Sensitivity, Specificity, Euclidean distance)
- สร้าง sanity check script โหลดภาพ DRIVE แสดงผล

### Week 2: Data Pipeline & Preprocessing
- เขียน dl_pipeline/dataset.py — PyTorch Dataset class
- รวม DRIVE + STARE + HRF, normalize 0-1, resize 512×512
- แบ่ง train/val/test (แบ่งระดับภาพก่อน ไม่ใช่ระดับ pixel)
- Data augmentation: HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast

### Week 3: Model & Training Pipeline
- เขียน dl_pipeline/model.py — smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', ...)
- เขียน dl_pipeline/train.py — training loop, BCE+Dice loss, Adam optimizer
- เทรนรอบแรกบน Colab, เช็คว่าไม่ predict ภาพดำล้วน

### Week 4: Tuning & Vessel Segmentation Evaluation
- จูน lr, batch_size, epochs, threshold
- คำนวณ Accuracy/Sensitivity/Specificity บน test set เทียบ ground truth
- ปิด Milestone: Vessel Extraction (DL)

### Week 5: Optic Disk Localization (DL Front-end)
- รับฟังก์ชัน find_optic_disk() จากดวิษ
- ป้อน vessel_map (resize กลับขนาดจริง) + green_image เข้าไป
- คำนวณ Euclidean distance เทียบ ground truth HRF

### Week 6: Fovea Localization (DL Front-end)
- รับฟังก์ชัน find_fovea() จากดวิษ
- ป้อน vessel_map + optic_disk_xy ที่ได้จาก Week 5
- คำนวณ Euclidean distance เทียบ ground truth HRF

### Week 7: Comparison & Report
- ตาราง Classical vs DL (vessel accuracy, OD success rate, fovea success rate)
- วิเคราะห์จุดที่ DL ดี/ไม่ดีกว่า Classical
- เขียนรายงานความก้าวหน้า Capstone I
