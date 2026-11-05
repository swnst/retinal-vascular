import json
from pathlib import Path
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# --- Configuration & Figure Mapping ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TEMP_DIR = PROJECT_ROOT / "scratch" / "thai_figures_staging"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FOLDER_02_ID = "1UteJoEiB87p3LNsKI6eZKzJglCnmKRRB"

FIGURE_SPECS = [
    ("Figure_2.1_Retinal_Anatomy.png", "รูปที่ 2.1 - กายวิภาคจอประสาทตาและพิกัดอ้างอิง.png"),
    ("Figure_2.2_Hemoglobin_Spectrum.png", "รูปที่ 2.2 - สเปกตรัมการดูดกลืนแสงของฮีโมโกลบิน.png"),
    ("Figure_2.3_Rutovitz_Crossing_Number.png", "รูปที่ 2.3 - ทฤษฎี Rutovitz Crossing Number.png"),
    ("Figure_2.4_Kabsch_SVD_Alignment.png", "รูปที่ 2.4 - การจัดเรียงพิกัดด้วย Kabsch SVD.png"),
    ("Figure_2.5_Delaunay_Triangulation.png", "รูปที่ 2.5 - โครงข่ายสามเหลี่ยมเดอลอนีย์และค่าคงที่.png"),
    ("Figure_3.1_System_Architecture.png", "รูปที่ 3.1 - แผนภาพสถาปัตยกรรมระบบ VERA.png"),
    ("Figure_3.2_Preprocessing_Pipeline.png", "รูปที่ 3.2 - ขั้นตอนการเตรียมภาพเบื้องต้น.png"),
    ("Figure_3.3_UNet_ResNet34_Loss.png", "รูปที่ 3.3 - แผนผังสถาปัตยกรรม ResNet-34 U-Net.png"),
    ("Figure_3.4_Feature_Extraction_Pipeline.png", "รูปที่ 3.4 - ท่อส่งข้อมูลการสกัดฟีเจอร์และการจับคู่.png"),
    ("Figure_3.5_Judge_Torture_Vectors.png", "รูปที่ 3.5 - การจำลองเวกเตอร์การเสื่อมสภาพ 8 รูปแบบ.png"),
    ("Figure_3.6_3D_Cup_Constellation.png", "รูปที่ 3.6 - การเรนเดอร์แอ่งจอตาเว้าสามมิติและ WebGL.png"),
    ("Figure_5.1_Training_Curves.png", "รูปที่ 5.1 - กราฟการฝึกโมเดล ResNet-34 U-Net.png"),
    ("Figure_5.2_Vessel_Segmentation_Samples.png", "รูปที่ 5.2 - ตัวอย่างผลการแบ่งส่วนภาพหลอดเลือด.png"),
    ("Figure_5.3_Threshold_Optimization.png", "รูปที่ 5.3 - กราฟวิเคราะห์ค่าขีดเริ่มเปลี่ยนความน่าจะเป็น.png"),
    ("Figure_5.4_Biometric_Matching_Curves.png", "รูปที่ 5.4 - เส้นโค้งสมรรถนะการพิสูจน์ตัวตนชีวมิติ.png"),
    ("Figure_5.5_IDRiD_001_Localization.png", "รูปที่ 5.5 - การตรวจจับขั้วประสาทตาภาพ IDRiD 001.png"),
    ("Figure_5.6_IDRiD_002_Localization.png", "รูปที่ 5.6 - การตรวจจับขั้วประสาทตาภาพ IDRiD 002.png"),
]


# --- Google Drive Cleaning Operations ---
GDRIVE_BIN = shutil.which("gdrive") or "gdrive"


def clean_old_files_in_folder_02():
    cmd = [GDRIVE_BIN, "readonly", "ls", FOLDER_02_ID, "--json"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, shell=True)
    items = json.loads(res.stdout) if res.stdout.strip() else []
    print(f"Found {len(items)} existing files in folder 02. Trashing old versions...")
    for item in items:
        f_id = item["id"]
        f_name = item["name"]
        print(f"Trashing: {f_name} ({f_id})")
        subprocess.run([GDRIVE_BIN, "mutate", "trash", f_id], capture_output=True, encoding="utf-8", errors="replace", check=True, shell=True)
    print("Folder 02 cleaned successfully.")


# --- Upload and Share Figures ---
def upload_and_share_all_figures() -> dict[str, dict[str, str]]:
    uploaded_map: dict[str, dict[str, str]] = {}
    for eng_name, thai_name in FIGURE_SPECS:
        src_path = FIGURES_DIR / eng_name
        if not src_path.is_file():
            print(f"ERROR: Missing source figure {src_path}", file=sys.stderr)
            sys.exit(1)

        staged_path = TEMP_DIR / thai_name
        from PIL import Image
        im = Image.open(src_path)
        if max(im.width, im.height) > 800:
            scale = 800.0 / max(im.width, im.height)
            new_size = (int(round(im.width * scale)), int(round(im.height * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im_quant = im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        im_quant.save(staged_path, "PNG", optimize=True)

        print(f"Uploading {thai_name} ({staged_path.stat().st_size} bytes)...")
        up_cmd = [GDRIVE_BIN, "mutate", "upload", str(staged_path), "--parent", FOLDER_02_ID, "--json"]
        up_res = subprocess.run(up_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, shell=True)
        out_text = up_res.stdout.strip()

        file_id = None
        if "ID: " in out_text:
            file_id = out_text.split("ID: ")[1].split(")")[0].strip()
        elif out_text.startswith("{"):
            obj = json.loads(out_text)
            file_id = obj.get("id")

        if not file_id:
            print(f"ERROR: Could not parse file_id from output: {out_text}", file=sys.stderr)
            sys.exit(1)

        share_cmd = [GDRIVE_BIN, "mutate", "share", file_id, "--type", "anyone", "--role", "reader"]
        subprocess.run(share_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, shell=True)

        lh3_url = f"https://lh3.googleusercontent.com/d/{file_id}"
        uc_url = f"https://drive.google.com/uc?id={file_id}"

        uploaded_map[eng_name] = {
            "thai_name": thai_name,
            "file_id": file_id,
            "lh3_url": lh3_url,
            "uc_url": uc_url,
        }
        print(f"Successfully uploaded & shared: {thai_name} -> {file_id}")

    out_json = PROJECT_ROOT / "results" / "drive_uploaded_figures.json"
    out_json.write_text(json.dumps(uploaded_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved mapping for 17 figures to {out_json}")
    return uploaded_map


if __name__ == "__main__":
    clean_old_files_in_folder_02()
    upload_and_share_all_figures()
