import sys
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- Project Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classical_pipeline.localization_api import find_fovea, find_optic_disk
from dl_pipeline.postprocess import refine_vessel_mask
from dl_pipeline.predict import load_model_checkpoint, predict_vessel_mask
from matching.bifurcation import detect_bifurcations
from matching.matcher import extract_identity_template, match_templates, verify_identity
from matching.skeleton import skeletonize_vessel
from matching.triangulation import build_delaunay_triangulation

# --- Global Model Initialization ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_ROOT / "models" / "best_unet_model_v2.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "models" / "best_unet_model.pth"

print(f"Loading model checkpoint for demo from: {MODEL_PATH} on {DEVICE}")
MODEL = load_model_checkpoint(str(MODEL_PATH), device=DEVICE)


# --- Tab 1: Vessel Segmentation Inference ---
def process_vessel_segmentation(image: np.ndarray, threshold: float, use_postprocess: bool) -> tuple[np.ndarray, np.ndarray]:
    if image is None:
        return None, None

    # Predict probability and raw mask
    binary_mask, prob_map = predict_vessel_mask(
        MODEL, image, device=DEVICE, threshold=threshold, use_tta=True
    )

    if use_postprocess:
        binary_mask = refine_vessel_mask(binary_mask)

    # Visualization: Overlay red vessels on fundus
    overlay = image.copy()
    overlay[binary_mask > 0] = [255, 30, 30]

    # Convert probability map to colormap
    prob_visual = (prob_map * 255).astype(np.uint8)
    prob_colored = cv2.applyColorMap(prob_visual, cv2.COLORMAP_JET)
    prob_colored = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)

    return overlay, prob_colored


# --- Tab 2: Anatomical Landmarks (Optic Disc & Fovea) ---
def process_landmark_localization(image: np.ndarray) -> tuple[np.ndarray, str]:
    if image is None:
        return None, "กรุณาอัปโหลดภาพก่อน"

    # 1. Segment vessels
    vessel_mask, _ = predict_vessel_mask(MODEL, image, device=DEVICE, threshold=0.60, use_tta=True)
    vessel_mask = refine_vessel_mask(vessel_mask)

    # 2. Extract green channel
    green_channel = image[:, :, 1]

    # 3. Locate Optic Disc and Fovea
    od_xy = find_optic_disk(vessel_mask, green_channel)
    fovea_xy = find_fovea(vessel_mask, od_xy, green_channel)

    # Visualization
    annotated = image.copy()
    h, w, _ = annotated.shape
    radius = max(10, int(w * 0.025))

    # Optic Disc (Green circle + Red cross)
    cv2.circle(annotated, (int(od_xy[0]), int(od_xy[1])), radius, (0, 255, 0), 4)
    cv2.drawMarker(annotated, (int(od_xy[0]), int(od_xy[1])), (255, 0, 0), cv2.MARKER_CROSS, 24, 4)

    # Fovea (Yellow circle + Blue cross)
    cv2.circle(annotated, (int(fovea_xy[0]), int(fovea_xy[1])), radius, (255, 255, 0), 4)
    cv2.drawMarker(annotated, (int(fovea_xy[0]), int(fovea_xy[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 4)

    info_text = f"Optic Disc Center (X, Y): ({od_xy[0]:.1f}, {od_xy[1]:.1f})\nFovea Center (X, Y): ({fovea_xy[0]:.1f}, {fovea_xy[1]:.1f})"
    return annotated, info_text


# --- Tab 3: Biometric Identity Matching ---
def process_biometric_matching(image_a: np.ndarray, image_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if image_a is None or image_b is None:
        return None, None, "กรุณาอัปโหลดภาพทั้ง 2 ฝั่งเพื่อทำการจับคู่"

    # Process Image A
    vessel_a, _ = predict_vessel_mask(MODEL, image_a, device=DEVICE, threshold=0.60, use_tta=True)
    vessel_a = refine_vessel_mask(vessel_a)
    od_a = find_optic_disk(vessel_a, image_a[:, :, 1])
    fovea_a = find_fovea(vessel_a, od_a, image_a[:, :, 1])
    tmpl_a = extract_identity_template(vessel_a, od_center=od_a, fovea_center=fovea_a)

    # Process Image B
    vessel_b, _ = predict_vessel_mask(MODEL, image_b, device=DEVICE, threshold=0.60, use_tta=True)
    vessel_b = refine_vessel_mask(vessel_b)
    od_b = find_optic_disk(vessel_b, image_b[:, :, 1])
    fovea_b = find_fovea(vessel_b, od_b, image_b[:, :, 1])
    tmpl_b = extract_identity_template(vessel_b, od_center=od_b, fovea_center=fovea_b)

    # Compute similarity score
    score = match_templates(tmpl_a, tmpl_b)
    is_match = verify_identity(score, threshold=0.20)

    # Render Delaunay graphs
    def draw_delaunay_overlay(rgb: np.ndarray, template: dict[str, Any]) -> np.ndarray:
        vis = rgb.copy()
        pts = template["keypoints"]
        tri, pts_arr = build_delaunay_triangulation(pts)
        if tri is not None:
            for simplex in tri.simplices:
                p1 = (int(pts_arr[simplex[0]][0]), int(pts_arr[simplex[0]][1]))
                p2 = (int(pts_arr[simplex[1]][0]), int(pts_arr[simplex[1]][1]))
                p3 = (int(pts_arr[simplex[2]][0]), int(pts_arr[simplex[2]][1]))
                cv2.line(vis, p1, p2, (0, 255, 255), 2)
                cv2.line(vis, p2, p3, (0, 255, 255), 2)
                cv2.line(vis, p3, p1, (0, 255, 255), 2)
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
        return vis

    graph_a = draw_delaunay_overlay(image_a, tmpl_a)
    graph_b = draw_delaunay_overlay(image_b, tmpl_b)

    result_status = "ยืนยันตัวตนสำเร็จ (MATCHED - บุคคลเดียวกัน)" if is_match else "ยืนยันตัวตนไม่ผ่าน (NOT MATCHED - ต่างบุคคล)"
    summary_text = (
        f"ผลการตรวจสอบ: {result_status}\n"
        f"Similarity Score: {score:.4f} (Optimal Threshold: 0.2000)\n"
        f"Image A - จุด Bifurcation: {tmpl_a['num_keypoints']}, สามเหลี่ยม Delaunay: {tmpl_a['delaunay_triangles']}\n"
        f"Image B - จุด Bifurcation: {tmpl_b['num_keypoints']}, สามเหลี่ยม Delaunay: {tmpl_b['delaunay_triangles']}"
    )

    return graph_a, graph_b, summary_text


# --- Build Gradio Interface ---
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Retinal Vascular Biometric Authentication") as app:
        gr.Markdown(
            "# ระบบยืนยันตัวบุคคลจากลวดลายหลอดเลือดจอประสาทตา (Retinal Biometrics)\n"
            "โครงงานปริญญานิพนธ์: ต่อยอดจากงานวิจัย Kanjanasurat et al. (2020) ด้วย Deep Learning U-Net ResNet-34"
        )

        with gr.Tabs():
            # Tab 1: Vessel Segmentation
            with gr.TabItem("1. Vessel Segmentation"):
                gr.Markdown("### แบ่งส่วนหลอดเลือดจอตาด้วยโมเดล Deep Learning (U-Net ResNet-34)")
                with gr.Row():
                    input_img = gr.Image(type="numpy", label="ภาพถ่ายจอประสาทตา (Fundus Image)")
                    with gr.Column():
                        th_slider = gr.Slider(0.1, 0.9, value=0.60, step=0.05, label="Probability Threshold")
                        post_check = gr.Checkbox(value=True, label="เปิดใช้ Morphological Post-processing (ลบ Noise ขยะ)")
                        seg_btn = gr.Button("ประมวลผลแบ่งส่วนหลอดเลือด", variant="primary")

                with gr.Row():
                    out_overlay = gr.Image(type="numpy", label="Vessel Segmentation Overlay")
                    out_prob = gr.Image(type="numpy", label="Probability Heatmap")

                seg_btn.click(
                    process_vessel_segmentation,
                    inputs=[input_img, th_slider, post_check],
                    outputs=[out_overlay, out_prob],
                )

            # Tab 2: Landmarks
            with gr.TabItem("2. Anatomical Landmarks"):
                gr.Markdown("### ค้นหาจุดศูนย์กลางจานประสาทตา (Optic Disc) และจุดรับภาพชัด (Fovea)")
                with gr.Row():
                    land_input = gr.Image(type="numpy", label="ภาพถ่ายจอประสาทตา (Fundus Image)")
                    with gr.Column():
                        land_btn = gr.Button("ค้นหาจุดพิกัดทางกายวิภาค", variant="primary")
                        land_text = gr.Textbox(label="พิกัดที่ตรวจพบ (Coordinates)", lines=3)

                out_landmarks = gr.Image(type="numpy", label="ผลการระบุตำแหน่ง (เขียว/แดง: OD, เหลือง/น้ำเงิน: Fovea)")

                land_btn.click(
                    process_landmark_localization,
                    inputs=[land_input],
                    outputs=[out_landmarks, land_text],
                )

            # Tab 3: Matching
            with gr.TabItem("3. Biometric Identity Matching"):
                gr.Markdown("### ตรวจสอบและยืนยันตัวตนด้วยโครงข่ายสามเหลี่ยม Delaunay และ Geometric Invariants")
                with gr.Row():
                    in_a = gr.Image(type="numpy", label="ภาพอ้างอิง A (Reference Image)")
                    in_b = gr.Image(type="numpy", label="ภาพตรวจสอบ B (Inquiry Image)")

                match_btn = gr.Button("ตรวจสอบและจับคู่ลวดลายชีวมิติ", variant="primary")

                with gr.Row():
                    out_graph_a = gr.Image(type="numpy", label="Delaunay Triangulation Graph A")
                    out_graph_b = gr.Image(type="numpy", label="Delaunay Triangulation Graph B")

                match_result = gr.Textbox(label="ผลการตรวจสอบและค่าสถิติความคล้ายคลึง", lines=4)

                match_btn.click(
                    process_biometric_matching,
                    inputs=[in_a, in_b],
                    outputs=[out_graph_a, out_graph_b, match_result],
                )

                # Pre-prepared clickable examples for immediate testing
                examples_dir = PROJECT_ROOT / "demo" / "examples"
                if (examples_dir / "subject1_original.png").exists():
                    gr.Markdown("#### ตัวอย่างภาพสำเร็จรูปสำหรับทดสอบคลิกเดียว:")
                    gr.Examples(
                        examples=[
                            [
                                str(examples_dir / "subject1_original.png"),
                                str(examples_dir / "subject1_tilted_genuine.png"),
                            ],
                            [
                                str(examples_dir / "subject1_original.png"),
                                str(examples_dir / "subject2_impostor.png"),
                            ],
                        ],
                        inputs=[in_a, in_b],
                        outputs=[out_graph_a, out_graph_b, match_result],
                        fn=process_biometric_matching,
                        label="คลิกเลือกชุดทดสอบ: คู่ที่ 1 (บุคคลเดียวกัน - ภาพเอียง 5 องศา) หรือ คู่ที่ 2 (คนละคน - ผู้แอบอ้าง)",
                    )

        gr.Markdown("--- \n**Retinal Vascular Biometrics Capstone I Research Engine**")

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
