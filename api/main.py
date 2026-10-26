import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from api.schemas import (
    LocalizationRequest,
    LocalizationResponse,
    MatchRequest,
    MatchResponse,
    NormalizeImageRequest,
    NormalizeImageResponse,
    PresetItem,
    SegmentationRequest,
    SegmentationResponse,
)
from api.services import PipelineService


# --- FastAPI Application Setup ---
app = FastAPI(
    title="VERA | Vascular Extraction & Retinal Authentication API",
    description="Production-grade AI pipeline for 3D ocular anatomy, vessel segmentation, and judge-proof biometric verification.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_service = PipelineService()


# --- Endpoint Handlers ---
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "online",
        "device": str(pipeline_service.device),
        "model_loaded": pipeline_service.model is not None,
    }


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHED_PRESETS: list[PresetItem] | None = None


@app.get("/api/v1/presets", response_model=list[PresetItem])
def get_presets():
    global _CACHED_PRESETS
    if _CACHED_PRESETS is not None:
        return _CACHED_PRESETS

    presets = []
    base_dir = PROJECT_ROOT / "data"

    drive_training_dir = base_dir / "DRIVE" / "training" / "images"
    drive_test_dir = base_dir / "DRIVE" / "test" / "images"
    drive_images: list[Path] = []
    for d in [drive_training_dir, drive_test_dir]:
        if d.exists():
            drive_images.extend(sorted(list(d.glob("*.tif")) + list(d.glob("*.png"))))

    def load_b64(path: Path | None) -> tuple[str, str]:
        if path is None or not path.exists():
            return "", ""
        img = cv2.imread(str(path))
        if img is None:
            return "", ""
        h, w = img.shape[:2]
        if max(h, w) > 1000:
            scale = 768.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        thumb = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        return (
            pipeline_service.encode_base64_image(img, ".png"),
            pipeline_service.encode_base64_image(thumb, ".jpeg"),
        )

    img_21_path = next((p for p in drive_images if "21" in p.name or "01" in p.name), None)
    img_22_path = next((p for p in drive_images if "22" in p.name or "02" in p.name), None)
    img_23_path = next((p for p in drive_images if "23" in p.name or "03" in p.name), None)
    img_24_path = next((p for p in drive_images if "24" in p.name or "04" in p.name), None)

    hrf_dir = base_dir / "HRF" / "images"
    hrf_path = None
    if hrf_dir.exists():
        hrf_candidates = list(hrf_dir.glob("*_dr.JPG")) + list(hrf_dir.glob("*_dr.jpg")) + list(hrf_dir.glob("*.jpg"))
        if hrf_candidates:
            hrf_path = hrf_candidates[0]

    b64_21, thumb_21 = load_b64(img_21_path)
    b64_22, thumb_22 = load_b64(img_22_path)
    b64_23, thumb_23 = load_b64(img_23_path)
    b64_24, thumb_24 = load_b64(img_24_path)
    b64_hrf, thumb_hrf = load_b64(hrf_path) if hrf_path else (b64_21, thumb_21)

    # --- Step 1: Ocular Anatomy & Vessel Stage Presets ---
    presets.append(PresetItem(
        id="ocular_healthy_01",
        title="Macula-Centered Baseline",
        description="Healthy fundus scan demonstrating clear Optic Disc convergence and avascular Fovea depression.",
        dataset="DRIVE",
        scenario="clinical",
        target_stage="ocular",
        badge_label="Normal Anatomy",
        image_base64=b64_21,
        image_b_base64=None,
        thumbnail_base64=thumb_21,
    ))

    presets.append(PresetItem(
        id="ocular_dense_24",
        title="High-Density Microvasculature",
        description="High-density vascular tree (12.7% vessel density) ideal for 3D shader relief extrusion and exploded skeleton layers.",
        dataset="DRIVE",
        scenario="dense_clinical",
        target_stage="ocular",
        badge_label="Rich Arcade Mesh",
        image_base64=b64_24,
        image_b_base64=None,
        thumbnail_base64=thumb_24,
    ))

    presets.append(PresetItem(
        id="ocular_sparse_23",
        title="Low-Caliber Peripheral Tree",
        description="Sparse/faint vascular network (7.8% density) for testing U-Net probability threshold sensitivity and gap filling.",
        dataset="DRIVE",
        scenario="sparse_clinical",
        target_stage="ocular",
        badge_label="Attenuated Vessels",
        image_base64=b64_23,
        image_b_base64=None,
        thumbnail_base64=thumb_23,
    ))

    presets.append(PresetItem(
        id="ocular_pathology_dr",
        title="Pathological Fundus (HRF)",
        description="Clinical fundus scan (Diabetic Retinopathy) testing landmark localization under pathological lesions and media shifts.",
        dataset="HRF",
        scenario="clinical",
        target_stage="ocular",
        badge_label="Clinical Pathology",
        image_base64=b64_hrf,
        image_b_base64=None,
        thumbnail_base64=thumb_hrf,
    ))

    # --- Step 2: Biometric Identity Comparison Stage Presets ---
    presets.append(PresetItem(
        id="match_genuine_01",
        title="Subject 01 Genuine Pair",
        description="Same retina for testing positive biometric verification across all 11 interactive perturbation controls.",
        dataset="DRIVE",
        scenario="genuine",
        target_stage="matching",
        badge_label="Same Person",
        image_base64=b64_21,
        image_b_base64=b64_21,
        thumbnail_base64=thumb_21,
    ))

    presets.append(PresetItem(
        id="match_impostor_01",
        title="Subject 01 vs Subject 02",
        description="Distinct individuals to benchmark zero false-acceptance biometric spoofing defense.",
        dataset="DRIVE",
        scenario="impostor",
        target_stage="matching",
        badge_label="Impostor Test",
        image_base64=b64_21,
        image_b_base64=b64_22,
        thumbnail_base64=thumb_22,
    ))

    presets.append(PresetItem(
        id="match_sparse_23",
        title="Subject 23 Low-Node Pair",
        description="Same subject with only 34 bifurcation nodes testing Delaunay graph stability under sparse keypoints.",
        dataset="DRIVE",
        scenario="sparse_clinical",
        target_stage="matching",
        badge_label="Sparse Graph",
        image_base64=b64_23,
        image_b_base64=b64_23,
        thumbnail_base64=thumb_23,
    ))

    presets.append(PresetItem(
        id="match_dense_24",
        title="Subject 24 High-Node Pair",
        description="Same subject with 142 Delaunay triangles testing Hungarian bipartite assignment in crowded topologies.",
        dataset="DRIVE",
        scenario="dense_clinical",
        target_stage="matching",
        badge_label="Dense Graph",
        image_base64=b64_24,
        image_b_base64=b64_24,
        thumbnail_base64=thumb_24,
    ))

    _CACHED_PRESETS = presets
    return presets


@app.post("/api/v1/segment", response_model=SegmentationResponse)
def segment_endpoint(req: SegmentationRequest):
    try:
        img_bgr = pipeline_service.decode_base64_image(req.image_base64)
        h, w = img_bgr.shape[:2]
        vessel_mask, prob_map, skeleton, density, keypoints, elapsed = pipeline_service.segment_vessels(
            bgr_image=img_bgr,
            threshold=req.threshold,
            apply_postprocess=req.apply_postprocess,
            min_size=req.min_size,
            max_hole_size=req.max_hole_size,
            use_tta=req.use_tta,
        )
        return SegmentationResponse(
            vessel_mask_base64=pipeline_service.encode_base64_image(vessel_mask, ".png"),
            prob_map_base64=pipeline_service.encode_base64_image(prob_map, ".png"),
            skeleton_base64=pipeline_service.encode_base64_image(skeleton, ".png"),
            vessel_density=density,
            keypoints=keypoints,
            num_keypoints=len(keypoints),
            elapsed_ms=elapsed,
            width=w,
            height=h,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/localize", response_model=LocalizationResponse)
def localize_endpoint(req: LocalizationRequest):
    try:
        img_bgr = pipeline_service.decode_base64_image(req.image_base64)
        h, w = img_bgr.shape[:2]
        vessel_mask = None
        if req.vessel_mask_base64:
            raw_mask = pipeline_service.decode_base64_image(req.vessel_mask_base64)
            mask_2d = raw_mask[:, :, 0] if raw_mask.ndim == 3 else raw_mask
            vessel_mask = ((mask_2d > 127).astype(np.uint8)) * 255

        od, fovea, dist, elapsed = pipeline_service.localize_landmarks(
            bgr_image=img_bgr,
            vessel_mask=vessel_mask,
        )
        return LocalizationResponse(
            optic_disc=od,
            fovea=fovea,
            disc_to_fovea_distance=dist,
            elapsed_ms=elapsed,
            width=w,
            height=h,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/match", response_model=MatchResponse)
def match_endpoint(req: MatchRequest):
    try:
        img_a = pipeline_service.decode_base64_image(req.image_a_base64)
        img_b = pipeline_service.decode_base64_image(req.image_b_base64)
        result = pipeline_service.match_identities(
            bgr_image_a=img_a,
            bgr_image_b=img_b,
            ratio_tolerance=req.ratio_tolerance,
            angle_tolerance_rad=req.angle_tolerance_rad,
            rotate_deg_b=req.rotate_deg_b,
            scale_factor_b=req.scale_factor_b,
            brightness_b=req.brightness_b,
            blur_sigma_b=req.blur_sigma_b,
            haze_factor_b=req.haze_factor_b,
            glare_intensity_b=req.glare_intensity_b,
            noise_sigma_b=req.noise_sigma_b,
            occlusion_frac_b=req.occlusion_frac_b,
            gamma_b=req.gamma_b,
            spherical_warp_b=req.spherical_warp_b,
            elastic_alpha_b=req.elastic_alpha_b,
        )
        return MatchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/upload-image", response_model=NormalizeImageResponse)
async def upload_image_endpoint(
    file: UploadFile = File(...),
    max_dim: int = Form(768),
):
    try:
        if max_dim < 128 or max_dim > 2048:
            raise ValueError("max_dim must be between 128 and 2048.")
        raw_bytes = await file.read()
        result = pipeline_service.normalize_image_for_studio(
            raw_bytes=raw_bytes,
            max_dim=max_dim,
            filename=file.filename or "uploaded_scan",
        )
        return NormalizeImageResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/normalize-image", response_model=NormalizeImageResponse)
def normalize_image_endpoint(req: NormalizeImageRequest):
    try:
        b64_payload = req.image_base64.split(",", 1)[1] if "," in req.image_base64 else req.image_base64
        raw_bytes = base64.b64decode(b64_payload)
        result = pipeline_service.normalize_image_for_studio(
            raw_bytes=raw_bytes,
            max_dim=req.max_dim,
        )
        return NormalizeImageResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
