import io
from pathlib import Path
import time
import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from api.main import app, pipeline_service
from api.services import PipelineService
from matching.matcher import match_templates_detailed
from matching.triangulation import build_delaunay_triangulation, compute_triangle_invariants


# --- API Test Suite Fixtures & Helpers ---
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_dummy_fundus_base64(w: int = 250, h: int = 250) -> str:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), int(w * 0.42), (20, 50, 180), -1)
    cv2.circle(img, (int(w * 0.35), int(h * 0.5)), int(w * 0.08), (140, 210, 240), -1)
    cv2.circle(img, (int(w * 0.65), int(h * 0.5)), int(w * 0.05), (10, 25, 90), -1)
    cv2.line(img, (int(w * 0.35), int(h * 0.5)), (int(w * 0.5), int(h * 0.2)), (80, 120, 240), 2)
    cv2.line(img, (int(w * 0.35), int(h * 0.5)), (int(w * 0.5), int(h * 0.8)), (80, 120, 240), 2)
    return pipeline_service.encode_base64_image(img, ".png")


# --- Baseline API Endpoint Tests ---
def test_api_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "device" in data
    assert data["model_loaded"] is True


def test_api_presets(client):
    res = client.get("/api/v1/presets")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    if len(data) > 0:
        item = data[0]
        assert "image_base64" in item
        assert "thumbnail_base64" in item
        assert len(item["image_base64"]) > 50


def test_api_segment(client):
    dummy_b64 = create_dummy_fundus_base64()
    payload = {
        "image_base64": dummy_b64,
        "threshold": 0.60,
        "apply_postprocess": True,
        "use_tta": False,
    }
    res = client.post("/api/v1/segment", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "vessel_mask_base64" in data
    assert "prob_map_base64" in data
    assert "skeleton_base64" in data
    assert "vessel_density" in data
    assert data["width"] == 250
    assert data["height"] == 250


def test_api_localize(client):
    dummy_b64 = create_dummy_fundus_base64()
    payload = {
        "image_base64": dummy_b64,
    }
    res = client.post("/api/v1/localize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "optic_disc" in data
    assert "fovea" in data
    assert "disc_to_fovea_distance" in data
    assert data["width"] == 250


def test_api_match(client):
    dummy_b64 = create_dummy_fundus_base64()
    payload = {
        "image_a_base64": dummy_b64,
        "image_b_base64": dummy_b64,
        "ratio_tolerance": 0.05,
        "angle_tolerance_rad": 0.08,
    }
    res = client.post("/api/v1/match", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "similarity_score" in data
    assert "is_match" in data
    assert "matched_pairs" in data
    assert "template_a" in data
    assert "template_b" in data
    assert data["template_a"]["width"] == 250
    assert "perturbed_image_b_base64" in data


def test_api_match_perturbed_image_b_returned(client):
    dummy_b64 = create_dummy_fundus_base64()
    payload = {
        "image_a_base64": dummy_b64,
        "image_b_base64": dummy_b64,
        "rotate_deg_b": 15.0,
        "scale_factor_b": 0.90,
    }
    res = client.post("/api/v1/match", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "perturbed_image_b_base64" in data
    assert data["perturbed_image_b_base64"] is not None
    assert len(data["perturbed_image_b_base64"]) > 50


def test_api_match_out_of_bounds_rejected(client):
    dummy_b64 = create_dummy_fundus_base64()
    payload = {
        "image_a_base64": dummy_b64,
        "image_b_base64": dummy_b64,
        "scale_factor_b": 5.0,
    }
    res = client.post("/api/v1/match", json=payload)
    assert res.status_code == 422


# --- Forensic E2E & Hardening Tests ---
def test_api_upload_and_normalize_all_clinical_formats(client):
    real_drive_tif = Path("data/DRIVE/test/images/01_test.tif")
    if real_drive_tif.is_file():
        tif_bytes = real_drive_tif.read_bytes()
    else:
        syn_tif = np.zeros((584, 565, 3), dtype=np.uint8)
        cv2.circle(syn_tif, (282, 292), 260, (20, 50, 180), -1)
        _, buf = cv2.imencode(".tif", syn_tif)
        tif_bytes = buf.tobytes()

    res_upload = client.post(
        "/api/v1/upload-image",
        files={"file": ("01_test.tif", tif_bytes, "image/tiff")},
    )
    assert res_upload.status_code == 200
    data_upload = res_upload.json()
    assert data_upload["image_base64"].startswith("data:image/png;base64,")
    assert data_upload["original_width"] == 565
    assert data_upload["original_height"] == 584

    ppm_bytes = b"P6\n64 64\n255\n" + np.full((64, 64, 3), 180, dtype=np.uint8).tobytes()
    res_ppm = client.post(
        "/api/v1/upload-image",
        files={"file": ("sample.ppm", ppm_bytes, "application/octet-stream")},
    )
    assert res_ppm.status_code == 200
    data_ppm = res_ppm.json()
    assert data_ppm["width"] == 64
    assert data_ppm["height"] == 64

    pgm_bytes = b"P5\n64 64\n255\n" + np.full((64, 64), 120, dtype=np.uint8).tobytes()
    res_pgm = client.post(
        "/api/v1/upload-image",
        files={"file": ("sample.pgm", pgm_bytes, "application/octet-stream")},
    )
    assert res_pgm.status_code == 200
    data_pgm = res_pgm.json()
    assert data_pgm["width"] == 64
    assert data_pgm["height"] == 64

    gif_io = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 140, dtype=np.uint8)).save(gif_io, format="GIF")
    res_gif = client.post(
        "/api/v1/upload-image",
        files={"file": ("sample.gif", gif_io.getvalue(), "image/gif")},
    )
    assert res_gif.status_code == 200

    u16_img = np.full((64, 64, 3), 32000, dtype=np.uint16)
    _, u16_buf = cv2.imencode(".tif", u16_img)
    res_u16 = client.post(
        "/api/v1/upload-image",
        files={"file": ("sample_u16.tif", u16_buf.tobytes(), "image/tiff")},
    )
    assert res_u16.status_code == 200
    data_u16 = res_u16.json()
    decoded_u16 = pipeline_service.decode_base64_image(data_u16["image_base64"])
    assert decoded_u16.dtype == np.uint8

    rgba_img = np.full((64, 64, 4), 200, dtype=np.uint8)
    rgba_img[0:16, 0:16] = [255, 255, 255, 0]
    _, rgba_buf = cv2.imencode(".png", rgba_img)
    rgba_b64 = pipeline_service.encode_base64_image(rgba_img, ".png")

    res_norm = client.post(
        "/api/v1/normalize-image",
        json={"image_base64": rgba_b64, "max_dim": 768},
    )
    assert res_norm.status_code == 200
    data_norm = res_norm.json()
    decoded_rgba = pipeline_service.decode_base64_image(data_norm["image_base64"])
    assert decoded_rgba.shape[2] == 3
    assert np.all(decoded_rgba[0, 0] == [0, 0, 0])


def test_api_high_resolution_idrid_and_hrf_uploads(client):
    high_res_h, high_res_w = 2848, 4288
    high_res_img = np.zeros((high_res_h, high_res_w, 3), dtype=np.uint8)
    cv2.circle(high_res_img, (high_res_w // 2, high_res_h // 2), 1200, (20, 50, 180), -1)
    _, high_res_jpg = cv2.imencode(".jpg", high_res_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])

    res_upload = client.post(
        "/api/v1/upload-image",
        files={"file": ("idrid_4288x2848.jpg", high_res_jpg.tobytes(), "image/jpeg")},
        data={"max_dim": "768"},
    )
    assert res_upload.status_code == 200
    data_upload = res_upload.json()
    assert data_upload["original_width"] == 4288
    assert data_upload["original_height"] == 2848
    assert data_upload["width"] == 768

    norm_b64 = data_upload["image_base64"]
    res_seg = client.post("/api/v1/segment", json={"image_base64": norm_b64, "threshold": 0.60})
    assert res_seg.status_code == 200

    res_loc = client.post("/api/v1/localize", json={"image_base64": norm_b64})
    assert res_loc.status_code == 200

    res_match = client.post(
        "/api/v1/match",
        json={"image_a_base64": norm_b64, "image_b_base64": norm_b64},
    )
    assert res_match.status_code == 200


def test_api_cross_resolution_genuine_verification(client):
    img_p = Path("data/DRIVE/test/images/01_test.tif")
    if not img_p.is_file():
        img_p = Path("data/DRIVE/training/images/21_training.tif")
    img1 = cv2.imread(str(img_p))
    b64_1 = pipeline_service.encode_base64_image(img1, ".png")

    img2 = cv2.resize(img1, (int(img1.shape[1] * 1.5), int(img1.shape[0] * 1.5)), interpolation=cv2.INTER_LINEAR)
    b64_2 = pipeline_service.encode_base64_image(img2, ".png")

    res = client.post(
        "/api/v1/match",
        json={"image_a_base64": b64_1, "image_b_base64": b64_2},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_match"] is True
    assert data["similarity_score"] > 0.50


def test_api_dark_blank_and_extreme_threshold_resilience(client):
    dark_img = np.zeros((250, 250, 3), dtype=np.uint8)
    dark_b64 = pipeline_service.encode_base64_image(dark_img, ".png")

    res_seg_dark = client.post(
        "/api/v1/segment",
        json={"image_base64": dark_b64, "threshold": 0.95},
    )
    assert res_seg_dark.status_code == 200

    res_loc_dark = client.post(
        "/api/v1/localize",
        json={"image_base64": dark_b64},
    )
    assert res_loc_dark.status_code == 200

    dummy_b64 = create_dummy_fundus_base64()
    res_match_extreme = client.post(
        "/api/v1/match",
        json={
            "image_a_base64": dummy_b64,
            "image_b_base64": dark_b64,
            "brightness_b": 0.20,
            "gamma_b": 3.0,
        },
    )
    assert res_match_extreme.status_code == 200
    data_match = res_match_extreme.json()
    assert data_match["is_match"] is False


def test_matcher_dense_random_noise_rejected():
    rng = np.random.default_rng(42)
    kps_a = rng.uniform([50.0, 50.0], [500.0, 500.0], size=(500, 2)).tolist()
    kps_b = rng.uniform([50.0, 50.0], [500.0, 500.0], size=(500, 2)).tolist()

    tri_a, pts_a = build_delaunay_triangulation(kps_a)
    tri_b, pts_b = build_delaunay_triangulation(kps_b)
    inv_a = compute_triangle_invariants(tri_a, pts_a, min_area=10.0) if tri_a is not None else []
    inv_b = compute_triangle_invariants(tri_b, pts_b, min_area=10.0) if tri_b is not None else []

    tpl_a = {
        "keypoints": kps_a,
        "invariants": inv_a,
    }
    tpl_b = {
        "keypoints": kps_b,
        "invariants": inv_b,
    }

    t0 = time.perf_counter()
    res = match_templates_detailed(tpl_a, tpl_b)
    elapsed = time.perf_counter() - t0

    assert res["score"] < 0.20
    assert elapsed < 2.0


def test_segment_threshold_monotonicity(client):
    base = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(base, (150, 150), 130, (30, 60, 180), -1)
    cv2.line(base, (100, 150), (200, 150), (80, 140, 255), 4)
    haze = np.full((300, 300, 3), 70, dtype=np.uint8)
    blended = cv2.addWeighted(base, 0.7, haze, 0.3, 0)
    b64 = pipeline_service.encode_base64_image(blended, ".png")

    thresholds = [0.62, 0.60, 0.58, 0.50]
    densities: list[float] = []

    for th in thresholds:
        res = client.post("/api/v1/segment", json={"image_base64": b64, "threshold": th})
        assert res.status_code == 200
        densities.append(res.json()["vessel_density"])

    for i in range(len(densities) - 1):
        assert densities[i] <= densities[i + 1] + 1e-5


def test_api_match_perturbation_uint8_and_noise_resilience(client):
    presets = client.get("/api/v1/presets").json()
    genuine = next(p for p in presets if p["id"] == "match_genuine_01")

    res_noise = client.post(
        "/api/v1/match",
        json={
            "image_a_base64": genuine["image_base64"],
            "image_b_base64": genuine["image_b_base64"],
            "noise_sigma_b": 16.0,
            "brightness_b": 0.85,
        },
    )
    assert res_noise.status_code == 200
    data_noise = res_noise.json()
    assert data_noise["is_match"] is True
    assert data_noise["similarity_score"] > 0.65

    res_pose = client.post(
        "/api/v1/match",
        json={
            "image_a_base64": genuine["image_base64"],
            "image_b_base64": genuine["image_b_base64"],
            "rotate_deg_b": 25.0,
            "scale_factor_b": 0.85,
        },
    )
    assert res_pose.status_code == 200
    data_pose = res_pose.json()
    assert data_pose["is_match"] is True
    assert data_pose["similarity_score"] > 0.75



# --- Additional Edge Case Tests ---
def test_predict_single_image_dtype_robustness(client):
    from dl_pipeline.predict import predict_single_image

    model = pipeline_service.model
    assert model is not None, "Model should be loaded when client fixture is initialized"
    active_device = str(pipeline_service.device)

    dummy_f32_01 = np.full((128, 128, 3), 0.5, dtype=np.float32)
    mask1, prob1 = predict_single_image(model, dummy_f32_01, device=active_device, return_original_size=True)
    assert mask1.shape == (128, 128)
    assert prob1.shape == (128, 128)
    assert prob1.max() <= 1.0 and prob1.min() >= 0.0

    dummy_overshoot = np.full((128, 128, 3), 1.02, dtype=np.float32)
    mask_os, prob_os = predict_single_image(model, dummy_overshoot, device=active_device, return_original_size=True)
    assert mask_os.shape == (128, 128)
    assert prob_os.max() <= 1.0 and prob_os.min() >= 0.0

    dummy_rgba_f32 = np.full((128, 128, 4), 0.8, dtype=np.float32)
    mask2, prob2 = predict_single_image(model, dummy_rgba_f32, device=active_device, return_original_size=True)
    assert mask2.shape == (128, 128)

    dummy_rgba_u16 = np.full((128, 128, 4), 45000, dtype=np.uint16)
    mask_rgba_u16, prob_rgba_u16 = predict_single_image(model, dummy_rgba_u16, device=active_device, return_original_size=True)
    assert mask_rgba_u16.shape == (128, 128)

    dummy_u16 = np.full((128, 128, 3), 32000, dtype=np.uint16)
    mask3, prob3 = predict_single_image(model, dummy_u16, device=active_device, return_original_size=True)
    assert mask3.shape == (128, 128)


def test_fill_holes_preserves_binary_range():
    from dl_pipeline.postprocess import fill_holes

    mask_255 = np.zeros((50, 50), dtype=np.uint8)
    mask_255[10:40, 10:40] = 255
    mask_255[20:25, 20:25] = 0
    filled_255 = fill_holes(mask_255, max_hole_size=50)
    unique_vals_255 = set(np.unique(filled_255))
    assert unique_vals_255.issubset({0, 255})
    assert 1 not in unique_vals_255

    mask_1 = (mask_255 > 0).astype(np.uint8)
    filled_1 = fill_holes(mask_1, max_hole_size=50)
    unique_vals_1 = set(np.unique(filled_1))
    assert unique_vals_1.issubset({0, 1})
    assert 255 not in unique_vals_1


def test_decode_image_negative_floats():
    neg_float_img = np.random.uniform(-1.0, 1.0, size=(64, 64, 3)).astype(np.float32)
    is_success, buffer = cv2.imencode(".tif", neg_float_img)
    assert is_success
    decoded = pipeline_service.decode_image_bytes(buffer.tobytes())
    assert decoded.dtype == np.uint8
    assert decoded.shape == (64, 64, 3)

    pos_float_over_255 = np.linspace(0.0, 1000.0, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
    is_success2, buffer2 = cv2.imencode(".tif", pos_float_over_255)
    assert is_success2
    decoded2 = pipeline_service.decode_image_bytes(buffer2.tobytes())
    assert decoded2.dtype == np.uint8
    assert decoded2.shape == (64, 64, 3)
    assert float(decoded2.std()) > 40.0


def test_serialize_template_area_scaling():
    dummy_template = {
        "keypoints": [(100.0, 100.0), (200.0, 100.0), (150.0, 200.0)],
        "invariants": [{
            "triangle_index": 0,
            "vertices": [[100.0, 100.0], [200.0, 100.0], [150.0, 200.0]],
            "centroid": [150.0, 133.33],
            "area": 5000.0,
            "angles": [1.0, 1.0, 1.14],
            "adjacent_ratios": [0.8],
        }],
        "od_center": (150.0, 150.0),
        "fovea_center": None,
    }
    M_inv = np.array([
        [2.0, 0.0, 10.0],
        [0.0, 2.0, 20.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    serialized = PipelineService.serialize_template(dummy_template, M_inv=M_inv)
    tri = serialized["triangles"][0]
    expected_area = 5000.0 * 4.0
    assert abs(tri["area"] - expected_area) < 1e-3
    assert abs(tri["vertices"][0][0] - (100.0 * 2.0 + 10.0)) < 1e-3
    assert abs(tri["vertices"][0][1] - (100.0 * 2.0 + 20.0)) < 1e-3
    assert abs(tri["centroid"][0] - (150.0 * 2.0 + 10.0)) < 1e-3
