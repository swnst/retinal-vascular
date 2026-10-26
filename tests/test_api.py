import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app, pipeline_service


# --- API Test Suite ---
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
