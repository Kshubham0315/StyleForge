"""Integration tests for StyleForge v2.0 FastAPI Backend."""

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert data["version"] == "2.0.0"
    assert data["models_loaded"] is True


def test_readiness_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_prometheus_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "styleforge_requests_total" in response.text
    assert "styleforge_inference_latency_seconds" in response.text


def test_models_current_endpoint():
    response = client.get("/api/v1/models/current")
    assert response.status_code == 200
    data = response.json()
    assert "StyleForge" in data["name"]
    assert data["resolution"] == 64
    assert data["latent_dim"] == 512
    assert "modes_supported" in data
    assert "vram_telemetry" in data


def test_models_list_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) >= 1
    assert items[0]["version_tag"] == "v1.0.0"


def test_generate_unauthorized():
    # If auth is enabled, missing X-API-Key should return 401
    if settings.ENABLE_AUTH:
        payload = {"num_samples": 1, "seed": 42}
        response = client.post("/api/v1/generate", json=payload)
        assert response.status_code == 401
        err = response.json()["error"]
        assert err["code"] == "MISSING_CREDENTIALS"


def test_generate_standard_and_smooth_modes():
    headers = {"X-API-Key": settings.API_KEY}

    # 1. Standard Mode
    payload_std = {
        "num_samples": 1,
        "seed": 2026,
        "mode": "standard",
        "truncation_psi": 0.8,
    }
    res_std = client.post("/api/v1/generate", json=payload_std, headers=headers)
    assert res_std.status_code == 200, res_std.text
    data_std = res_std.json()
    assert data_std["status"] == "completed"
    assert data_std["mode"] == "standard"
    assert data_std["noise_scale"] == 1.0
    assert len(data_std["images"]) == 1
    assert data_std["images"][0]["base64_data"].startswith("data:image/png;base64,")

    # 2. Smooth Mode
    payload_smooth = {
        "num_samples": 1,
        "seed": 2026,
        "mode": "smooth",
        "truncation_psi": 0.8,
    }
    res_smooth = client.post("/api/v1/generate", json=payload_smooth, headers=headers)
    assert res_smooth.status_code == 200, res_smooth.text
    data_smooth = res_smooth.json()
    assert data_smooth["mode"] == "smooth"
    assert data_smooth["noise_scale"] == 0.25


def test_generation_history_and_artifact_download():
    headers = {"X-API-Key": settings.API_KEY}

    # Generate one sample first
    res_gen = client.post("/api/v1/generate", json={"num_samples": 1, "seed": 777}, headers=headers)
    assert res_gen.status_code == 200
    gen_id = res_gen.json()["generation_id"]
    download_url = res_gen.json()["images"][0]["download_url"]

    # Query history
    res_hist = client.get("/api/v1/generations?limit=10")
    assert res_hist.status_code == 200
    hist_data = res_hist.json()
    assert hist_data["total"] >= 1
    assert any(item["id"] == gen_id for item in hist_data["items"])

    # Query single generation by ID
    res_single = client.get(f"/api/v1/generations/{gen_id}")
    assert res_single.status_code == 200
    assert res_single.json()["id"] == gen_id

    # Download artifact binary
    res_art = client.get(download_url)
    assert res_art.status_code == 200
    assert res_art.headers["content-type"] == "image/png"
    assert len(res_art.content) > 0


def test_generate_input_validation():
    headers = {"X-API-Key": settings.API_KEY}

    # Invalid num_samples > 16
    res_invalid_samples = client.post("/api/v1/generate", json={"num_samples": 99}, headers=headers)
    assert res_invalid_samples.status_code == 422

    # Invalid truncation psi > 1.0
    res_invalid_psi = client.post("/api/v1/generate", json={"truncation_psi": 5.0}, headers=headers)
    assert res_invalid_psi.status_code == 422

    # Invalid mode
    res_invalid_mode = client.post("/api/v1/generate", json={"mode": "INVALID_MODE"}, headers=headers)
    assert res_invalid_mode.status_code == 422
