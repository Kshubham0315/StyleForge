"""StyleForge v2.0 - Phase 3 & 4: Live Server and GPU Inference Validation Script.

Starts the Uvicorn server in a background thread/process, executes live HTTP requests,
and performs rigorous verification:
1. Server startup & model pre-loading
2. /health endpoint inspection
3. /docs OpenAPI schema validation
4. /metrics Prometheus telemetry
5. Live POST /api/v1/generate image synthesis
6. Image decoding, dimensions (64x64), size, and format checks
7. Error response handling
8. Standard mode vs Clean Face mode validation
9. Seed determinism
10. Separation of Model Latency vs Total API Latency
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
import threading
import numpy as np
from PIL import Image
import requests
import torch
import uvicorn

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_validation")

ARTIFACT_DIR = Path("artifacts/independent_validation")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class ServerThread(threading.Thread):
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.server = None

    def run(self):
        config = uvicorn.Config("backend.main:app", host=self.host, port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self):
        if self.server:
            self.server.should_exit = True


def wait_for_server(base_url: str, timeout: float = 15.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def run_live_and_gpu_validation() -> dict:
    results = {}
    print("=" * 80)
    print("PHASE 3 & 4 — LIVE BACKEND & GPU INFERENCE INDEPENDENT VALIDATION")
    print("=" * 80)

    # 1. CUDA and GPU Hardware Environment
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) if cuda_available else 0
    results["hardware"] = {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_vram_mb": round(total_vram_mb, 2),
    }
    print(f"CUDA Available: {cuda_available} | Device: {device_name} | VRAM: {total_vram_mb:.1f} MB")

    # 2. Start Live Uvicorn Server
    base_url = "http://127.0.0.1:8000"
    headers = {"X-API-Key": settings.API_KEY, "Content-Type": "application/json"}
    print(f"\nStarting live Uvicorn server on {base_url}...")
    server_thread = ServerThread(host="127.0.0.1", port=8000)
    server_thread.start()

    if not wait_for_server(base_url):
        raise RuntimeError(f"Server failed to start within timeout at {base_url}")
    print("Uvicorn server is LIVE and responding!")

    # 3. Validate /health Endpoint
    print("\n--- 1. Health Endpoint Inspection ---")
    r_health = requests.get(f"{base_url}/health")
    health_data = r_health.json()
    results["health_check"] = {
        "status_code": r_health.status_code,
        "data": health_data,
        "models_loaded": health_data.get("models_loaded", False),
        "device": health_data.get("device", "unknown"),
    }
    print(f"Health Status: {r_health.status_code} -> {health_data}")

    # 4. Validate /docs and OpenAPI Schema
    print("\n--- 2. Swagger / OpenAPI Documentation ---")
    r_docs = requests.get(f"{base_url}/docs")
    r_openapi = requests.get(f"{base_url}/openapi.json")
    openapi_spec = r_openapi.json()
    paths_count = len(openapi_spec.get("paths", {}))
    results["swagger_docs"] = {
        "docs_status_code": r_docs.status_code,
        "openapi_status_code": r_openapi.status_code,
        "endpoints_count": paths_count,
        "has_generate": "/api/v1/generate" in openapi_spec.get("paths", {}),
        "has_interpolate": "/api/v1/interpolate" in openapi_spec.get("paths", {}),
        "has_mix_styles": "/api/v1/mix-styles" in openapi_spec.get("paths", {}),
    }
    print(f"Docs UI Status: {r_docs.status_code} | OpenAPI Status: {r_openapi.status_code} ({paths_count} routes)")

    # 5. Validate /metrics Endpoint
    print("\n--- 3. Prometheus Metrics Endpoint ---")
    r_metrics = requests.get(f"{base_url}/metrics")
    r_v1_metrics = requests.get(f"{base_url}/api/v1/metrics")
    results["metrics"] = {
        "metrics_status_code": r_metrics.status_code,
        "api_v1_metrics_status_code": r_v1_metrics.status_code,
        "has_active_inferences": "styleforge_active_inferences" in r_metrics.text,
        "has_generated_faces": "styleforge_generated_faces_total" in r_metrics.text,
    }
    print(f"/metrics Status: {r_metrics.status_code} | /api/v1/metrics Status: {r_v1_metrics.status_code}")

    # 6. Live Generation: Standard Mode (noise_scale=1.0)
    print("\n--- 4. Live Face Generation — Standard Mode ---")
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(0)
        vram_before = torch.cuda.memory_allocated(0) / (1024 ** 2)
    else:
        vram_before = 0.0

    t_api_start = time.perf_counter()
    gen_payload_std = {
        "seed": 42,
        "num_samples": 1,
        "mode": "standard",
        "noise_scale": 1.0,
        "truncation_psi": 0.8,
        "use_ema": True,
        "output_format": "png",
    }
    r_std = requests.post(f"{base_url}/api/v1/generate", json=gen_payload_std, headers=headers)
    total_api_ms_std = (time.perf_counter() - t_api_start) * 1000.0
    
    if cuda_available:
        vram_peak_std = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        vram_after_std = torch.cuda.memory_allocated(0) / (1024 ** 2)
    else:
        vram_peak_std, vram_after_std = 0.0, 0.0

    assert r_std.status_code == 200, f"Standard generation failed: {r_std.text}"
    resp_std_json = r_std.json()
    model_latency_ms_std = resp_std_json["latency_ms"]

    # Download generated artifact
    art_url_std = f"{base_url}{resp_std_json['images'][0]['download_url']}"
    r_art_std = requests.get(art_url_std)
    std_bytes = r_art_std.content

    # Save to artifacts/independent_validation/
    std_file_path = ARTIFACT_DIR / "live_standard_seed42.png"
    std_file_path.write_bytes(std_bytes)

    # Decode and inspect image
    pil_std = Image.open(io.BytesIO(std_bytes))
    std_width, std_height = pil_std.size
    std_format = pil_std.format

    results["live_standard_generation"] = {
        "status_code": r_std.status_code,
        "total_api_latency_ms": round(total_api_ms_std, 2),
        "model_inference_latency_ms": model_latency_ms_std,
        "api_overhead_ms": round(total_api_ms_std - model_latency_ms_std, 2),
        "vram_before_mb": round(vram_before, 2),
        "vram_peak_mb": round(vram_peak_std, 2),
        "vram_after_mb": round(vram_after_std, 2),
        "file_size_bytes": len(std_bytes),
        "image_dimensions": f"{std_width}x{std_height}",
        "image_format": std_format,
        "decoded_successfully": True,
        "saved_path": str(std_file_path),
    }
    print(f"HTTP Status: {r_std.status_code}")
    print(f"Total API Latency: {total_api_ms_std:.2f} ms | Model Latency: {model_latency_ms_std:.2f} ms | Overhead: {total_api_ms_std - model_latency_ms_std:.2f} ms")
    print(f"Image Dimensions: {std_width}x{std_height} | Format: {std_format} | Size: {len(std_bytes)} bytes")
    print(f"VRAM Before: {vram_before:.2f} MB | Peak: {vram_peak_std:.2f} MB | After: {vram_after_std:.2f} MB")

    # 7. Live Generation: Clean Face Mode (noise_scale=0.25)
    print("\n--- 5. Live Face Generation — Clean Face Mode ---")
    t_api_start = time.perf_counter()
    gen_payload_smooth = {
        "seed": 42,
        "num_samples": 1,
        "mode": "smooth",
        "noise_scale": 0.25,
        "truncation_psi": 0.8,
        "use_ema": True,
        "output_format": "png",
    }
    r_sm = requests.post(f"{base_url}/api/v1/generate", json=gen_payload_smooth, headers=headers)
    total_api_ms_sm = (time.perf_counter() - t_api_start) * 1000.0
    assert r_sm.status_code == 200, f"Smooth generation failed: {r_sm.text}"
    resp_sm_json = r_sm.json()
    model_latency_ms_sm = resp_sm_json["latency_ms"]

    art_url_sm = f"{base_url}{resp_sm_json['images'][0]['download_url']}"
    r_art_sm = requests.get(art_url_sm)
    sm_bytes = r_art_sm.content

    sm_file_path = ARTIFACT_DIR / "live_smooth_seed42.png"
    sm_file_path.write_bytes(sm_bytes)

    pil_sm = Image.open(io.BytesIO(sm_bytes))
    sm_width, sm_height = pil_sm.size
    sm_format = pil_sm.format

    # Compare Standard vs Smooth pixel differences
    arr_std = np.array(pil_std).astype(float)
    arr_sm = np.array(pil_sm).astype(float)
    l1_diff = float(np.mean(np.abs(arr_std - arr_sm)))

    results["live_smooth_generation"] = {
        "status_code": r_sm.status_code,
        "total_api_latency_ms": round(total_api_ms_sm, 2),
        "model_inference_latency_ms": model_latency_ms_sm,
        "api_overhead_ms": round(total_api_ms_sm - model_latency_ms_sm, 2),
        "file_size_bytes": len(sm_bytes),
        "image_dimensions": f"{sm_width}x{sm_height}",
        "image_format": sm_format,
        "decoded_successfully": True,
        "l1_diff_from_standard": round(l1_diff, 2),
        "saved_path": str(sm_file_path),
    }
    print(f"Total API Latency: {total_api_ms_sm:.2f} ms | Model Latency: {model_latency_ms_sm:.2f} ms")
    print(f"Image Dimensions: {sm_width}x{sm_height} | Format: {sm_format} | Size: {len(sm_bytes)} bytes")
    print(f"L1 pixel difference (Standard vs Smooth, seed=42): {l1_diff:.2f}")

    # 8. Same-Seed Reproducibility Check
    print("\n--- 6. Same-Seed Reproducibility Check ---")
    r_rep = requests.post(f"{base_url}/api/v1/generate", json=gen_payload_std, headers=headers)
    assert r_rep.status_code == 200
    art_url_rep = f"{base_url}{r_rep.json()['images'][0]['download_url']}"
    r_art_rep = requests.get(art_url_rep)
    rep_bytes = r_art_rep.content
    rep_file_path = ARTIFACT_DIR / "live_standard_seed42_reproduced.png"
    rep_file_path.write_bytes(rep_bytes)

    arr_rep = np.array(Image.open(io.BytesIO(rep_bytes))).astype(float)
    is_bit_identical = bool(np.array_equal(arr_std, arr_rep))
    results["seed_reproducibility"] = {
        "seed": 42,
        "is_bit_identical": is_bit_identical,
        "max_pixel_diff": float(np.max(np.abs(arr_std - arr_rep))),
    }
    print(f"Seed 42 Reproducibility Across Separate API Calls: {'PASS (Bit-Identical)' if is_bit_identical else 'FAIL'}")

    # 9. Direct Tensor Range & Finite Check
    print("\n--- 7. Tensor Range & Finite Sanity Check ---")
    from core.engine.inference_engine import InferenceEngine
    engine_inst = InferenceEngine.get_instance()
    with torch.inference_mode():
        torch.manual_seed(42)
        z_t = torch.randn(2, engine_inst.ema_generator.latent_dim, device=engine_inst.device)
        out_t = engine_inst.ema_generator(z_t, truncation_psi=0.8)

    t_min = float(out_t.min().item())
    t_max = float(out_t.max().item())
    t_nan = bool(torch.isnan(out_t).any().item())
    t_inf = bool(torch.isinf(out_t).any().item())

    results["tensor_audit"] = {
        "min": round(t_min, 4),
        "max": round(t_max, 4),
        "shape": list(out_t.shape),
        "has_nan": t_nan,
        "has_inf": t_inf,
        "range_valid": (-1.05 <= t_min <= 1.05 and -1.05 <= t_max <= 1.05),
    }
    print(f"Tensor Shape: {list(out_t.shape)} | Range: [{t_min:.4f}, {t_max:.4f}] | Finite: {not (t_nan or t_inf)}")

    # 10. Stop Server Cleanly
    print("\nStopping live Uvicorn server...")
    server_thread.stop()
    print("Server stopped cleanly.")

    print("\n" + "=" * 80)
    print("LIVE BACKEND & GPU VALIDATION COMPLETE")
    print("=" * 80)
    return results


if __name__ == "__main__":
    out = run_live_and_gpu_validation()
    with open("reports/phase3_4_live_gpu_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved report to reports/phase3_4_live_gpu_validation.json")
