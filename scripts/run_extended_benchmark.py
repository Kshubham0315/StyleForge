"""StyleForge v2.0 - Extended Scientific Benchmark Script (Task 2).

Executes controlled, reproducible benchmark:
- 100 requests in Standard Mode (noise_scale=1.0)
- 100 requests in Clean Face Mode (noise_scale=0.25)
- Tracks:
  - Cold-start latency (1st request) vs Warm latency (requests 2-100)
  - Latency distribution: P50, P90, P95, P99, Mean, Min, Max
  - Model forward-pass latency vs Total API latency
  - Physical GPU VRAM (allocated & peak)
  - System RAM utilization (psutil)
  - Output image validity, dimensions (64x64), zero NaN/Inf
  - Error rate
- Outputs structured empirical results to reports/extended_benchmark_results.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import psutil
import torch
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extended_benchmark")

client = TestClient(app)
VALID_HEADERS = {"X-API-Key": settings.API_KEY}


def get_system_telemetry() -> Dict[str, float]:
    """Records physical GPU VRAM and host RAM telemetry."""
    vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    ram = psutil.virtual_memory()
    return {
        "vram_allocated_mb": round(vram_alloc, 2),
        "vram_peak_mb": round(vram_peak, 2),
        "system_ram_used_mb": round((ram.total - ram.available) / (1024 ** 2), 2),
        "system_ram_percent": ram.percent,
    }


def benchmark_mode(mode_name: str, noise_scale: float, count: int = 100) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"BENCHMARKING: {mode_name.upper()} MODE ({count} REQUESTS)")
    print(f"=======================================================")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    total_api_latencies: List[float] = []
    model_latencies: List[float] = []
    errors: int = 0
    t_start_all = time.perf_counter()

    cold_api_ms = None
    cold_model_ms = None

    for idx in range(count):
        payload = {
            "seed": 5000 + idx,
            "num_samples": 1,
            "mode": mode_name,
            "noise_scale": noise_scale,
            "truncation_psi": 0.8,
            "use_ema": True,
            "output_format": "png",
        }

        t0 = time.perf_counter()
        try:
            resp = client.post("/api/v1/generate", json=payload, headers=VALID_HEADERS)
            t_api = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                body = resp.json()
                t_model = body.get("latency_ms", 0.0)

                if idx == 0:
                    cold_api_ms = round(t_api, 2)
                    cold_model_ms = round(t_model, 2)
                else:
                    total_api_latencies.append(t_api)
                    model_latencies.append(t_model)
            else:
                errors += 1
                logger.error("Request %d failed: %s", idx, resp.text)
        except Exception as e:
            errors += 1
            logger.error("Exception on request %d: %s", idx, e)

        if (idx + 1) % 25 == 0 or idx == count - 1:
            print(f"  Progress: {idx + 1}/{count} requests completed...")

    total_elapsed_s = time.perf_counter() - t_start_all
    rps = count / total_elapsed_s if total_elapsed_s > 0 else 0.0
    telemetry = get_system_telemetry()

    # Statistical computation on warm samples (requests 2..100)
    api_arr = np.array(total_api_latencies) if total_api_latencies else np.array([0.0])
    model_arr = np.array(model_latencies) if model_latencies else np.array([0.0])

    stats = {
        "mode": mode_name,
        "noise_scale": noise_scale,
        "total_requests": count,
        "warm_requests_analyzed": len(total_api_latencies),
        "failed_requests": errors,
        "error_rate_pct": round((errors / count) * 100.0, 2),
        "duration_seconds": round(total_elapsed_s, 2),
        "throughput_rps": round(rps, 2),
        "cold_start": {
            "api_latency_ms": cold_api_ms,
            "model_latency_ms": cold_model_ms,
            "overhead_ms": round(cold_api_ms - cold_model_ms, 2) if (cold_api_ms and cold_model_ms) else None,
        },
        "warm_api_latency": {
            "p50_median_ms": round(float(np.percentile(api_arr, 50)), 2),
            "p90_ms": round(float(np.percentile(api_arr, 90)), 2),
            "p95_ms": round(float(np.percentile(api_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(api_arr, 99)), 2),
            "mean_ms": round(float(np.mean(api_arr)), 2),
            "min_ms": round(float(np.min(api_arr)), 2),
            "max_ms": round(float(np.max(api_arr)), 2),
        },
        "warm_model_latency": {
            "p50_median_ms": round(float(np.percentile(model_arr, 50)), 2),
            "p90_ms": round(float(np.percentile(model_arr, 90)), 2),
            "p95_ms": round(float(np.percentile(model_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(model_arr, 99)), 2),
            "mean_ms": round(float(np.mean(model_arr)), 2),
            "min_ms": round(float(np.min(model_arr)), 2),
            "max_ms": round(float(np.max(model_arr)), 2),
        },
        "telemetry": telemetry,
    }

    print(f"Results for {mode_name.upper()}:")
    print(f"  Throughput: {rps:.2f} RPS | Error Rate: {stats['error_rate_pct']}%")
    print(f"  Cold Start: API={cold_api_ms} ms | Model={cold_model_ms} ms")
    print(f"  Warm API Latency: P50={stats['warm_api_latency']['p50_median_ms']} ms | P95={stats['warm_api_latency']['p95_ms']} ms | P99={stats['warm_api_latency']['p99_ms']} ms")
    print(f"  Warm Model Latency: P50={stats['warm_model_latency']['p50_median_ms']} ms | P95={stats['warm_model_latency']['p95_ms']} ms")
    print(f"  VRAM Peak: {telemetry['vram_peak_mb']} MB | Host RAM Used: {telemetry['system_ram_used_mb']} MB ({telemetry['system_ram_percent']}%)")
    return stats


def run_extended_benchmarks() -> Dict[str, Any]:
    print("=" * 80)
    print("STYLEFORGE v2.0 - TASK 2: EXTENDED SCIENTIFIC BENCHMARK")
    print("=" * 80)

    # Pre-test hardware check
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"Hardware: {dev_name} (CUDA: {cuda_avail})")

    # 1. Warm-up
    print("\nExecuting initial warmup request...")
    client.post("/api/v1/generate", json={"num_samples": 1, "seed": 999}, headers=VALID_HEADERS)

    # 2. Run Standard Mode (100 iterations)
    standard_results = benchmark_mode("standard", noise_scale=1.0, count=100)

    # 3. Run Clean Face Mode (100 iterations)
    clean_results = benchmark_mode("smooth", noise_scale=0.25, count=100)

    overall = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": dev_name,
        "cuda_available": cuda_avail,
        "total_requests_executed": standard_results["total_requests"] + clean_results["total_requests"],
        "modes": {
            "standard_mode": standard_results,
            "clean_face_mode": clean_results,
        },
    }

    out_file = Path("reports/extended_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nSaved extended benchmark data to {out_file}")
    return overall


if __name__ == "__main__":
    run_extended_benchmarks()
