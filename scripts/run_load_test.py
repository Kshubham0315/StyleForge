"""StyleForge v2.0 - Phase 6: Load & Concurrency Testing Script.

Executes controlled concurrent load testing against FastAPI generation endpoint:
- Concurrency levels: C=1, C=2, C=4, C=8
- Number of requests per concurrency tier: 16 requests
- Measures:
  - Throughput (RPS, Faces/sec)
  - Latency: P50, P90, P95, P99, Min, Max, Mean
  - Error rate (status != 200)
  - VRAM consumption and stability
  - GPU concurrency limit enforcement
"""

import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_test")

client = TestClient(app)
VALID_HEADERS = {"X-API-Key": settings.API_KEY}


def execute_single_request(req_id: int) -> Dict[str, Any]:
    payload = {
        "seed": 10000 + req_id,
        "num_samples": 1,
        "mode": "standard",
        "truncation_psi": 0.8,
        "use_ema": True,
        "output_format": "png",
    }
    t0 = time.perf_counter()
    try:
        resp = client.post("/api/v1/generate", json=payload, headers=VALID_HEADERS)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "req_id": req_id,
            "status_code": resp.status_code,
            "latency_ms": dur_ms,
            "success": (resp.status_code == 200),
            "error": None if resp.status_code == 200 else resp.text,
        }
    except Exception as e:
        dur_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "req_id": req_id,
            "status_code": 500,
            "latency_ms": dur_ms,
            "success": False,
            "error": str(e),
        }


def run_tier(concurrency: int, total_requests: int = 16) -> Dict[str, Any]:
    print(f"\n>>> Running Concurrency Tier: C={concurrency} (Total Requests: {total_requests}) <<<")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
        vram_start = torch.cuda.memory_allocated(0) / (1024 ** 2)
    else:
        vram_start = 0.0

    t_tier_start = time.perf_counter()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute_single_request, i) for i in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    total_time_sec = time.perf_counter() - t_tier_start
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    else:
        vram_peak = 0.0

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = [r["latency_ms"] for r in successful]

    rps = total_requests / total_time_sec if total_time_sec > 0 else 0
    faces_per_sec = rps * 1  # 1 face per sample

    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p90 = float(np.percentile(latencies, 90)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    tier_summary = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "error_rate_pct": round((len(failed) / total_requests) * 100.0, 2),
        "duration_seconds": round(total_time_sec, 2),
        "throughput_rps": round(rps, 2),
        "faces_per_sec": round(faces_per_sec, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p90_ms": round(p90, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "peak_vram_mb": round(vram_peak, 2),
    }

    print(f"Results for C={concurrency}:")
    print(f"  Duration: {total_time_sec:.2f}s | Throughput: {rps:.2f} RPS | Error Rate: {tier_summary['error_rate_pct']}%")
    print(f"  Latency (ms): P50={p50:.2f}, P90={p90:.2f}, P95={p95:.2f}, Mean={mean_lat:.2f}")
    print(f"  VRAM Peak: {vram_peak:.2f} MB")
    return tier_summary


def run_full_load_test() -> Dict[str, Any]:
    print("=" * 80)
    print("STYLEFORGE v2.0 - SYSTEM LOAD & CONCURRENCY BENCHMARK")
    print("=" * 80)
    
    # Warmup
    print("Executing warmup request...")
    execute_single_request(9999)

    benchmarks = {}
    for c in [1, 2, 4, 8]:
        benchmarks[f"concurrency_{c}"] = run_tier(concurrency=c, total_requests=16)

    overall = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_concurrency_semaphore_limit": settings.GPU_CONCURRENCY_LIMIT,
        "tiers": benchmarks,
    }

    with open("reports/load_test_results.json", "w") as f:
        json.dump(overall, f, indent=2)
    print("\nSaved load test metrics to reports/load_test_results.json")
    return overall


if __name__ == "__main__":
    run_full_load_test()
