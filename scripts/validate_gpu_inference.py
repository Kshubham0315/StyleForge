"""StyleForge v2.0 - Phase 2: Real GPU Inference & Reliability Validation Script.

Executes rigorous empirical measurements:
1. Production checkpoint loading & SHA-256 verification.
2. Cold start load latency.
3. Warm inference latency benchmarks (20 iterations: P50, P95, mean, min, max).
4. Physical GPU VRAM tracking before, during, and after inference.
5. Tensor verification: shape (B, 3, 64, 64), range [-1, 1], finite check (no NaN/Inf).
6. Controlled mode comparison: Standard Mode (noise_scale=1.0) vs Smooth Mode (noise_scale=0.25).
7. Seed determinism verification across independent runs.
8. CPU fallback execution verification.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from core.engine.checkpoint_manager import CheckpointManager
from core.engine.config import GenerationConfig, InferenceMode
from core.engine.device_manager import DeviceManager
from core.engine.inference_engine import InferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpu_validation")

CHECKPOINT_PATH = Path("checkpoints/styleforge_final_model.pt")


def run_gpu_validation() -> dict:
    results = {}
    print("=" * 80)
    print("STYLEFORGE v2.0 - REAL HARDWARE INFERENCE & GPU VALIDATION")
    print("=" * 80)

    # 1. Device and Hardware Telemetry
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Only"
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) if cuda_available else 0
    results["hardware"] = {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "total_vram_mb": round(total_vram_mb, 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
    }
    print(f"Device: {device_name} (CUDA Available: {cuda_available})")
    print(f"Total VRAM: {total_vram_mb:.1f} MB")

    # 2. Checkpoint Verification
    print("\n--- 1. Checkpoint Verification ---")
    sha256 = CheckpointManager.compute_sha256(CHECKPOINT_PATH)
    meta = CheckpointManager.inspect_checkpoint(CHECKPOINT_PATH)
    results["checkpoint"] = {
        "path": str(CHECKPOINT_PATH),
        "sha256": sha256,
        "file_size_mb": meta["file_size_mb"],
        "has_generator_raw": meta["has_raw_weights"],
        "has_generator_ema": meta["has_ema_weights"],
        "img_size": meta["img_size"],
    }
    print(f"Checkpoint SHA-256: {sha256}")
    print(f"File size: {meta['file_size_mb']:.2f} MB")
    print(f"Has Raw: {meta['has_raw_weights']}, Has EMA: {meta['has_ema_weights']}")

    # 3. Model Loading & Cold Start Latency
    print("\n--- 2. Cold Start Model Loading ---")
    vram_before_load = torch.cuda.memory_allocated(0) / (1024 ** 2) if cuda_available else 0
    t0 = time.perf_counter()
    target_device = "cuda" if cuda_available else "cpu"
    engine = InferenceEngine(checkpoint_path=CHECKPOINT_PATH, device=target_device)
    cold_load_ms = (time.perf_counter() - t0) * 1000.0
    vram_after_load = torch.cuda.memory_allocated(0) / (1024 ** 2) if cuda_available else 0
    results["cold_start"] = {
        "load_time_ms": round(cold_load_ms, 2),
        "vram_before_load_mb": round(vram_before_load, 2),
        "vram_after_load_mb": round(vram_after_load, 2),
        "vram_model_weights_mb": round(vram_after_load - vram_before_load, 2),
    }
    print(f"Cold model load latency: {cold_load_ms:.2f} ms")
    print(f"VRAM for model weights: {vram_after_load - vram_before_load:.2f} MB")

    # 4. First Inference (Cold Inference Latency)
    print("\n--- 3. First Inference Execution ---")
    t0 = time.perf_counter()
    cold_cfg = GenerationConfig(seed=42, num_samples=1, mode=InferenceMode.STANDARD, use_ema=True)
    imgs_cold, tele_cold = engine.generate(cold_cfg)
    first_inf_ms = (time.perf_counter() - t0) * 1000.0
    results["first_inference"] = {
        "total_first_inf_ms": round(first_inf_ms, 2),
        "internal_telemetry_ms": tele_cold["latency_ms"],
    }
    print(f"First inference total latency: {first_inf_ms:.2f} ms")

    # 5. Warm Benchmark (20 Iterations, Batch Size = 1)
    print("\n--- 4. Warm Latency Benchmark (20 Iterations, Batch=1) ---")
    latencies = []
    vram_peaks = []
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(0)

    for i in range(20):
        t_start = time.perf_counter()
        cfg = GenerationConfig(seed=1000 + i, num_samples=1, mode=InferenceMode.STANDARD, use_ema=True)
        _, t_info = engine.generate(cfg)
        latencies.append(t_info["latency_ms"])
        if cuda_available:
            vram_peaks.append(torch.cuda.max_memory_allocated(0) / (1024 ** 2))

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    peak_vram = float(np.max(vram_peaks)) if vram_peaks else 0.0

    results["warm_benchmark_b1"] = {
        "iterations": 20,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean_lat, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2),
        "peak_vram_mb": round(peak_vram, 2),
    }
    print(f"Latencies (ms): P50={p50:.2f}, P95={p95:.2f}, Mean={mean_lat:.2f}, Min={min_lat:.2f}, Max={max_lat:.2f}")
    print(f"Peak VRAM during batch=1: {peak_vram:.2f} MB")

    # 6. Batch Scaling (Batch Size = 4, Batch Size = 8)
    print("\n--- 5. Batch Scaling (B=4 and B=8) ---")
    results["batch_scaling"] = {}
    for bs in [4, 8]:
        lat_b = []
        for _ in range(5):
            _, tele_b = engine.generate(GenerationConfig(seed=2026, num_samples=bs, mode=InferenceMode.STANDARD, use_ema=True))
            lat_b.append(tele_b["latency_ms"])
        mean_b = float(np.mean(lat_b))
        results["batch_scaling"][f"batch_{bs}"] = {
            "mean_latency_ms": round(mean_b, 2),
            "latency_per_face_ms": round(mean_b / bs, 2),
        }
        print(f"Batch {bs}: Mean={mean_b:.2f} ms ({mean_b / bs:.2f} ms/face)")

    # 7. Image Tensor Verification: Shape, Range [-1, 1], Finite Check
    print("\n--- 6. Image Tensor & Output Range Verification ---")
    # Generate raw tensor directly via generator
    gen_model = engine.ema_generator
    with torch.inference_mode():
        torch.manual_seed(999)
        z_sample = torch.randn(4, gen_model.latent_dim, device=engine.device)
        t_out = gen_model(z_sample, truncation_psi=0.8)
    
    t_shape = list(t_out.shape)
    t_min = float(t_out.min().item())
    t_max = float(t_out.max().item())
    t_mean = float(t_out.mean().item())
    has_nan = bool(torch.isnan(t_out).any().item())
    has_inf = bool(torch.isinf(t_out).any().item())

    results["tensor_audit"] = {
        "shape": t_shape,
        "expected_shape": [4, 3, 64, 64],
        "shape_correct": (t_shape == [4, 3, 64, 64]),
        "min_val": round(t_min, 4),
        "max_val": round(t_max, 4),
        "mean_val": round(t_mean, 4),
        "range_valid": (-1.05 <= t_min <= 1.05 and -1.05 <= t_max <= 1.05),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "finite": not (has_nan or has_inf),
    }
    print(f"Tensor Shape: {t_shape} (Expected: [4, 3, 64, 64]) -> {'PASS' if t_shape == [4, 3, 64, 64] else 'FAIL'}")
    print(f"Tensor Range: [{t_min:.4f}, {t_max:.4f}] (Expected in [-1, 1] tanh range) -> {'PASS' if (-1.05 <= t_min and t_max <= 1.05) else 'FAIL'}")
    print(f"Finite Check (No NaN/Inf): {'PASS' if not (has_nan or has_inf) else 'FAIL'}")

    # 8. Controlled Mode Comparison (Standard vs Smooth)
    print("\n--- 7. Controlled Mode Comparison (Standard vs Smooth) ---")
    imgs_std, tele_std = engine.generate(GenerationConfig(seed=42, num_samples=1, mode=InferenceMode.STANDARD, use_ema=True))
    imgs_sm, tele_sm = engine.generate(GenerationConfig(seed=42, num_samples=1, mode=InferenceMode.SMOOTH, use_ema=True))
    
    arr_std = np.array(imgs_std[0])
    arr_sm = np.array(imgs_sm[0])
    diff_l1 = float(np.mean(np.abs(arr_std.astype(float) - arr_sm.astype(float))))

    results["mode_comparison"] = {
        "standard_noise_scale": tele_std["noise_scale"],
        "smooth_noise_scale": tele_sm["noise_scale"],
        "l1_pixel_diff": round(diff_l1, 2),
        "different_outputs": diff_l1 > 0.0,
    }
    print(f"Standard noise_scale: {tele_std['noise_scale']}, Smooth noise_scale: {tele_sm['noise_scale']}")
    print(f"L1 Pixel Difference between Standard and Smooth (seed 42): {diff_l1:.2f} (Different: {diff_l1 > 0})")

    # 9. Determinism Across Runs
    print("\n--- 8. Seed Determinism Check ---")
    run_a_imgs, _ = engine.generate(GenerationConfig(seed=12345, num_samples=1, mode=InferenceMode.STANDARD, use_ema=True))
    run_b_imgs, _ = engine.generate(GenerationConfig(seed=12345, num_samples=1, mode=InferenceMode.STANDARD, use_ema=True))
    arr_a = np.array(run_a_imgs[0])
    arr_b = np.array(run_b_imgs[0])
    is_identical = bool(np.array_equal(arr_a, arr_b))
    results["determinism"] = {
        "seed": 12345,
        "identical": is_identical,
        "max_abs_diff": float(np.max(np.abs(arr_a.astype(float) - arr_b.astype(float)))),
    }
    print(f"Seed 12345 Determinism Across 2 Independent Calls: {'PASS (Bit-identical)' if is_identical else 'FAIL'}")

    # 10. CPU Fallback Verification
    print("\n--- 9. CPU Fallback Verification ---")
    t0 = time.perf_counter()
    engine_cpu = InferenceEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
    cpu_load_ms = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    cpu_imgs, cpu_tele = engine_cpu.generate(GenerationConfig(seed=42, num_samples=1, mode=InferenceMode.STANDARD, use_ema=True))
    cpu_inf_ms = (time.perf_counter() - t0) * 1000.0

    results["cpu_fallback"] = {
        "device": "cpu",
        "cpu_load_ms": round(cpu_load_ms, 2),
        "cpu_inference_ms": round(cpu_inf_ms, 2),
        "success": len(cpu_imgs) == 1 and cpu_imgs[0].size == (64, 64),
    }
    print(f"CPU load latency: {cpu_load_ms:.2f} ms")
    print(f"CPU inference latency: {cpu_inf_ms:.2f} ms")
    print(f"CPU fallback status: {'PASS' if results['cpu_fallback']['success'] else 'FAIL'}")

    # Output Summary
    print("\n" + "=" * 80)
    print("PHASE 2 VALIDATION COMPLETE")
    print("=" * 80)
    return results


if __name__ == "__main__":
    out = run_gpu_validation()
    with open("reports/phase2_gpu_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved report to reports/phase2_gpu_validation.json")
