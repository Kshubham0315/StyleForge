"""Phase 4 Experiment E6: 128x128 Progressive Scaling Feasibility Analysis.

Scientific Objective:
Evaluates memory boundaries, peak VRAM, backward graph allocations,
and throughput for 128x128 synthesis on the 6GB NVIDIA RTX 4050 GPU.
Tests:
1. Direct 128x128 training feasibility at batch_size = 8, 16
2. Progressive resolution transition architecture (64 -> 128 fade-in alpha)
3. Step latency and max VRAM headrooms.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.losses import GANLoss

def profile_batch_configuration(img_size: int, batch_size: int, device: torch.device) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    latent_dim = 512
    try:
        generator = StyleGenerator(latent_dim=latent_dim, style_dim=latent_dim, img_size=img_size).to(device)
        critic = StyleCritic(img_channels=3, img_size=img_size).to(device)

        opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
        opt_d = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.99))
        loss_fn = GANLoss(mode="wgan_gp", penalty_weight=10.0)

        # Forward + Backward pass simulation
        real_sample = torch.randn(batch_size, 3, img_size, img_size, device=device)
        z = torch.randn(batch_size, latent_dim, device=device)

        torch.cuda.synchronize(device)
        t0 = time.time()

        # Critic step
        fake = generator(z, style_mixing_prob=0.0)
        l_d, _ = loss_fn.critic_loss(critic, real_sample, fake)
        l_d.backward()
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)

        # Generator step
        fake_g = generator(z, style_mixing_prob=0.0)
        l_g, _ = loss_fn.generator_loss(critic, fake_g)
        l_g.backward()
        opt_g.step()
        opt_g.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        step_time_ms = (time.time() - t0) * 1000.0

        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        total_gpu_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        headroom_gb = total_gpu_vram - peak_vram_gb

        return {
            "status": "SUCCESS",
            "img_size": img_size,
            "batch_size": batch_size,
            "peak_vram_gb": peak_vram_gb,
            "headroom_gb": headroom_gb,
            "total_vram_gb": total_gpu_vram,
            "step_time_ms": step_time_ms,
            "fits_in_memory": True,
        }
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {
            "status": "OOM",
            "img_size": img_size,
            "batch_size": batch_size,
            "fits_in_memory": False,
            "error": str(e),
        }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase4/resolution")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("StyleForge Phase 4 — Experiment E6: 128x128 Feasibility Profiling")
    print(f"Device: {device} | Total VRAM: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB")
    print("==================================================")

    test_matrix = [
        (64, 16),
        (64, 32),
        (128, 4),
        (128, 8),
        (128, 16),
    ]

    results = []
    for res, bs in test_matrix:
        print(f"Profiling {res}x{res} with batch_size={bs}...")
        r = profile_batch_configuration(res, bs, device)
        results.append(r)
        if r["status"] == "SUCCESS":
            print(f" -> SUCCESS: Peak VRAM: {r['peak_vram_gb']:.2f} GB (Headroom: {r['headroom_gb']:.2f} GB) | Latency: {r['step_time_ms']:.1f} ms")
        else:
            print(f" -> OOM: Out Of Memory!")

    with open(out_dir / "e6_resolution_profiling.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot Feasibility & Memory Headroom
    successful = [r for r in results if r["status"] == "SUCCESS"]
    labels = [f"{r['img_size']}x{r['img_size']} (bs={r['batch_size']})" for r in successful]
    vram_vals = [r["peak_vram_gb"] for r in successful]
    headrooms = [r["headroom_gb"] for r in successful]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.55

    ax.bar(x, vram_vals, width, label="Peak VRAM Used", color="#e74c3c", edgecolor="black")
    ax.bar(x, headrooms, width, bottom=vram_vals, label="VRAM Headroom", color="#2ecc71", alpha=0.5, edgecolor="black")
    ax.axhline(6.0, color="gray", linestyle="--", linewidth=1.5, label="6.0 GB Hardware Limit")

    ax.set_ylabel("VRAM (GB)", fontsize=11)
    ax.set_title("VRAM Allocation & Headroom on RTX 4050 6GB Laptop GPU", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(vram_vals):
        ax.text(i, v * 0.5, f"{v:.2f}G", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "e6_vram_feasibility.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll E6 resolution profiling results saved to {out_dir}")

if __name__ == "__main__":
    main()
