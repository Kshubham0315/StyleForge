"""Phase 4 Experiment E4 & E5: Generator & Critic Capacity Scaling at 64x64.

Scientific Objective:
Evaluates parameter capacity trade-offs on 6GB RTX 4050:
1. Standard Base Channels (base_channels=64, max_channels=512)
2. Scaled / Wide Capacity (base_channels=128, max_channels=512)

Measures:
- Parameter counts (Generator & Critic)
- Peak VRAM consumption (GB)
- Step latency (ms / step)
- Convergence stability & Inception FID
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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from data.dataset import get_dataloader
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.losses import GANLoss

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return ((t * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_and_train_capacity(
    config_name: str,
    gen_base_channels: int,
    critic_base_channels: int,
    steps: int = 150,
    device: torch.device = None,
    val_loader = None,
    real_uint8_cache = None,
) -> dict:
    print(f"\n==================================================")
    print(f"Capacity Test: {config_name}")
    print(f"G base: {gen_base_channels} | Critic base: {critic_base_channels}")
    print("==================================================")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(42)

    train_loader = get_dataloader("data/celeba/img_align_celeba", split="train", img_size=64, batch_size=16, num_workers=0, seed=42)

    latent_dim = 512
    generator = StyleGenerator(
        latent_dim=latent_dim,
        style_dim=latent_dim,
        img_size=64,
        base_channels=gen_base_channels,
        max_channels=512,
    ).to(device)

    critic = StyleCritic(
        img_channels=3,
        img_size=64,
        base_channels=critic_base_channels,
        max_channels=512,
    ).to(device)

    g_params = count_parameters(generator)
    d_params = count_parameters(critic)
    total_params = g_params + d_params
    print(f"Generator Parameters: {g_params:,}")
    print(f"Critic Parameters:    {d_params:,}")
    print(f"Total Parameters:     {total_params:,}")

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.99))
    loss_fn = GANLoss(mode="wgan_gp", penalty_weight=10.0)

    n_critic = 2
    data_iter = iter(train_loader)

    # Warmup step
    for _ in range(n_critic):
        batch = next(data_iter)
        if isinstance(batch, (list, tuple)): batch = batch[0]
        batch = batch.to(device)
        z = torch.randn(batch.size(0), latent_dim, device=device)
        fake = generator(z, style_mixing_prob=0.0)
        l_d, _ = loss_fn.critic_loss(critic, batch, fake)
        l_d.backward()
        opt_d.step()
        opt_d.zero_grad(set_to_none=True)

    z = torch.randn(16, latent_dim, device=device)
    fake = generator(z, style_mixing_prob=0.0)
    l_g, _ = loss_fn.generator_loss(critic, fake)
    l_g.backward()
    opt_g.step()
    opt_g.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    start_time = time.time()

    for step in range(1, steps + 1):
        for _ in range(n_critic):
            try:
                real_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                real_batch = next(data_iter)
            if isinstance(real_batch, (list, tuple)): real_batch = real_batch[0]
            real_batch = real_batch.to(device)

            opt_d.zero_grad(set_to_none=True)
            z_c = torch.randn(real_batch.size(0), latent_dim, device=device)
            fake_c = generator(z_c, style_mixing_prob=0.0)
            loss_d, _ = loss_fn.critic_loss(critic, real_batch, fake_c)
            loss_d.backward()
            opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        z_g = torch.randn(16, latent_dim, device=device)
        fake_g = generator(z_g, style_mixing_prob=0.0)
        loss_g, _ = loss_fn.generator_loss(critic, fake_g)
        loss_g.backward()
        opt_g.step()

    torch.cuda.synchronize(device)
    total_time = time.time() - start_time
    step_latency_ms = (total_time / steps) * 1000.0
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"Elapsed: {total_time:.1f}s | Latency: {step_latency_ms:.2f} ms/step | Peak VRAM: {peak_vram_gb:.2f} GB")

    # Validation FID
    print("Evaluating Validation FID...")
    generator.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for u_real in real_uint8_cache:
        fid_metric.update(u_real.to(device), real=True)

    N_val = len(val_loader.dataset)
    torch.manual_seed(42)
    count = 0
    with torch.no_grad():
        while count < N_val:
            cur_bs = min(32, N_val - count)
            z = torch.randn(cur_bs, latent_dim, device=device)
            f_img = generator(z, truncation_psi=0.8)
            fid_metric.update(to_uint8(f_img), real=False)
            count += cur_bs

    fid_score = float(fid_metric.compute().item())
    print(f"Validation FID: {fid_score:.2f}")

    return {
        "config_name": config_name,
        "gen_base_channels": gen_base_channels,
        "critic_base_channels": critic_base_channels,
        "g_params": g_params,
        "d_params": d_params,
        "total_params": total_params,
        "peak_vram_gb": peak_vram_gb,
        "step_latency_ms": step_latency_ms,
        "fid": fid_score,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase4/capacity")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_loader = get_dataloader("data/celeba/img_align_celeba", split="val", img_size=64, batch_size=32, num_workers=0, seed=42)
    real_uint8_cache = []
    for b in val_loader:
        if isinstance(b, (list, tuple)): b = b[0]
        real_uint8_cache.append(to_uint8(b))

    configs = [
        {"name": "Standard Capacity (Base=64)", "g_base": 512, "d_base": 64},
        {"name": "Scaled Critic Capacity (Base=128)", "g_base": 512, "d_base": 128},
    ]

    results = []
    for cfg in configs:
        res = profile_and_train_capacity(
            config_name=cfg["name"],
            gen_base_channels=cfg["g_base"],
            critic_base_channels=cfg["d_base"],
            steps=150,
            device=device,
            val_loader=val_loader,
            real_uint8_cache=real_uint8_cache,
        )
        results.append(res)

    with open(out_dir / "e4_capacity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = [r["config_name"] for r in results]
    vrams = [r["peak_vram_gb"] for r in results]
    latencies = [r["step_latency_ms"] for r in results]
    fids = [r["fid"] for r in results]

    axes[0].bar(names, vrams, color=["#3498db", "#9b59b6"], width=0.5, edgecolor="black")
    axes[0].set_ylabel("Peak VRAM (GB)")
    axes[0].set_title("Peak VRAM Consumption (6GB Budget)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, v in enumerate(vrams):
        axes[0].text(i, v * 0.5, f"{v:.2f} GB", ha="center", va="center", color="white", fontweight="bold")

    axes[1].bar(names, latencies, color=["#3498db", "#9b59b6"], width=0.5, edgecolor="black")
    axes[1].set_ylabel("Step Latency (ms/step)")
    axes[1].set_title("Training Step Latency")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, l in enumerate(latencies):
        axes[1].text(i, l * 0.5, f"{l:.1f} ms", ha="center", va="center", color="white", fontweight="bold")

    axes[2].bar(names, fids, color=["#3498db", "#9b59b6"], width=0.5, edgecolor="black")
    axes[2].set_ylabel("Validation FID ↓")
    axes[2].set_title("Validation FID Score")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for i, f in enumerate(fids):
        axes[2].text(i, f * 0.5, f"{f:.1f}", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "e4_capacity_tradeoffs.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll capacity results saved to {out_dir}")

if __name__ == "__main__":
    main()
