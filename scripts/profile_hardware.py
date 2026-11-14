"""Hardware Profiling and Stability Test Suite for StyleForge Phase 2.

Profiles:
- Baseline WGAN-GP vs StyleForge AdaIN
- Batch sizes (16, 32)
- FP32 vs AMP (FP16)
- Peak VRAM allocation & throughput (steps/sec)
- Gradient penalty & R1 computation stability
- Memory leak detection across consecutive iterations
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from models.wgan_baseline import WGANBaselineGenerator, WGANBaselineCritic
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.losses import compute_gradient_penalty, compute_r1_penalty
from data.dataset import get_dataloader


def profile_configuration(
    model_type: str = "styleforge",
    batch_size: int = 16,
    use_amp: bool = False,
    steps: int = 50,
    device: str = "cuda",
) -> dict:
    if not torch.cuda.is_available():
        device = "cpu"

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    latent_dim = 512
    img_size = 64

    # Build models
    if model_type == "baseline":
        generator = WGANBaselineGenerator(latent_dim=128, img_size=img_size).to(device)
        critic = WGANBaselineCritic(img_size=img_size).to(device)
        g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.9))
        c_opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.0, 0.9))
        gen_z_fn = lambda b: torch.randn(b, 128, device=device)
    else:
        generator = StyleGenerator(latent_dim=latent_dim, img_size=img_size).to(device)
        critic = StyleCritic(img_size=img_size).to(device)
        g_opt = torch.optim.Adam(generator.parameters(), lr=1.5e-4, betas=(0.0, 0.99))
        c_opt = torch.optim.Adam(critic.parameters(), lr=1.5e-4, betas=(0.0, 0.99))
        gen_z_fn = lambda b: torch.randn(b, latent_dim, device=device)

    dev_str = "cuda" if "cuda" in device else "cpu"
    scaler_c = torch.amp.GradScaler(dev_str, enabled=use_amp)
    scaler_g = torch.amp.GradScaler(dev_str, enabled=use_amp)

    loader = get_dataloader(
        dataset_path="data/celeba/img_align_celeba",
        split="train",
        img_size=img_size,
        batch_size=batch_size,
        num_workers=0,
    )
    data_iter = iter(loader)

    nan_detected = False
    vram_samples = []

    # Warmup
    for _ in range(5):
        try:
            real_batch = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(loader)
            real_batch = next(data_iter).to(device)

        z = gen_z_fn(batch_size)
        with torch.amp.autocast(dev_str, enabled=use_amp):
            fake = generator(z)
            c_real = critic(real_batch)
            c_fake = critic(fake.detach())

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for step in range(steps):
        try:
            real_batch = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(loader)
            real_batch = next(data_iter).to(device)

        c_opt.zero_grad(set_to_none=True)
        z = gen_z_fn(batch_size)

        with torch.amp.autocast(dev_str, enabled=use_amp):
            fake = generator(z)
            c_real = critic(real_batch)
            c_fake = critic(fake.detach())

            if model_type == "baseline":
                d_loss = c_fake.mean() - c_real.mean()
                gp, _ = compute_gradient_penalty(critic, real_batch, fake.detach(), gp_weight=10.0)
                total_c_loss = d_loss + gp
            else:
                d_loss = nn.functional.softplus(c_fake).mean() + nn.functional.softplus(-c_real).mean()
                r1, _ = compute_r1_penalty(critic, real_batch, r1_gamma=10.0)
                total_c_loss = d_loss + r1

        scaler_c.scale(total_c_loss).backward()
        scaler_c.step(c_opt)
        scaler_c.update()

        # Generator step
        g_opt.zero_grad(set_to_none=True)
        z2 = gen_z_fn(batch_size)
        with torch.amp.autocast(dev_str, enabled=use_amp):
            fake2 = generator(z2)
            c_fake2 = critic(fake2)
            if model_type == "baseline":
                g_loss = -c_fake2.mean()
            else:
                g_loss = nn.functional.softplus(-c_fake2).mean()

        scaler_g.scale(g_loss).backward()
        scaler_g.step(g_opt)
        scaler_g.update()

        # Check NaNs
        if torch.isnan(total_c_loss) or torch.isnan(g_loss):
            nan_detected = True
            break

        if step % 10 == 0:
            vram_samples.append(torch.cuda.memory_allocated() / (1024 ** 2))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    fps = (steps * batch_size) / elapsed
    it_per_sec = steps / elapsed

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # Memory growth between step 10 and end
    mem_leak_indicator = (vram_samples[-1] - vram_samples[0]) if len(vram_samples) > 1 else 0.0

    return {
        "model": model_type,
        "batch_size": batch_size,
        "use_amp": use_amp,
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "steps_per_sec": round(it_per_sec, 2),
        "samples_per_sec": round(fps, 2),
        "elapsed_sec": round(elapsed, 2),
        "nan_detected": nan_detected,
        "memory_drift_mb": round(mem_leak_indicator, 2),
        "stable": (not nan_detected) and (peak_reserved_mb < 5500),
    }


def main():
    print("=" * 65)
    print("StyleForge Phase 2: Hardware Profiling on RTX 4050 Laptop GPU")
    print("=" * 65)

    experiments = [
        {"model_type": "baseline", "batch_size": 32, "use_amp": False},
        {"model_type": "baseline", "batch_size": 32, "use_amp": True},
        {"model_type": "styleforge", "batch_size": 16, "use_amp": False},
        {"model_type": "styleforge", "batch_size": 16, "use_amp": True},
        {"model_type": "styleforge", "batch_size": 32, "use_amp": False},
    ]

    results = []
    for exp in experiments:
        print(f"\nProfiling {exp['model_type'].upper()} | Batch: {exp['batch_size']} | AMP: {exp['use_amp']}...")
        res = profile_configuration(**exp, steps=30)
        print(f"  Peak VRAM: {res['peak_allocated_mb']} MB (Reserved: {res['peak_reserved_mb']} MB)")
        print(f"  Throughput: {res['steps_per_sec']} it/s ({res['samples_per_sec']} samples/s)")
        print(f"  NaN Detected: {res['nan_detected']} | Stable: {res['stable']} | Mem Drift: {res['memory_drift_mb']} MB")
        results.append(res)

    out_file = Path("docs/hardware_profile.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nProfiling report saved to {out_file}")


if __name__ == "__main__":
    main()
