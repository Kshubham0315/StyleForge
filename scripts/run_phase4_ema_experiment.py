"""Phase 4 Experiment E1: Systematic EMA Initialization, Warmup & Bias Correction Study."""

import copy
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
from tqdm import tqdm

from data.dataset import get_dataloader
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.losses import GANLoss
from training.trainer import ExponentialMovingAverage
from evaluation.diversity import compute_pairwise_distances, compute_precision_recall

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return ((t * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def run_experiment(steps: int = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase4/ema")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("StyleForge Phase 4 — Experiment E1: EMA Optimization")
    print(f"Device: {device} | Total Steps: {steps}")
    print("Conditions: Standard vs Debiased vs Zero-Init vs Warmup vs Raw")
    print("==================================================")

    # 1. DataLoader
    train_loader = get_dataloader("data/celeba/img_align_celeba", split="train", img_size=64, batch_size=16, num_workers=0, seed=42)
    val_loader = get_dataloader("data/celeba/img_align_celeba", split="val", img_size=64, batch_size=32, num_workers=0, seed=42)

    # 2. Models
    torch.manual_seed(42)
    generator = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    critic = StyleCritic(img_channels=3, img_size=64).to(device)

    # 3. EMA Conditions
    ema_std_0999 = ExponentialMovingAverage(generator, decay=0.999, use_warmup=False, bias_correction=False, init_mode="model")
    ema_debiased_0999 = ExponentialMovingAverage(generator, decay=0.999, use_warmup=False, bias_correction=True, init_mode="model")
    ema_zero_0999 = ExponentialMovingAverage(generator, decay=0.999, use_warmup=False, bias_correction=True, init_mode="zero")
    ema_warmup_0999 = ExponentialMovingAverage(generator, decay=0.999, use_warmup=True, bias_correction=False, init_mode="model")
    ema_warmup_0990 = ExponentialMovingAverage(generator, decay=0.990, use_warmup=True, bias_correction=False, init_mode="model")

    # 4. Optimizers & Loss
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.99))
    loss_fn = GANLoss(mode="wgan_gp", penalty_weight=10.0, r1_gamma=10.0)

    n_critic = 4
    latent_dim = 512
    data_iter = iter(train_loader)

    start_time = time.time()
    pbar = tqdm(total=steps, desc="Phase 4 E1 EMA Training")

    for step in range(1, steps + 1):
        for _ in range(n_critic):
            try:
                real_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                real_batch = next(data_iter)
            if isinstance(real_batch, (list, tuple)):
                real_batch = real_batch[0]
            real_batch = real_batch.to(device)

            opt_d.zero_grad(set_to_none=True)
            z_c = torch.randn(real_batch.size(0), latent_dim, device=device)
            fake_c = generator(z_c, style_mixing_prob=0.0, update_w_avg=True)
            loss_d, _ = loss_fn.critic_loss(critic, real_batch, fake_c)
            loss_d.backward()
            opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        z_g = torch.randn(16, latent_dim, device=device)
        fake_g = generator(z_g, style_mixing_prob=0.0)
        loss_g, _ = loss_fn.generator_loss(critic, fake_g)
        loss_g.backward()
        opt_g.step()

        has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in generator.parameters())
        if not has_nan:
            ema_std_0999.update(generator)
            ema_debiased_0999.update(generator)
            ema_zero_0999.update(generator)
            ema_warmup_0999.update(generator)
            ema_warmup_0990.update(generator)

        pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.2f} min).")

    # Extract evaluated models
    eval_variants = [
        ("Raw Instantaneous", generator),
        ("Standard EMA 0.999 (Uncorrected)", ema_std_0999.shadow),
        ("Debiased EMA 0.999 (Model-Init)", ema_debiased_0999.get_debiased_shadow()),
        ("Debiased EMA 0.999 (Zero-Init)", ema_zero_0999.get_debiased_shadow()),
        ("Warmup EMA 0.999 (Scheduled)", ema_warmup_0999.shadow),
        ("Warmup EMA 0.990 (Scheduled)", ema_warmup_0990.shadow),
    ]

    # Inception feature extractor
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    inc = fid_metric.inception.eval()

    # Pre-extract real validation features
    real_feats_list = []
    with torch.no_grad():
        for b in val_loader:
            if isinstance(b, (list, tuple)): b = b[0]
            real_feats_list.append(inc(to_uint8(b.to(device))).view(b.size(0), -1).cpu())
    real_feats = torch.cat(real_feats_list, dim=0)

    results = []
    fixed_samples = []
    torch.manual_seed(42)
    z_vis = torch.randn(16, latent_dim, device=device)
    N_val = len(val_loader.dataset)

    print("\nEvaluating all 6 conditions on authentic CelebA validation set (N=614)...")
    for label, model in eval_variants:
        model.eval()
        cur_fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        cur_kid = KernelInceptionDistance(feature=2048, subsets=20, subset_size=50, normalize=False).to(device)

        with torch.no_grad():
            for b in val_loader:
                if isinstance(b, (list, tuple)): b = b[0]
                u_b = to_uint8(b.to(device))
                cur_fid.update(u_b, real=True)
                cur_kid.update(u_b, real=True)

        torch.manual_seed(42)
        fake_imgs_list, fake_feats_list = [], []
        count = 0
        with torch.no_grad():
            while count < N_val:
                cur_bs = min(32, N_val - count)
                z = torch.randn(cur_bs, latent_dim, device=device)
                f_img = model(z, truncation_psi=0.8)
                fake_imgs_list.append(f_img.cpu())
                u_f = to_uint8(f_img)
                cur_fid.update(u_f, real=False)
                cur_kid.update(u_f, real=False)
                fake_feats_list.append(inc(u_f).view(cur_bs, -1).cpu())
                count += cur_bs

            vis_img = model(z_vis, truncation_psi=0.8)
            fixed_samples.append((label, (vis_img * 0.5 + 0.5).clamp(0, 1).cpu()))

        fake_feats = torch.cat(fake_feats_list, dim=0)
        all_fakes = torch.cat(fake_imgs_list, dim=0)
        flat = all_fakes.view(all_fakes.size(0), -1)

        fid_val = float(cur_fid.compute().item())
        kid_m, kid_s = cur_kid.compute()
        p, r = compute_precision_recall(real_feats, fake_feats, k=3)

        idx1 = torch.randint(0, N_val, (1000,))
        idx2 = torch.randint(0, N_val, (1000,))
        mask = idx1 != idx2
        diff = flat[idx1[mask]] - flat[idx2[mask]]
        l1_div = float(torch.abs(diff).mean(dim=1).mean().item())
        b_var = float(torch.var(flat, dim=0).mean().item())

        rec = {
            "condition": label,
            "fid": fid_val,
            "kid_mean": float(kid_m.item()),
            "kid_std": float(kid_s.item()),
            "precision": p,
            "recall": r,
            "pairwise_l1": l1_div,
            "batch_variance": b_var,
        }
        results.append(rec)
        print(f"[{label:35s}] FID: {fid_val:6.2f} | KID: {kid_m.item():.4f} | L1: {l1_div:.4f} | Var: {b_var:.5f}")

    with open(out_dir / "e1_ema_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot Visual Comparison Grid (6 conditions x 8 samples)
    fig, axes = plt.subplots(6, 8, figsize=(16, 12.5))
    for row_idx, (label, imgs) in enumerate(fixed_samples):
        for col_idx in range(8):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx].permute(1, 2, 0).numpy())
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(label, fontsize=10, fontweight="bold", loc="left", pad=10)

    plt.suptitle("StyleForge Phase 4 — Experiment E1: EMA Strategy & Bias Correction Comparison\n(Fixed Latents, 300 Training Steps, CelebA 64x64)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / "e1_ema_visual_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nVisual comparison saved to {out_dir / 'e1_ema_visual_comparison.png'}")

if __name__ == "__main__":
    run_experiment(steps=300)
