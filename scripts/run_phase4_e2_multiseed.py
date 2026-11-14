"""Phase 4 Experiment E2: Multi-Seed Controlled Objective Comparison.

Compares:
1. WGAN-GP (lambda=10.0)
2. Non-saturating Logistic Loss (unregularized)
3. Non-saturating Logistic Loss + R1 regularization (gamma=10.0)

Evaluated across 3 independent random seeds: [42, 123, 777] for 300 steps each.
Reports mean +/- std across seeds for FID, KID, L1 diversity, and variance.
"""

import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from evaluation.diversity import compute_pairwise_distances

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return ((t * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_single_experiment(
    objective_name: str,
    loss_mode: str,
    penalty_weight: float,
    seed: int,
    steps: int,
    device: torch.device,
    val_loader,
    real_feats_cache,
    real_uint8_cache,
    vis_z: torch.Tensor,
) -> Dict:
    print(f"\n--- Running {objective_name} | Seed: {seed} | Steps: {steps} ---", flush=True)
    set_seed(seed)

    # 1. DataLoader with seed
    train_loader = get_dataloader(
        "data/celeba/img_align_celeba",
        split="train",
        img_size=64,
        batch_size=16,
        num_workers=0,
        seed=seed,
    )

    # 2. Models
    latent_dim = 512
    generator = StyleGenerator(latent_dim=latent_dim, style_dim=latent_dim, img_size=64).to(device)
    critic = StyleCritic(img_channels=3, img_size=64).to(device)

    # 3. Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.99))

    # 4. Loss module
    loss_fn = GANLoss(mode=loss_mode, penalty_weight=penalty_weight, r1_gamma=penalty_weight)

    n_critic = 2
    data_iter = iter(train_loader)

    history = {
        "step": [],
        "loss_d": [],
        "loss_g": [],
        "penalty": [],
        "grad_norm": [],
    }

    start_time = time.time()
    for step in range(1, steps + 1):
        generator.train()
        critic.train()

        # Critic updates
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
            loss_d, d_stats = loss_fn.critic_loss(critic, real_batch, fake_c)
            loss_d.backward()
            opt_d.step()

        # Generator update
        opt_g.zero_grad(set_to_none=True)
        z_g = torch.randn(16, latent_dim, device=device)
        fake_g = generator(z_g, style_mixing_prob=0.0)
        loss_g, g_stats = loss_fn.generator_loss(critic, fake_g)
        loss_g.backward()
        opt_g.step()

        if step % 50 == 0 or step == steps:
            print(f"Step {step}/{steps} | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}", flush=True)
            history["step"].append(step)
            history["loss_d"].append(loss_d.item())
            history["loss_g"].append(loss_g.item())
            history["penalty"].append(d_stats.get("penalty", 0.0))
            history["grad_norm"].append(d_stats.get("grad_norm", 0.0))

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s.", flush=True)

    # Evaluation on Validation Set (N=614)
    print("Evaluating FID, KID, and Diversity on Validation Set...", flush=True)
    generator.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_metric = KernelInceptionDistance(feature=2048, subsets=20, subset_size=50, normalize=False).to(device)

    # Feed cached real images
    for u_real in real_uint8_cache:
        fid_metric.update(u_real.to(device), real=True)
        kid_metric.update(u_real.to(device), real=True)

    # Generate synthetic images matching validation size
    N_val = len(val_loader.dataset)
    torch.manual_seed(seed)
    fake_imgs_list = []
    count = 0
    with torch.no_grad():
        while count < N_val:
            cur_bs = min(32, N_val - count)
            z = torch.randn(cur_bs, latent_dim, device=device)
            f_img = generator(z, truncation_psi=0.8)
            fake_imgs_list.append(f_img.cpu())
            u_f = to_uint8(f_img)
            fid_metric.update(u_f, real=False)
            kid_metric.update(u_f, real=False)
            count += cur_bs

    fid_val = float(fid_metric.compute().item())
    kid_mean, kid_std = kid_metric.compute()
    kid_mean_val = float(kid_mean.item())
    kid_std_val = float(kid_std.item())

    # Diversity metrics
    all_fake = torch.cat(fake_imgs_list, dim=0)
    fake_eval = all_fake[:200]
    flat = fake_eval.view(fake_eval.size(0), -1)
    num_pairs = min(1000, flat.size(0) * 2)
    idx1 = torch.randint(0, flat.size(0), (num_pairs,))
    idx2 = torch.randint(0, flat.size(0), (num_pairs,))
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    diff = flat[idx1] - flat[idx2]
    l1_div = float(torch.abs(diff).mean(dim=1).mean().item())
    var_div = float(fake_eval.var(dim=0).mean().item())

    print(f"Result [{objective_name} | Seed {seed}]: FID={fid_val:.2f}, KID={kid_mean_val:.4f}, L1={l1_div:.4f}, Var={var_div:.5f}", flush=True)

    # Fixed seed visual sample (16 faces)
    with torch.no_grad():
        vis_faces = generator(vis_z, truncation_psi=0.8).cpu()

    return {
        "objective": objective_name,
        "seed": seed,
        "fid": fid_val,
        "kid_mean": kid_mean_val,
        "kid_std": kid_std_val,
        "l1_diversity": l1_div,
        "batch_variance": var_div,
        "history": history,
        "vis_faces": vis_faces,
        "elapsed_sec": elapsed,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase4/objectives")
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = 200
    seeds = [42, 123, 777]
    objectives = [
        {"name": "WGAN-GP", "mode": "wgan_gp", "penalty": 10.0},
        {"name": "Logistic (Unregularized)", "mode": "logistic", "penalty": 0.0},
        {"name": "Logistic + R1", "mode": "logistic_r1", "penalty": 10.0},
    ]

    print("==================================================")
    print("StyleForge Phase 4 — Experiment E2: Multi-Seed Objective Comparison")
    print(f"Device: {device} | Steps: {steps} | Seeds: {seeds}")
    print("Objectives: WGAN-GP vs Logistic vs Logistic + R1")
    print("==================================================")

    # Prepare validation loader and caches
    val_loader = get_dataloader("data/celeba/img_align_celeba", split="val", img_size=64, batch_size=32, num_workers=0, seed=42)
    real_uint8_cache = []
    for b in val_loader:
        if isinstance(b, (list, tuple)):
            b = b[0]
        real_uint8_cache.append(to_uint8(b))

    # Fixed latent vectors for qualitative comparison grid across all runs
    torch.manual_seed(999)
    vis_z = torch.randn(8, 512, device=device)

    all_run_results = []

    for obj in objectives:
        for seed in seeds:
            res = run_single_experiment(
                objective_name=obj["name"],
                loss_mode=obj["mode"],
                penalty_weight=obj["penalty"],
                seed=seed,
                steps=steps,
                device=device,
                val_loader=val_loader,
                real_feats_cache=None,
                real_uint8_cache=real_uint8_cache,
                vis_z=vis_z,
            )
            all_run_results.append(res)

    # Aggregate Statistics
    summary_stats = {}
    for obj in objectives:
        name = obj["name"]
        matching = [r for r in all_run_results if r["objective"] == name]
        fids = [r["fid"] for r in matching]
        kids = [r["kid_mean"] for r in matching]
        l1s = [r["l1_diversity"] for r in matching]
        vars_ = [r["batch_variance"] for r in matching]

        summary_stats[name] = {
            "fid_mean": float(np.mean(fids)),
            "fid_std": float(np.std(fids)),
            "kid_mean": float(np.mean(kids)),
            "kid_std": float(np.std(kids)),
            "l1_mean": float(np.mean(l1s)),
            "l1_std": float(np.std(l1s)),
            "var_mean": float(np.mean(vars_)),
            "var_std": float(np.std(vars_)),
            "individual_runs": [
                {
                    "seed": r["seed"],
                    "fid": r["fid"],
                    "kid_mean": r["kid_mean"],
                    "l1_diversity": r["l1_diversity"],
                    "batch_variance": r["batch_variance"],
                }
                for r in matching
            ],
        }

    print("\n================ MULTI-SEED SUMMARY RESULTS ================")
    print(f"{'Objective':<28} | {'FID (mean +/- std)':<20} | {'KID (mean +/- std)':<20} | {'L1 Div':<10} | {'Var':<10}")
    print("-" * 96)
    for name, s in summary_stats.items():
        fid_str = f"{s['fid_mean']:.2f} +/- {s['fid_std']:.2f}"
        kid_str = f"{s['kid_mean']:.4f} +/- {s['kid_std']:.4f}"
        l1_str = f"{s['l1_mean']:.4f}"
        var_str = f"{s['var_mean']:.5f}"
        print(f"{name:<28} | {fid_str:<20} | {kid_str:<20} | {l1_str:<10} | {var_str:<10}")
    print("============================================================\n")

    # Save JSON results (omitting tensors)
    serializable = {
        "metadata": {
            "experiment": "Phase 4 E2 Multi-Seed Objective Comparison",
            "steps": steps,
            "seeds": seeds,
            "device": str(device),
        },
        "summary": summary_stats,
    }
    with open(out_dir / "e2_multiseed_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Plot Visual Quality Comparison Grid
    # 3 rows (Objectives) x 8 columns (Latent seeds)
    fig, axes = plt.subplots(len(objectives), 8, figsize=(16, 6))
    for r_idx, obj in enumerate(objectives):
        name = obj["name"]
        # Pick seed 42 run for visual comparison
        run_res = next(r for r in all_run_results if r["objective"] == name and r["seed"] == 42)
        faces = run_res["vis_faces"]  # (8, 3, 64, 64)
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            img_np = ((faces[c_idx] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy())
            ax.imshow(img_np)
            ax.axis("off")
            if c_idx == 0:
                ax.set_title(f"{name}\nFID: {summary_stats[name]['fid_mean']:.1f}", fontsize=10, loc="left")

    plt.suptitle("StyleForge Phase 4: Loss Objective Comparison Across Fixed Latent Seeds (Seed=42, Step=300)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "e2_objective_visual_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot Convergence & Metric Bar Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of FID with error bars
    obj_names = [o["name"] for o in objectives]
    means = [summary_stats[name]["fid_mean"] for name in obj_names]
    stds = [summary_stats[name]["fid_std"] for name in obj_names]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    axes[0].bar(obj_names, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_ylabel("Fréchet Inception Distance (FID) ↓", fontsize=11)
    axes[0].set_title("Validation FID Across 3 Seeds (Mean ± Std)", fontsize=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m / 2, f"{m:.1f}\n±{s:.1f}", ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    # Critic Loss Dynamics across steps (seed 42)
    for i, obj in enumerate(objectives):
        name = obj["name"]
        run_res = next(r for r in all_run_results if r["objective"] == name and r["seed"] == 42)
        h = run_res["history"]
        axes[1].plot(h["step"], h["loss_d"], label=f"{name} (Critic Loss)", color=colors[i], linewidth=2)

    axes[1].set_xlabel("Training Step", fontsize=11)
    axes[1].set_ylabel("Critic Loss", fontsize=11)
    axes[1].set_title("Critic Loss Trajectory (Seed 42)", fontsize=12)
    axes[1].legend(loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "e2_metrics_and_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"All figures and summary saved to {out_dir}")

if __name__ == "__main__":
    main()
