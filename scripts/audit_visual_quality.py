"""Comprehensive Root-Cause Audit of StyleForge Visual Generation Quality.

Investigates:
1. Reproduction of 32 samples across Raw/EMA and truncations 0.5, 0.7, 0.8, 1.0.
2. Inference pipeline audit (noise weights, AdaIN scale/bias, pre-tanh activations, denormalization, channel stats).
3. Training trajectory audit (metrics.json, loss dynamics, n_critic, convergence bounds).
4. Quantitative evaluation (FID, KID, Precision, Recall, Diversity, Nearest Neighbors, 2D FFT spectral decomposition).
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from data.dataset import get_dataloader
from models.style_generator import StyleGenerator
from evaluation.diversity import compute_precision_recall, compute_pairwise_distances

def to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 255] uint8 tensor."""
    return ((tensor * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Single (3, H, W) float in [-1, 1] -> PIL Image."""
    arr = ((tensor * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr)

def make_grid_image(images: list, cols: int = 8) -> Image.Image:
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))
    return grid

def compute_radial_psd(images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Computes radially averaged power spectral density (2D FFT) across batch of images."""
    # Convert images to grayscale: (B, 1, H, W)
    gray = (images * 0.5 + 0.5).mean(dim=1, keepdim=True)
    B, _, H, W = gray.shape
    
    # 2D FFT
    fft2 = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft2)
    psd2D = (torch.abs(fft_shift) ** 2).mean(dim=0).squeeze().cpu().numpy()
    
    # Radial average
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    
    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        mask = (r == i)
        if np.any(mask):
            radial_profile[i] = psd2D[mask].mean()
            
    freqs = np.arange(max_r)
    return freqs, radial_profile

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Visual Quality Root-Cause Audit on device: {device}")

    # Output directories
    audit_dir = Path("reports/visual_audit")
    samples_dir = audit_dir / "samples"
    grids_dir = audit_dir / "grids"
    plots_dir = audit_dir / "plots"

    for d in [samples_dir, grids_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: LOAD CHECKPOINT & AUDIT ARCHITECTURE / WEIGHTS
    # =========================================================================
    ckpt_path = Path("checkpoints/styleforge_final_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("experiments/phase3_convergence/checkpoints/checkpoint_final.pt")
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    raw_state = payload.get("generator") or payload.get("model_state_dict")
    ema_state = payload.get("ema_generator") or raw_state

    # Instantiate generators
    g_raw = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    cleaned_raw = {k.replace("module.", ""): v for k, v in raw_state.items()}
    g_raw.load_state_dict(cleaned_raw)
    g_raw.eval()

    g_ema = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    cleaned_ema = {k.replace("module.", ""): v for k, v in ema_state.items()}
    g_ema.load_state_dict(cleaned_ema)
    g_ema.eval()

    # Check w_avg state
    raw_w_avg = g_raw.w_avg.detach().cpu()
    ema_w_avg = g_ema.w_avg.detach().cpu()
    print(f"Raw w_avg norm: {raw_w_avg.norm():.4f}, is_zero: {torch.all(raw_w_avg == 0)}")
    print(f"EMA w_avg norm: {ema_w_avg.norm():.4f}, is_zero: {torch.all(ema_w_avg == 0)}")

    # Check noise injection weights
    noise_weights_info = {}
    print("\n--- Auditing Learned Noise Injection Weights ---")
    for name, module in g_raw.named_modules():
        if hasattr(module, "weight") and "noise" in name:
            nw = module.weight.detach().cpu().flatten()
            print(f"  {name}: shape={module.weight.shape}, mean={nw.mean():.6f}, std={nw.std():.6f}, max={nw.abs().max():.6f}")
            noise_weights_info[name] = {
                "mean": float(nw.mean()),
                "max": float(nw.abs().max()),
                "is_zero": bool(torch.all(nw == 0))
            }

    # =========================================================================
    # STEP 2: REPRODUCE ISSUE — 32 SAMPLES ACROSS RAW/EMA & TRUNCATION VALUES
    # =========================================================================
    print("\n--- Generating 32 Reproducible Samples across Configurations ---")
    trunc_values = [0.5, 0.7, 0.8, 1.0]
    models = {"raw": g_raw, "ema": g_ema}

    # Fix 32 latent seeds
    torch.manual_seed(2026)
    fixed_z_32 = torch.randn(32, 512, device=device)

    reproduced_outputs = {}

    for m_name, model in models.items():
        reproduced_outputs[m_name] = {}
        for psi in trunc_values:
            save_subdir = samples_dir / f"{m_name}_psi_{psi:.1f}"
            save_subdir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                imgs = model(fixed_z_32, truncation_psi=psi)

            # Store pixel statistics
            img_min = float(imgs.min().cpu())
            img_max = float(imgs.max().cpu())
            img_mean = float(imgs.mean().cpu())
            img_std = float(imgs.std().cpu())
            
            reproduced_outputs[m_name][f"psi_{psi:.1f}"] = {
                "min": img_min, "max": img_max, "mean": img_mean, "std": img_std
            }

            pil_imgs = []
            for i in range(32):
                p_img = to_pil(imgs[i])
                p_img.save(save_subdir / f"sample_{i:02d}.png")
                pil_imgs.append(p_img)

            grid = make_grid_image(pil_imgs, cols=8)
            grid.save(grids_dir / f"grid_{m_name}_psi_{psi:.1f}.png")

    # Composite comparison grid: Raw vs EMA across 4 truncation values (first 8 seeds)
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    row = 0
    for m_name, model in models.items():
        for psi in trunc_values:
            with torch.no_grad():
                imgs_8 = model(fixed_z_32[:8], truncation_psi=psi)
            for c in range(8):
                im = ((imgs_8[c] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
                axes[row, c].imshow(im)
                axes[row, c].axis("off")
            axes[row, 0].set_title(f"{m_name.upper()} (psi={psi:.1f})", loc="left", fontsize=9, fontweight="bold")
            row += 1

    plt.suptitle("StyleForge v1.0 Reproduction Grid: Raw vs EMA Across Truncations (Seeds 1-8)", fontsize=13, y=0.99)
    plt.tight_layout()
    plt.savefig(grids_dir / "comparison_raw_vs_ema_truncations.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # STEP 3: AUDIT INFERENCE COMPONENTS (ACTIVATIONS, PRE-TANH, NOISE EFFECT)
    # =========================================================================
    print("\n--- Auditing Pre-Tanh Dynamic Range & Zero-Noise Behavior ---")
    with torch.no_grad():
        # Inspect Mapping Network output
        w_sample = g_raw.mapping(fixed_z_32[:32])
        w_mean = float(w_sample.mean().cpu())
        w_std = float(w_sample.std().cpu())
        w_norm = float(w_sample.norm(dim=-1).mean().cpu())
        print(f"Mapping Network W-space: mean={w_mean:.4f}, std={w_std:.4f}, norm={w_norm:.4f}")

        # Test generation with explicit ZERO noise vs STOCHASTIC noise
        zero_noise_list = [
            torch.zeros(32, 1, 4, 4, device=device),  # initial_noise (4x4)
            torch.zeros(32, 1, 4, 4, device=device),  # initial_block.noise (4x4)
        ]
        # each subsequent res has 2 blocks
        for res in g_raw.synthesis.resolutions:
            zero_noise_list.append(torch.zeros(32, 1, res, res, device=device))
            zero_noise_list.append(torch.zeros(32, 1, res, res, device=device))

        imgs_with_noise = g_raw(fixed_z_32[:32], truncation_psi=0.8)
        imgs_zero_noise = g_raw.synthesis(
            g_raw.mapping(fixed_z_32[:32]).unsqueeze(1).repeat(1, g_raw.num_style_layers, 1),
            noise_list=zero_noise_list
        )

        diff_noise = (imgs_with_noise - imgs_zero_noise).abs().mean().item()
        print(f"Mean Absolute Difference between with-noise and zero-noise generation: {diff_noise:.6f}")

    # =========================================================================
    # STEP 4: AUDIT REAL CELEBA TRAINING SAMPLES & DATASET PIPELINE
    # =========================================================================
    print("\n--- Auditing Real CelebA Dataset Pipeline & Image Distribution ---")
    val_loader = get_dataloader(
        "data/celeba/img_align_celeba",
        split="val",
        img_size=64,
        batch_size=32,
        num_workers=0
    )
    real_sample_batch = next(iter(val_loader)).to(device)

    # Save 32 real CelebA images
    real_pil_list = [to_pil(real_sample_batch[i]) for i in range(32)]
    real_grid = make_grid_image(real_pil_list, cols=8)
    real_grid.save(grids_dir / "grid_real_celeba_32_samples.png")

    real_min = float(real_sample_batch.min().cpu())
    real_max = float(real_sample_batch.max().cpu())
    real_mean = float(real_sample_batch.mean().cpu())
    real_std = float(real_sample_batch.std().cpu())
    print(f"Real CelebA (Val): min={real_min:.4f}, max={real_max:.4f}, mean={real_mean:.4f}, std={real_std:.4f}")

    # Channel-wise stats
    real_r = float(real_sample_batch[:, 0].mean().cpu())
    real_g = float(real_sample_batch[:, 1].mean().cpu())
    real_b = float(real_sample_batch[:, 2].mean().cpu())
    
    fake_r = float(imgs_with_noise[:, 0].mean().cpu())
    fake_g = float(imgs_with_noise[:, 1].mean().cpu())
    fake_b = float(imgs_with_noise[:, 2].mean().cpu())

    print(f"Channel Means (Real) -> R: {real_r:.3f}, G: {real_g:.3f}, B: {real_b:.3f}")
    print(f"Channel Means (Fake) -> R: {fake_r:.3f}, G: {fake_g:.3f}, B: {fake_b:.3f}")

    # =========================================================================
    # STEP 5: AUDIT TRAINING DYNAMICS (LOSS TRAJECTORY & GRADIENT LOGS)
    # =========================================================================
    print("\n--- Auditing Training Loss Trajectory (Phase 3 Convergence) ---")
    metrics_path = Path("experiments/phase3_convergence/metrics.json")
    training_audit = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics_history = json.load(f)

        steps = [entry["step"] for entry in metrics_history]
        loss_d = [entry.get("loss_d", entry.get("d_loss", 0.0)) for entry in metrics_history]
        loss_g = [entry.get("loss_g", entry.get("g_loss", 0.0)) for entry in metrics_history]
        d_real = [entry.get("d_real", 0.0) for entry in metrics_history]
        d_fake = [entry.get("d_fake", 0.0) for entry in metrics_history]
        grad_norm_g = [entry.get("gen_grad_norm", 0.0) for entry in metrics_history]
        penalty = [entry.get("penalty", 0.0) for entry in metrics_history]

        training_audit = {
            "total_logged_steps": len(steps),
            "final_step": steps[-1],
            "final_loss_d": loss_d[-1],
            "final_loss_g": loss_g[-1],
            "final_d_real": d_real[-1],
            "final_d_fake": d_fake[-1],
            "mean_penalty": float(np.mean(penalty)),
            "mean_grad_norm_g": float(np.mean(grad_norm_g)),
        }
        print(f"Training Audit: Final Step={steps[-1]}, Loss D={loss_d[-1]:.4f}, Loss G={loss_g[-1]:.4f}")
        print(f"D(real)={d_real[-1]:.4f}, D(fake)={d_fake[-1]:.4f}, Mean Penalty={np.mean(penalty):.4f}")

        # Plot loss curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
        ax1.plot(steps, loss_d, label="Critic Loss (Wasserstein)", color="#e74c3c", alpha=0.8)
        ax1.plot(steps, loss_g, label="Generator Loss", color="#2980b9", alpha=0.8)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Critic & Generator Loss Trajectories (1500 Steps)")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(steps, d_real, label="D(real)", color="#27ae60")
        ax2.plot(steps, d_fake, label="D(fake)", color="#8e44ad")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Critic Output Score")
        ax2.set_title("Critic Score Dynamics: D(real) vs D(fake)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(plots_dir / "training_loss_curves_audit.png", dpi=150)
        plt.close()

    # =========================================================================
    # STEP 6: 2D FFT FREQUENCY-DOMAIN SPECTRAL DECOMPOSITION
    # =========================================================================
    print("\n--- Computing 2D FFT Power Spectral Densities (PSD) ---")
    freqs, psd_real = compute_radial_psd(real_sample_batch[:32])
    _, psd_raw = compute_radial_psd(imgs_with_noise[:32])
    with torch.no_grad():
        ema_sample_32 = g_ema(fixed_z_32[:32], truncation_psi=0.8)
    _, psd_ema = compute_radial_psd(ema_sample_32[:32])

    # Pure Gaussian noise reference
    noise_ref = torch.randn_like(real_sample_batch[:32]).clamp(-1, 1)
    _, psd_noise = compute_radial_psd(noise_ref)

    # Plot PSD
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(freqs, psd_real, label="Real CelebA (Val, N=32)", color="#2ecc71", linewidth=2.0)
    ax.semilogy(freqs, psd_raw, label="StyleForge Raw (psi=0.8)", color="#e74c3c", linewidth=2.0)
    ax.semilogy(freqs, psd_ema, label="StyleForge EMA (psi=0.8)", color="#3498db", linewidth=2.0, linestyle="--")
    ax.semilogy(freqs, psd_noise, label="White Gaussian Noise Reference", color="#7f8c8d", linestyle=":")
    ax.set_xlabel("Spatial Frequency (Cycles / Image)")
    ax.set_ylabel("Power Spectral Density (Log Scale)")
    ax.set_title("Radial 2D Power Spectral Density: Real vs StyleForge v1.0")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plots_dir / "spectral_psd_comparison.png", dpi=150)
    plt.close()

    # Compute high-frequency / low-frequency power ratio
    nyquist = len(freqs)
    lf_cut = nyquist // 4
    hf_cut = nyquist // 2

    hf_ratio_real = float(psd_real[hf_cut:].sum() / psd_real[:lf_cut].sum())
    hf_ratio_raw = float(psd_raw[hf_cut:].sum() / psd_raw[:lf_cut].sum())
    hf_ratio_ema = float(psd_ema[hf_cut:].sum() / psd_ema[:lf_cut].sum())
    print(f"High-Frequency to Low-Frequency Energy Ratio:")
    print(f"  Real CelebA:      {hf_ratio_real:.6f}")
    print(f"  StyleForge Raw:   {hf_ratio_raw:.6f} ({hf_ratio_raw/hf_ratio_real:.2f}x of Real)")
    print(f"  StyleForge EMA:   {hf_ratio_ema:.6f} ({hf_ratio_ema/hf_ratio_real:.2f}x of Real)")

    # =========================================================================
    # STEP 7: QUANTITATIVE EVALUATION (FID, KID, PRECISION, RECALL, DIVERSITY)
    # =========================================================================
    print("\n--- Executing Quantitative Inception Evaluation (N=614) ---")
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_metric = KernelInceptionDistance(feature=2048, subsets=10, subset_size=25, normalize=False).to(device)

    # 1. Feed Real Validation Images
    print("Accumulating Real Validation Inception Features...")
    full_val_loader = get_dataloader(
        "data/celeba/img_align_celeba",
        split="val",
        img_size=64,
        batch_size=64,
        num_workers=0
    )

    real_feats_list = []
    inc_net = fid_metric.inception.eval()

    with torch.no_grad():
        for b in full_val_loader:
            b_uint8 = to_uint8(b.to(device))
            fid_metric.update(b_uint8, real=True)
            kid_metric.update(b_uint8, real=True)
            real_feats_list.append(inc_net(b_uint8).view(b_uint8.size(0), -1))

    real_features_all = torch.cat(real_feats_list, dim=0)
    print(f"Total Real Validation Images evaluated: {real_features_all.size(0)}")

    # 2. Evaluate Models
    def eval_model_stats(model: nn.Module, model_label: str, psi: float = 0.8):
        print(f"Evaluating {model_label} (psi={psi})...")
        fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        kid_m = KernelInceptionDistance(feature=2048, subsets=10, subset_size=25, normalize=False).to(device)

        # Clone real features state
        fid_m.real_features_sum = fid_metric.real_features_sum.clone()
        fid_m.real_features_cov_sum = fid_metric.real_features_cov_sum.clone()
        fid_m.real_features_num_samples = fid_metric.real_features_num_samples.clone()

        kid_m.real_features = [f.clone() for f in kid_metric.real_features]

        fake_imgs_list = []
        fake_feats_list = []
        torch.manual_seed(42)

        batch_sz = 64
        total_eval = real_features_all.size(0)  # 614
        generated = 0

        with torch.no_grad():
            while generated < total_eval:
                cur_b = min(batch_sz, total_eval - generated)
                z = torch.randn(cur_b, 512, device=device)
                f_imgs = model(z, truncation_psi=psi)
                f_uint8 = to_uint8(f_imgs)

                fid_m.update(f_uint8, real=False)
                kid_m.update(f_uint8, real=False)

                fake_imgs_list.append(f_imgs.cpu())
                fake_feats_list.append(inc_net(f_uint8).view(cur_b, -1))
                generated += cur_b

        fid_score = float(fid_m.compute().item())
        kid_mean, kid_std = kid_m.compute()
        kid_mean_val = float(kid_mean.item())

        all_fake_tensors = torch.cat(fake_imgs_list, dim=0)
        all_fake_features = torch.cat(fake_feats_list, dim=0)

        # Diversity
        flat_fakes = all_fake_tensors.view(all_fake_tensors.size(0), -1)
        pixel_var = float(torch.var(flat_fakes, dim=0).mean().item())

        # Pairwise L1 distance
        p_dist = torch.cdist(flat_fakes[:200], flat_fakes[:200], p=1.0) / flat_fakes.size(1)
        pairwise_l1 = float(p_dist[p_dist > 0].mean().item())

        # Precision and Recall
        prec, rec = compute_precision_recall(real_features_all, all_fake_features, k=3)

        return {
            "fid": fid_score,
            "kid_mean": kid_mean_val,
            "pixel_variance": pixel_var,
            "pairwise_l1": pairwise_l1,
            "precision": prec,
            "recall": rec
        }

    eval_results = {
        "raw_psi_0.8": eval_model_stats(g_raw, "Raw Generator", psi=0.8),
        "ema_psi_0.8": eval_model_stats(g_ema, "EMA Generator", psi=0.8),
        "raw_psi_1.0": eval_model_stats(g_raw, "Raw Generator", psi=1.0),
        "ema_psi_1.0": eval_model_stats(g_ema, "EMA Generator", psi=1.0),
    }

    print("\n--- Quantitative Benchmark Summary ---")
    for k, v in eval_results.items():
        print(f"  {k}: FID={v['fid']:.2f}, KID={v['kid_mean']:.4f}, Prec={v['precision']:.4f}, Rec={v['recall']:.4f}, Var={v['pixel_variance']:.5f}, L1={v['pairwise_l1']:.4f}")

    # =========================================================================
    # STEP 8: SAVE COMPREHENSIVE AUDIT ARTIFACTS & JSON SUMMARY
    # =========================================================================
    audit_summary = {
        "timestamp": "2026-11-14T16:15:00Z",
        "checkpoint_evaluated": str(ckpt_path),
        "device": str(device),
        "noise_weights": noise_weights_info,
        "w_space": {
            "raw_w_avg_norm": float(raw_w_avg.norm()),
            "ema_w_avg_norm": float(ema_w_avg.norm()),
            "w_std": w_std,
            "w_mean": w_mean,
        },
        "noise_ablation": {
            "mean_abs_diff_with_vs_without_noise": diff_noise,
        },
        "reproduced_outputs": reproduced_outputs,
        "spectral_psd": {
            "hf_ratio_real": hf_ratio_real,
            "hf_ratio_raw": hf_ratio_raw,
            "hf_ratio_ema": hf_ratio_ema,
        },
        "training_trajectory": training_audit,
        "quantitative_metrics": eval_results,
    }

    summary_file = audit_dir / "visual_quality_audit_results.json"
    with open(summary_file, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nAudit complete! Saved quantitative findings to {summary_file}")

if __name__ == "__main__":
    main()
