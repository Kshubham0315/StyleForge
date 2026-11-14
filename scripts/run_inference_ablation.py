"""Controlled Inference-Only Ablation Study for StyleForge v1.0.0.

Evaluates 6 configurations across Raw and EMA weights (12 conditions total):
- C1: Original stochastic noise, scale 1.0, no AdaIN clamp (Baseline)
- C2: Noise scale 0.5, no clamp
- C3: Noise scale 0.25, no clamp
- C4: Noise scale 0.0, no clamp
- C5: Noise scale 0.0, AdaIN scale clamp enabled
- C6: Noise scale 0.25, AdaIN scale clamp enabled

Measurements:
- FID and KID (official TorchMetrics Inception-v3 pool3 on N=614 validation set)
- Laplacian variance (edge sharpness / high-frequency metric)
- Radially averaged 2D FFT Power Spectral Density (PSD) high-to-low ratio
- Batch pixel variance (diversity / dynamic range metric)
- Pairwise L1 distance
- Inference latency (ms per face)
- Visual comparison grids and individual image samples
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
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
from evaluation.diversity import compute_precision_recall

def to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Convert float tensor in [-1, 1] to uint8 tensor in [0, 255]."""
    return ((tensor * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def to_pil(tensor: torch.Tensor) -> Image.Image:
    """Single (3, H, W) float tensor in [-1, 1] to PIL Image."""
    np_img = ((tensor * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(np_img)

def make_grid_image(images: List[Image.Image], cols: int = 8) -> Image.Image:
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))
    return grid

def compute_radial_psd(images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Computes radially averaged 2D FFT power spectral density (PSD)."""
    gray = (images * 0.5 + 0.5).mean(dim=1, keepdim=True)
    B, _, H, W = gray.shape
    fft2 = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft2)
    psd2D = (torch.abs(fft_shift) ** 2).mean(dim=0).squeeze().cpu().numpy()
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

def measure_laplacian_variance(images: torch.Tensor) -> Tuple[float, float]:
    """Computes mean and std of Laplacian variance across a batch of images in [-1, 1]."""
    u = ((images * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    vars_list = []
    for i in range(u.shape[0]):
        gray = cv2.cvtColor(u[i], cv2.COLOR_RGB2GRAY)
        vars_list.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.mean(vars_list)), float(np.std(vars_list))

def generate_controlled_sample(
    generator: StyleGenerator,
    z: torch.Tensor,
    truncation_psi: float = 0.8,
    noise_scale: float = 1.0,
    clamp_adain: bool = False,
    base_noise_list: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Generates synthetic images with explicit control over noise scale and AdaIN scale clamping."""
    batch_size = z.size(0)
    device = z.device

    # 1. Mapping network & truncation
    w = generator.mapping(z)
    styles = w.unsqueeze(1).repeat(1, generator.num_style_layers, 1)
    if truncation_psi < 1.0 and not torch.all(generator.w_avg == 0):
        w_avg = generator.w_avg.view(1, 1, -1)
        styles = w_avg + truncation_psi * (styles - w_avg)

    # 2. Build scaled noise list
    # Layers requiring noise: initial_noise (4x4), initial_block (4x4),
    # followed by res_8 (x2), res_16 (x2), res_32 (x2), res_64 (x2)
    synthesis = generator.synthesis
    noise_list = []

    if base_noise_list is not None:
        for b_noise in base_noise_list:
            if noise_scale == 0.0:
                noise_list.append(torch.zeros_like(b_noise))
            else:
                noise_list.append(b_noise * noise_scale)
    else:
        # Generate base noise if not supplied
        # 4x4 initial noise (two slots)
        for _ in range(2):
            if noise_scale == 0.0:
                noise_list.append(torch.zeros(batch_size, 1, 4, 4, device=device))
            else:
                noise_list.append(torch.randn(batch_size, 1, 4, 4, device=device) * noise_scale)
        # resolutions
        for res in synthesis.resolutions:
            for _ in range(2):
                if noise_scale == 0.0:
                    noise_list.append(torch.zeros(batch_size, 1, res, res, device=device))
                else:
                    noise_list.append(torch.randn(batch_size, 1, res, res, device=device) * noise_scale)

    # 3. Controlled synthesis pass
    # Helper to apply AdaIN with optional non-negative scale clamping
    def apply_adain(adain_module, feat, style_vec):
        mean = feat.mean(dim=[2, 3], keepdim=True)
        var = feat.var(dim=[2, 3], keepdim=True, unbiased=False)
        normalized = (feat - mean) / torch.sqrt(var + adain_module.eps)

        style = adain_module.style_mod(style_vec)
        scale, bias = style.chunk(2, dim=1)
        scale = (scale + 1.0).unsqueeze(-1).unsqueeze(-1)
        if clamp_adain:
            scale = torch.clamp(scale, min=0.0)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return normalized * scale + bias

    # 4x4 initial constant input
    x = synthesis.constant_input.repeat(batch_size, 1, 1, 1)
    x = synthesis.initial_noise(x, noise=noise_list[0])
    x = apply_adain(synthesis.initial_adain, x, styles[:, 0, :])

    # 4x4 initial block
    x = synthesis.initial_block.conv(x)
    x = synthesis.initial_block.noise(x, noise=noise_list[1])
    x = synthesis.initial_block.act(x)
    x = apply_adain(synthesis.initial_block.adain, x, styles[:, 1, :])

    style_idx = 2
    noise_idx = 2
    for res in synthesis.resolutions:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        b1 = synthesis.blocks[f"res_{res}_b1"]
        b2 = synthesis.blocks[f"res_{res}_b2"]

        # Block 1
        x = b1.conv(x)
        x = b1.noise(x, noise=noise_list[noise_idx])
        x = b1.act(x)
        x = apply_adain(b1.adain, x, styles[:, style_idx, :])

        # Block 2
        x = b2.conv(x)
        x = b2.noise(x, noise=noise_list[noise_idx + 1])
        x = b2.act(x)
        x = apply_adain(b2.adain, x, styles[:, style_idx + 1, :])

        style_idx += 2
        noise_idx += 2

        if f"res_{res}" in synthesis.attn:
            x = synthesis.attn[f"res_{res}"](x)

    return synthesis.to_rgb(x)

def build_fixed_base_noise(batch_size: int, device: torch.device, resolutions: List[int]) -> List[torch.Tensor]:
    """Generates fixed base noise tensors N(0, 1) for exact multi-scale ablation alignment."""
    noise_list = []
    # Two 4x4 slots
    noise_list.append(torch.randn(batch_size, 1, 4, 4, device=device))
    noise_list.append(torch.randn(batch_size, 1, 4, 4, device=device))
    for res in resolutions:
        noise_list.append(torch.randn(batch_size, 1, res, res, device=device))
        noise_list.append(torch.randn(batch_size, 1, res, res, device=device))
    return noise_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== StyleForge v1.0.0 Controlled Inference Ablation Study ===")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Output directories
    out_dir = Path("reports/visual_audit/ablation")
    samples_dir = out_dir / "samples"
    grids_dir = out_dir / "grids"
    plots_dir = out_dir / "plots"

    for d in [out_dir, samples_dir, grids_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load exact v1.0.0 final production checkpoint
    ckpt_path = Path("checkpoints/styleforge_final_model.pt")
    assert ckpt_path.exists(), f"Production checkpoint missing: {ckpt_path}"
    print(f"Loading checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw_state = payload.get("generator") or payload.get("model_state_dict")
    ema_state = payload.get("ema_generator") or raw_state

    raw_gen = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    raw_gen.load_state_dict({k.replace("module.", ""): v for k, v in raw_state.items()})
    raw_gen.eval()

    ema_gen = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    ema_gen.load_state_dict({k.replace("module.", ""): v for k, v in ema_state.items()})
    ema_gen.eval()

    # 2. Extract Real CelebA Validation Inception Features (N=614)
    print("\n--- Extracting Real Validation Inception Features (N=614) ---")
    val_loader = get_dataloader(
        "data/celeba/img_align_celeba",
        split="val",
        img_size=64,
        batch_size=64,
        num_workers=0
    )

    fid_template = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_template = KernelInceptionDistance(feature=2048, subsets=10, subset_size=25, normalize=False).to(device)
    inc_net = fid_template.inception.eval()

    real_imgs_all = []
    real_feats_all = []
    with torch.no_grad():
        for b in val_loader:
            b_dev = b.to(device)
            b_u8 = to_uint8(b_dev)
            fid_template.update(b_u8, real=True)
            kid_template.update(b_u8, real=True)
            real_imgs_all.append(b_dev)
            real_feats_all.append(inc_net(b_u8).view(b_u8.size(0), -1))

    real_imgs_all = torch.cat(real_imgs_all, dim=0)
    real_feats_all = torch.cat(real_feats_all, dim=0)
    num_val = real_imgs_all.size(0)
    print(f"Loaded {num_val} authentic CelebA validation images.")

    # Compute baseline metrics for Real CelebA
    freqs, psd_real = compute_radial_psd(real_imgs_all[:64])
    nyquist = len(freqs)
    lf_cut, hf_cut = nyquist // 4, nyquist // 2
    real_hf_ratio = float(psd_real[hf_cut:].sum() / psd_real[:lf_cut].sum())
    real_lap_mean, real_lap_std = measure_laplacian_variance(real_imgs_all[:64])
    real_flat = real_imgs_all.view(num_val, -1)
    real_pixel_var = float(torch.var(real_flat, dim=0).mean().item())

    print(f"Real CelebA Baseline:")
    print(f"  Laplacian Variance: {real_lap_mean:.2f} +- {real_lap_std:.2f}")
    print(f"  HF-to-LF PSD Ratio: {real_hf_ratio:.6f}")
    print(f"  Pixel Variance:     {real_pixel_var:.5f}")

    # 3. Define Configurations
    configs = [
        {"id": "C1", "name": "Stochastic Noise 1.0 (Baseline)", "noise_scale": 1.0, "clamp_adain": False},
        {"id": "C2", "name": "Noise Scale 0.5",                "noise_scale": 0.5, "clamp_adain": False},
        {"id": "C3", "name": "Noise Scale 0.25",               "noise_scale": 0.25, "clamp_adain": False},
        {"id": "C4", "name": "Noise Scale 0.0 (Denoised)",     "noise_scale": 0.0, "clamp_adain": False},
        {"id": "C5", "name": "Noise 0.0 + AdaIN Clamp",        "noise_scale": 0.0, "clamp_adain": True},
        {"id": "C6", "name": "Noise 0.25 + AdaIN Clamp",       "noise_scale": 0.25, "clamp_adain": True},
    ]

    models_to_test = [
        ("raw", raw_gen, "Raw Generator"),
        ("ema", ema_gen, "EMA Generator"),
    ]

    # Pre-generate fixed 32 seeds for visual comparisons and identical noise across configs
    torch.manual_seed(2026)
    fixed_z_32 = torch.randn(32, 512, device=device)
    base_noise_32 = build_fixed_base_noise(32, device, raw_gen.synthesis.resolutions)

    # Master visual store for 8-face comparative grid
    master_grid_rows = []
    condition_records = []
    psd_dict = {"Real": psd_real}

    # =========================================================================
    # EXPERIMENTAL LOOP: 12 CONDITIONS
    # =========================================================================
    print(f"\n=======================================================")
    print(f"Starting Evaluation of 12 Experimental Conditions...")
    print(f"=======================================================")

    for m_key, gen_model, m_label in models_to_test:
        for cfg in configs:
            cond_id = f"{m_key}_{cfg['id']}"
            cond_label = f"{m_label} - {cfg['id']} ({cfg['name']})"
            print(f"\n---> Evaluating: {cond_label}")

            noise_scale = cfg["noise_scale"]
            clamp_adain = cfg["clamp_adain"]

            # A. Visual Samples (32 images using fixed_z_32 & base_noise_32)
            cond_samples_dir = samples_dir / cond_id
            cond_samples_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                imgs_32 = generate_controlled_sample(
                    gen_model,
                    fixed_z_32,
                    truncation_psi=0.8,
                    noise_scale=noise_scale,
                    clamp_adain=clamp_adain,
                    base_noise_list=base_noise_32,
                )

            pil_32 = []
            for idx in range(32):
                p_im = to_pil(imgs_32[idx])
                p_im.save(cond_samples_dir / f"sample_{idx:02d}.png")
                pil_32.append(p_im)

            grid_32 = make_grid_image(pil_32, cols=8)
            grid_32.save(grids_dir / f"grid_{cond_id}.png")

            # Store first 8 images for master multi-condition comparison grid
            master_grid_rows.append((cond_id, cond_label, [to_pil(imgs_32[j]) for j in range(8)]))

            # B. Measurements on N=32 batch
            lap_mean, lap_std = measure_laplacian_variance(imgs_32)
            _, psd_cond = compute_radial_psd(imgs_32)
            hf_ratio = float(psd_cond[hf_cut:].sum() / psd_cond[:lf_cut].sum())

            if m_key == "raw":
                psd_dict[f"Raw_{cfg['id']}"] = psd_cond

            # C. Inception Metric Evaluation (N=614, matched evaluation protocol)
            fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            kid_m = KernelInceptionDistance(feature=2048, subsets=10, subset_size=25, normalize=False).to(device)

            fid_m.real_features_sum = fid_template.real_features_sum.clone()
            fid_m.real_features_cov_sum = fid_template.real_features_cov_sum.clone()
            fid_m.real_features_num_samples = fid_template.real_features_num_samples.clone()
            kid_m.real_features = [f.clone() for f in kid_template.real_features]

            batch_sz = 64
            generated = 0
            fake_tensors_list = []
            fake_feats_list = []

            # Fixed evaluation seed for latents
            torch.manual_seed(42)

            # Measure latency
            t_start = time.perf_counter()
            with torch.no_grad():
                while generated < num_val:
                    cur_b = min(batch_sz, num_val - generated)
                    z_eval = torch.randn(cur_b, 512, device=device)
                    # Fixed seed for noise evaluation batch
                    eval_noise = build_fixed_base_noise(cur_b, device, gen_model.synthesis.resolutions)
                    f_imgs = generate_controlled_sample(
                        gen_model,
                        z_eval,
                        truncation_psi=0.8,
                        noise_scale=noise_scale,
                        clamp_adain=clamp_adain,
                        base_noise_list=eval_noise,
                    )
                    f_u8 = to_uint8(f_imgs)
                    fid_m.update(f_u8, real=False)
                    kid_m.update(f_u8, real=False)

                    fake_tensors_list.append(f_imgs.cpu())
                    fake_feats_list.append(inc_net(f_u8).view(cur_b, -1))
                    generated += cur_b

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time_sec = time.perf_counter() - t_start
            latency_ms_per_face = (total_time_sec / num_val) * 1000.0

            fid_score = float(fid_m.compute().item())
            kid_mean, kid_std = kid_m.compute()
            kid_mean_val = float(kid_mean.item())

            all_fakes = torch.cat(fake_tensors_list, dim=0)
            all_fake_feats = torch.cat(fake_feats_list, dim=0)

            # D. Pixel Variance & Diversity Metrics
            flat_fakes = all_fakes.view(all_fakes.size(0), -1)
            pixel_var = float(torch.var(flat_fakes, dim=0).mean().item())

            # Pairwise L1 distance on 200 pairs
            p_dist = torch.cdist(flat_fakes[:200], flat_fakes[:200], p=1.0) / flat_fakes.size(1)
            pairwise_l1 = float(p_dist[p_dist > 0].mean().item())

            # Precision & Recall against real features
            prec, rec = compute_precision_recall(real_feats_all, all_fake_feats, k=3)

            rec_data = {
                "condition_id": cond_id,
                "model": m_key,
                "config_id": cfg["id"],
                "name": cfg["name"],
                "noise_scale": noise_scale,
                "clamp_adain": clamp_adain,
                "fid": fid_score,
                "kid_mean": kid_mean_val,
                "laplacian_variance_mean": lap_mean,
                "laplacian_variance_std": lap_std,
                "hf_to_lf_psd_ratio": hf_ratio,
                "hf_ratio_rel_to_real": hf_ratio / real_hf_ratio,
                "pixel_variance": pixel_var,
                "pairwise_l1": pairwise_l1,
                "precision": prec,
                "recall": rec,
                "latency_ms_per_face": latency_ms_per_face,
            }
            condition_records.append(rec_data)

            print(f"  FID: {fid_score:.2f} | KID: {kid_mean_val:.4f}")
            print(f"  Laplacian Var: {lap_mean:.2f} +- {lap_std:.2f} (Real: {real_lap_mean:.2f})")
            print(f"  HF PSD Ratio:  {hf_ratio:.6f} ({hf_ratio/real_hf_ratio:.2f}x of Real)")
            print(f"  Pixel Var:     {pixel_var:.5f} | Pairwise L1: {pairwise_l1:.4f}")
            print(f"  Latency:       {latency_ms_per_face:.2f} ms/face")

    # =========================================================================
    # VISUAL COMPARISON GRIDS & ABLATION PLOTS
    # =========================================================================
    print("\n--- Generating Master Comparative Visual Grids ---")

    # 1. Master 12-Row Comparison Grid (8 Seeds across all 12 conditions)
    fig, axes = plt.subplots(12, 8, figsize=(16, 24))
    for r_idx, (c_id, c_lbl, img_list) in enumerate(master_grid_rows):
        for c_idx in range(8):
            axes[r_idx, c_idx].imshow(np.array(img_list[c_idx]))
            axes[r_idx, c_idx].axis("off")
        axes[r_idx, 0].set_title(c_id, loc="left", fontsize=9, fontweight="bold")

    plt.suptitle("StyleForge v1.0.0 Controlled Inference Ablation (12 Conditions, 8 Fixed Seeds)", fontsize=13, y=0.995)
    plt.tight_layout()
    master_grid_path = grids_dir / "master_12_condition_comparison.png"
    plt.savefig(master_grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved master grid to {master_grid_path}")

    # 2. Side-by-side Zoom Comparison: C1 vs C3 vs C4 vs C5 vs Real
    print("Generating Detailed Side-by-Side Zoom Comparison...")
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    col_names = ["C1: Baseline (Noise 1.0)", "C3: Noise 0.25", "C4: Denoised (Noise 0.0)", "C5: Noise 0.0 + Clamp", "Authentic Real CelebA"]

    # Select 4 representative seeds
    for seed_idx in range(4):
        # C1 Raw
        p_c1 = Image.open(samples_dir / "raw_C1" / f"sample_{seed_idx:02d}.png")
        p_c3 = Image.open(samples_dir / "raw_C3" / f"sample_{seed_idx:02d}.png")
        p_c4 = Image.open(samples_dir / "raw_C4" / f"sample_{seed_idx:02d}.png")
        p_c5 = Image.open(samples_dir / "raw_C5" / f"sample_{seed_idx:02d}.png")
        p_real = to_pil(real_imgs_all[seed_idx])

        for c_i, p_im in enumerate([p_c1, p_c3, p_c4, p_c5, p_real]):
            axes[seed_idx, c_i].imshow(np.array(p_im))
            axes[seed_idx, c_i].axis("off")
            if seed_idx == 0:
                axes[seed_idx, c_i].set_title(col_names[c_i], fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("Controlled Visual Comparison: Effect of Noise Scaling & AdaIN Clamping (Raw Weights)", fontsize=13, y=0.98)
    plt.tight_layout()
    zoom_grid_path = grids_dir / "zoom_comparison_features.png"
    plt.savefig(zoom_grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved zoom comparison to {zoom_grid_path}")

    # 3. 2D FFT Power Spectral Density Plots (Comparing C1-C6 against Real)
    print("Generating PSD Spectral Plot...")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(freqs, psd_dict["Real"], label="Real CelebA (Val)", color="#2ecc71", linewidth=2.5)
    ax.semilogy(freqs, psd_dict["Raw_C1"], label="C1: Baseline (Noise 1.0)", color="#e74c3c", linewidth=1.8)
    ax.semilogy(freqs, psd_dict["Raw_C2"], label="C2: Noise 0.5", color="#e67e22", linewidth=1.5, linestyle="--")
    ax.semilogy(freqs, psd_dict["Raw_C3"], label="C3: Noise 0.25", color="#f39c12", linewidth=1.5, linestyle="--")
    ax.semilogy(freqs, psd_dict["Raw_C4"], label="C4: Noise 0.0 (Denoised)", color="#3498db", linewidth=1.8)
    ax.semilogy(freqs, psd_dict["Raw_C5"], label="C5: Noise 0.0 + Clamp", color="#9b59b6", linewidth=1.8, linestyle="-.")
    ax.semilogy(freqs, psd_dict["Raw_C6"], label="C6: Noise 0.25 + Clamp", color="#1abc9c", linewidth=1.8, linestyle=":")

    ax.set_xlabel("Spatial Frequency (Cycles / Image)", fontsize=11)
    ax.set_ylabel("Power Spectral Density (Log Scale)", fontsize=11)
    ax.set_title("Radially Averaged 2D FFT Power Spectral Density (C1–C6 vs Real CelebA)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    psd_plot_path = plots_dir / "psd_ablation_comparison.png"
    plt.savefig(psd_plot_path, dpi=150)
    plt.close()
    print(f"Saved PSD plot to {psd_plot_path}")

    # 4. Metric Trade-Off Plot (FID vs Laplacian Variance vs Noise Scale)
    print("Generating Metric Trade-Off Plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    raw_recs = [r for r in condition_records if r["model"] == "raw"]
    ema_recs = [r for r in condition_records if r["model"] == "ema"]

    # Plot 1: FID vs Laplacian Variance
    for recs, m_col, m_marker, m_name in [(raw_recs, "#e74c3c", "o", "Raw"), (ema_recs, "#3498db", "s", "EMA")]:
        xs = [r["laplacian_variance_mean"] for r in recs]
        ys = [r["fid"] for r in recs]
        labels = [r["config_id"] for r in recs]
        ax1.scatter(xs, ys, color=m_col, marker=m_marker, s=80, label=f"{m_name} Generator", zorder=3)
        for x, y, lbl in zip(xs, ys, labels):
            ax1.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax1.axvline(real_lap_mean, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Real CelebA Sharpness ({real_lap_mean:.0f})")
    ax1.set_xlabel("Laplacian Variance (Sharpness / High-Freq Noise)", fontsize=11)
    ax1.set_ylabel("Fréchet Inception Distance (FID ↓)", fontsize=11)
    ax1.set_title("FID vs. Laplacian Variance Across Configurations", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Plot 2: High-Frequency Ratio vs Noise Scale (Unclamped C1-C4)
    unclamped_raw = [r for r in raw_recs if not r["clamp_adain"]]
    unclamped_ema = [r for r in ema_recs if not r["clamp_adain"]]

    n_scales = [r["noise_scale"] for r in unclamped_raw]
    hf_raw = [r["hf_to_lf_psd_ratio"] * 1e5 for r in unclamped_raw]
    hf_ema = [r["hf_to_lf_psd_ratio"] * 1e5 for r in unclamped_ema]

    ax2.plot(n_scales, hf_raw, marker="o", color="#e74c3c", linewidth=2.0, label="Raw High-Freq Energy (x10^5)")
    ax2.plot(n_scales, hf_ema, marker="s", color="#3498db", linewidth=2.0, label="EMA High-Freq Energy (x10^5)")
    ax2.axhline(real_hf_ratio * 1e5, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Real CelebA Ratio ({real_hf_ratio*1e5:.2f})")
    ax2.set_xlabel("Stochastic Noise Scale", fontsize=11)
    ax2.set_ylabel("High-to-Low Frequency Energy Ratio (x10^5)", fontsize=11)
    ax2.set_title("High-Frequency Spectral Energy vs. Noise Scale", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    tradeoff_plot_path = plots_dir / "metric_tradeoffs.png"
    plt.savefig(tradeoff_plot_path, dpi=150)
    plt.close()
    print(f"Saved metric trade-off plot to {tradeoff_plot_path}")

    # =========================================================================
    # SERIALIZE METRICS JSON
    # =========================================================================
    metrics_summary = {
        "timestamp": "2026-11-14T16:30:00Z",
        "eval_split": "val",
        "num_eval_samples": num_val,
        "truncation_psi": 0.8,
        "real_baseline": {
            "laplacian_variance_mean": real_lap_mean,
            "laplacian_variance_std": real_lap_std,
            "hf_to_lf_psd_ratio": real_hf_ratio,
            "pixel_variance": real_pixel_var,
        },
        "conditions": condition_records,
    }

    metrics_json_path = out_dir / "ablation_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nAll ablation metrics successfully serialized to {metrics_json_path}")
    print("=== Ablation Study Completed Successfully! ===")

if __name__ == "__main__":
    main()
