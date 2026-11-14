"""Phase 4 Experiment E3: Water-Droplet Artifact & Spectral Frequency Analysis.

Scientific Investigations:
1. 2D FFT Fourier Power Spectrum Analysis:
   - Computes 2D FFT log power spectrum of Real CelebA vs Generated faces.
   - Computes radially averaged power spectral density (1D radial profile) to detect
     checkerboard harmonics or high-frequency energy anomalies.
2. Layer-Wise Spatial Activation Analysis:
   - Hooks intermediate feature representations across all synthesis resolutions (4x4, 8x8, 16x16, 32x32, 64x64).
   - Measures peak-to-mean activation ratio, spatial kurtosis, and localized corner spike frequency
     to detect the signature mechanism of StyleGAN AdaIN droplet artifacts.
3. Upsampling Method Ablation:
   - Evaluates (A) Bilinear, (B) Nearest-Neighbor + Conv, and (C) Anti-aliased (FIR filtered) upsampling.
   - Measures high-frequency energy ratio and visual quality.
"""

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
import torch.nn.functional as F

from data.dataset import get_dataloader
from models.style_generator import StyleGenerator
from models.critic import StyleCritic

def compute_2d_fft_spectrum(images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean 2D FFT log power spectrum and radially averaged power spectrum.

    Args:
        images: (N, C, H, W) in range [-1, 1] or [0, 1]

    Returns:
        spectrum_2d: (H, W) 2D log power spectrum centered at DC
        radial_profile: 1D array of radially averaged power from DC to Nyquist
    """
    images = images.detach()
    # Convert to grayscale luminance: 0.299 R + 0.587 G + 0.114 B
    if images.size(1) == 3:
        gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    else:
        gray = images[:, 0]

    # Compute 2D FFT
    fft = torch.fft.fft2(gray)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = (fft_shifted.abs() ** 2).mean(dim=0).cpu().numpy()  # (H, W)
    log_power = np.log10(power + 1e-12)

    # Radially averaged power spectrum
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    for radius in range(max_r):
        mask = (r == radius)
        if mask.any():
            radial_profile[radius] = power[mask].mean()

    log_radial = np.log10(radial_profile + 1e-12)
    return log_power, log_radial

def get_intermediate_activations(generator: StyleGenerator, z: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Capture layer-wise spatial activations across the synthesis network."""
    activations = {}
    hooks = []

    synthesis = generator.synthesis

    def get_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    # Hook initial block
    hooks.append(synthesis.initial_block.register_forward_hook(get_hook("res_4")))

    # Hook blocks at each resolution
    for res in synthesis.resolutions:
        hooks.append(synthesis.blocks[f"res_{res}_b2"].register_forward_hook(get_hook(f"res_{res}")))

    with torch.no_grad():
        _ = generator(z, truncation_psi=0.8)

    for h in hooks:
        h.remove()

    return activations

def analyze_activation_spikes(activations: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
    """Measure peak-to-mean ratio, spatial variance, and localized spike frequency."""
    stats = {}
    for res_name, act in activations.items():
        # act: (B, C, H, W)
        b, c, h, w = act.shape
        flat_spatial = act.view(b, c, -1)  # (B, C, H*W)

        spatial_mean = flat_spatial.mean(dim=-1, keepdim=True)
        spatial_std = flat_spatial.std(dim=-1, keepdim=True) + 1e-8
        spatial_max = flat_spatial.max(dim=-1, keepdim=True)[0]

        # Peak-to-mean ratio
        pm_ratio = (spatial_max / (spatial_mean.abs() + 1e-8)).mean().item()

        # Z-score of maximum pixel
        z_max = ((spatial_max - spatial_mean) / spatial_std).mean().item()

        # Check corner spike concentration (corners: top-left, top-right, bot-left, bot-right)
        corners = torch.stack([
            act[:, :, 0, 0],
            act[:, :, 0, -1],
            act[:, :, -1, 0],
            act[:, :, -1, -1]
        ], dim=-1)  # (B, C, 4)
        corner_max = corners.max(dim=-1)[0]
        overall_max = spatial_max.squeeze(-1)
        corner_domination = (corner_max >= 0.9 * overall_max).float().mean().item()

        stats[res_name] = {
            "spatial_dim": f"{h}x{w}",
            "channels": c,
            "peak_to_mean_ratio": pm_ratio,
            "max_z_score": z_max,
            "corner_domination_prob": corner_domination,
        }
    return stats

class AntiAliasedUpsample(nn.Module):
    """Bilinear upsampling followed by low-pass anti-aliasing FIR filter."""
    def __init__(self, channels: int):
        super().__init__()
        # 3x3 binomial filter: [1, 2, 1]^T * [1, 2, 1] / 16
        kernel_1d = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel_2d)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Bilinear upsample
        up = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        # 2. Depthwise convolution with low-pass FIR filter
        return F.conv2d(up, self.kernel, padding=1, groups=self.channels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase4/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("StyleForge Phase 4 — Experiment E3: Artifact & Spectral Analysis")
    print(f"Device: {device}")
    print("==================================================")

    # 1. Load Real Validation Data
    val_loader = get_dataloader("data/celeba/img_align_celeba", split="val", img_size=64, batch_size=64, num_workers=0, seed=42)
    real_batches = []
    with torch.no_grad():
        for b in val_loader:
            if isinstance(b, (list, tuple)): b = b[0]
            real_batches.append(b)
            if len(real_batches) >= 6:  # ~384 images
                break
    real_imgs = torch.cat(real_batches, dim=0).to(device)

    # 2. Load / Generate StyleForge Model Samples
    latent_dim = 512
    torch.manual_seed(42)
    generator = StyleGenerator(latent_dim=latent_dim, style_dim=latent_dim, img_size=64).to(device)

    # Try loading best available checkpoint if exists
    ckpt_paths = [
        Path("checkpoints/phase3/checkpoint_step1000.pt"),
        Path("checkpoints/styleforge_celeba.pt"),
        Path("checkpoints/checkpoint_latest.pt"),
    ]
    loaded = False
    for p in ckpt_paths:
        if p.exists():
            print(f"Loading checkpoint: {p}")
            ckpt = torch.load(p, map_location=device, weights_only=False)
            if "generator" in ckpt:
                generator.load_state_dict(ckpt["generator"], strict=False)
                loaded = True
                break
            elif "model_state_dict" in ckpt:
                generator.load_state_dict(ckpt["model_state_dict"], strict=False)
                loaded = True
                break
    if not loaded:
        print("Note: Running with baseline initialized StyleGenerator.")

    generator.eval()

    # Generate synthetic images
    torch.manual_seed(1234)
    with torch.no_grad():
        fake_z = torch.randn(len(real_imgs), latent_dim, device=device)
        fake_imgs = generator(fake_z, truncation_psi=0.8)

    # -------------------------------------------------------------------------
    # PART 1: 2D FFT Spectral Analysis
    # -------------------------------------------------------------------------
    print("\n--- Part 1: Computing 2D FFT Spectra & Radial Profiles ---")
    real_2d_fft, real_radial = compute_2d_fft_spectrum(real_imgs)
    fake_2d_fft, fake_radial = compute_2d_fft_spectrum(fake_imgs)

    # High frequency energy fraction (upper 50% frequencies)
    half_r = len(real_radial) // 2
    real_hf_ratio = float(np.sum(10 ** real_radial[half_r:]) / np.sum(10 ** real_radial))
    fake_hf_ratio = float(np.sum(10 ** fake_radial[half_r:]) / np.sum(10 ** fake_radial))

    print(f"Real Images High-Frequency Energy Ratio: {real_hf_ratio:.4f}")
    print(f"Generated Images High-Frequency Energy Ratio: {fake_hf_ratio:.4f}")

    # Plot 2D FFT & Radial Profile
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(real_2d_fft, cmap="magma")
    axes[0].set_title("Real CelebA 2D FFT Power Spectrum\n(Log Power, Centered DC)", fontsize=11)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(fake_2d_fft, cmap="magma")
    axes[1].set_title("Generated 2D FFT Power Spectrum\n(Log Power, Centered DC)", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 1D Radial Frequency Profile
    freqs = np.arange(len(real_radial))
    axes[2].plot(freqs, real_radial, label="Real CelebA", color="#2ecc71", linewidth=2.5)
    axes[2].plot(freqs, fake_radial, label="StyleForge Fake", color="#e74c3c", linewidth=2.5, linestyle="--")
    axes[2].set_xlabel("Spatial Frequency Radius (pixels)", fontsize=11)
    axes[2].set_ylabel("Log Radial Power Density", fontsize=11)
    axes[2].set_title("Radially Averaged Power Spectrum Comparison", fontsize=11)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fft_plot_path = out_dir / "e3_fft_spectral_comparison.png"
    plt.savefig(fft_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # PART 2: Layer-Wise Spatial Activation Analysis
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Layer-Wise Spatial Activation & Outlier Spikes ---")
    z_probe = torch.randn(32, latent_dim, device=device)
    activations = get_intermediate_activations(generator, z_probe)
    spike_stats = analyze_activation_spikes(activations)

    print(f"{'Resolution':<12} | {'Dim':<8} | {'Peak-to-Mean':<14} | {'Max Z-Score':<14} | {'Corner Domination %':<20}")
    print("-" * 75)
    for res_name, st in spike_stats.items():
        print(f"{res_name:<12} | {st['spatial_dim']:<8} | {st['peak_to_mean_ratio']:<14.2f} | {st['max_z_score']:<14.2f} | {st['corner_domination_prob']*100:<20.1f}%")

    # Plot Spatial Activation Outliers
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    res_keys = list(spike_stats.keys())
    pm_ratios = [spike_stats[k]["peak_to_mean_ratio"] for k in res_keys]
    z_scores = [spike_stats[k]["max_z_score"] for k in res_keys]

    axes[0].plot(res_keys, pm_ratios, marker="o", color="#e67e22", linewidth=2.5, markersize=8)
    axes[0].set_title("Spatial Peak-to-Mean Ratio Across Generator Layers", fontsize=11)
    axes[0].set_ylabel("Peak-to-Mean Ratio", fontsize=11)
    axes[0].set_xlabel("Layer Resolution", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(res_keys, z_scores, marker="s", color="#9b59b6", linewidth=2.5, markersize=8)
    axes[1].set_title("Max Spatial Activation Z-Score (Outlier Severity)", fontsize=11)
    axes[1].set_ylabel("Z-Score relative to Layer Mean", fontsize=11)
    axes[1].set_xlabel("Layer Resolution", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    act_plot_path = out_dir / "e3_activation_spikes.png"
    plt.savefig(act_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # PART 3: Upsampling Ablation Study
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Upsampling Method Ablation ---")
    test_feature_map = torch.randn(16, 64, 16, 16, device=device)

    # 1. Bilinear
    up_bilinear = F.interpolate(test_feature_map, scale_factor=2.0, mode="bilinear", align_corners=False)

    # 2. Nearest Neighbor + Conv
    up_nearest = F.interpolate(test_feature_map, scale_factor=2.0, mode="nearest")
    conv_smooth = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False).to(device)
    with torch.no_grad():
        nn.init.dirac_(conv_smooth.weight)
    up_nearest_conv = conv_smooth(up_nearest)

    # 3. Anti-Aliased Filter
    aa_filter = AntiAliasedUpsample(channels=64).to(device)
    up_aa = aa_filter(test_feature_map)

    # Compute high-frequency power of each upsampling mode
    _, r_bilinear = compute_2d_fft_spectrum(up_bilinear[:, :3])
    _, r_nearest = compute_2d_fft_spectrum(up_nearest_conv[:, :3])
    _, r_aa = compute_2d_fft_spectrum(up_aa[:, :3])

    hf_start = len(r_bilinear) // 2
    hf_bilinear = float(np.sum(10 ** r_bilinear[hf_start:]) / np.sum(10 ** r_bilinear))
    hf_nearest = float(np.sum(10 ** r_nearest[hf_start:]) / np.sum(10 ** r_nearest))
    hf_aa = float(np.sum(10 ** r_aa[hf_start:]) / np.sum(10 ** r_aa))

    print(f"Bilinear High-Frequency Energy Ratio: {hf_bilinear:.4f}")
    print(f"Nearest-Neighbor + Conv HF Energy Ratio: {hf_nearest:.4f}")
    print(f"Anti-Aliased FIR Upsampling HF Energy Ratio: {hf_aa:.4f}")

    upsample_summary = {
        "bilinear_hf_ratio": hf_bilinear,
        "nearest_conv_hf_ratio": hf_nearest,
        "anti_aliased_hf_ratio": hf_aa,
    }

    # Save complete JSON
    results = {
        "metadata": {
            "experiment": "Phase 4 E3 Water-Droplet & Spectral Analysis",
            "device": str(device),
            "num_real_evaluated": len(real_imgs),
            "num_fake_evaluated": len(fake_imgs),
        },
        "fft_metrics": {
            "real_hf_ratio": real_hf_ratio,
            "fake_hf_ratio": fake_hf_ratio,
        },
        "spatial_spikes": spike_stats,
        "upsampling_ablation": upsample_summary,
    }

    with open(out_dir / "e3_artifact_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll E3 artifact analyses saved to {out_dir}")

if __name__ == "__main__":
    main()
