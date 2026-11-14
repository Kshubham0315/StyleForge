"""Generates fixed-seed comparative visual sample grids across all 5 models."""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from models.wgan_baseline import WGANBaselineGenerator
from models.style_generator import StyleGenerator

def run_visual_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase3")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = 8
    torch.manual_seed(42)
    z_style = torch.randn(num_samples, 512, device=device)
    torch.manual_seed(42)
    z_wgan = torch.randn(num_samples, 128, device=device)

    models_info = [
        ("WGAN-GP Baseline (1000 steps)", "experiments/phase2_baseline_wgan_gp/checkpoints/checkpoint_final.pt", "wgan", False, False),
        ("StyleForge AdaIN (Raw / No EMA)", "experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt", "stylegan", False, True),
        ("StyleForge AdaIN (EMA beta=0.999)", "experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt", "stylegan", False, False),
        ("Ablation A: Self-Attention (16x16)", "experiments/phase2_ablation_self_attention/checkpoints/checkpoint_final.pt", "stylegan", True, False),
        ("Ablation B: Spectral Norm Critic", "experiments/phase2_ablation_spectral_norm/checkpoints/checkpoint_final.pt", "stylegan", False, False),
    ]

    generated_rows = []
    stats_list = []

    for label, ckpt_path, m_type, use_attn, force_no_ema in models_info:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if m_type == "wgan":
            gen = WGANBaselineGenerator(latent_dim=128, img_channels=3, img_size=64).to(device).eval()
            if "generator" in ckpt:
                gen.load_state_dict(ckpt["generator"])
            else:
                gen.load_state_dict(ckpt)
            with torch.no_grad():
                fakes = gen(z_wgan)
        else:
            gen = StyleGenerator(latent_dim=512, style_dim=512, img_size=64, use_attention=use_attn).to(device).eval()
            if force_no_ema or "ema_generator" not in ckpt:
                gen.load_state_dict(ckpt["generator"])
            else:
                gen.load_state_dict(ckpt["ema_generator"])
            with torch.no_grad():
                fakes = gen(z_style, truncation_psi=0.8)

        # Scale to [0, 1] for visualization
        fakes_norm = (fakes * 0.5 + 0.5).clamp(0.0, 1.0).cpu()
        generated_rows.append((label, fakes_norm))

        # Compute pixel statistics
        fakes_np = fakes_norm.numpy()
        stats_list.append({
            "model": label,
            "min_val": float(fakes_np.min()),
            "max_val": float(fakes_np.max()),
            "mean_val": float(fakes_np.mean()),
            "std_val": float(fakes_np.std()),
            "batch_var": float(np.var(fakes_np, axis=0).mean()),
        })

    # Plot comparison grid (5 rows x 8 columns)
    fig, axes = plt.subplots(5, num_samples, figsize=(16, 10.5))
    for row_idx, (label, row_tensors) in enumerate(generated_rows):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            img = row_tensors[col_idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(label, fontsize=11, fontweight="bold", loc="left", pad=10)

    plt.suptitle("StyleForge Phase 3: Systematic Visual Quality & Anatomical Audit\n(Fixed Seed=42, 64x64 CelebA)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_file = out_dir / "visual_quality_comparison.png"
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved visual comparison figure to {out_file}")

    # Print summary table
    print("\n=== Pixel Statistics of Fixed-Seed Generated Samples ===")
    print(f"{'Model':42s} | {'Min':5s} | {'Max':5s} | {'Mean':5s} | {'Std':5s} | {'Batch Var':9s}")
    print("-" * 80)
    for s in stats_list:
        print(f"{s['model']:42s} | {s['min_val']:5.3f} | {s['max_val']:5.3f} | {s['mean_val']:5.3f} | {s['std_val']:5.3f} | {s['batch_var']:9.5f}")

if __name__ == "__main__":
    run_visual_comparison()
