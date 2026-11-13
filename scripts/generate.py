"""StyleForge Generation & Latent Exploration CLI.

Usage:
    python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode samples --n 16
    python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode interpolate --steps 12
    python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode style_mix
    python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode truncation
"""

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path
import torch
from torchvision.utils import save_image

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.style_generator import StyleGenerator
from models.wgan_baseline import WGANBaselineGenerator
from evaluation.latent_explorer import LatentExplorer


def load_generator_from_checkpoint(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    m_cfg = config.get("model", {})
    m_type = m_cfg.get("type", "stylegan")
    img_size = m_cfg.get("img_size", 64)
    img_channels = m_cfg.get("img_channels", 3)
    latent_dim = m_cfg.get("latent_dim", 512)

    if m_type == "wgan_baseline":
        generator = WGANBaselineGenerator(
            latent_dim=latent_dim,
            img_channels=img_channels,
            feature_dim=m_cfg.get("feature_dim", 64),
            img_size=img_size,
        )
    else:
        generator = StyleGenerator(
            latent_dim=latent_dim,
            style_dim=m_cfg.get("style_dim", 512),
            num_mapping_layers=m_cfg.get("num_mapping_layers", 8),
            img_channels=img_channels,
            img_size=img_size,
            base_channels=m_cfg.get("base_channels", 512),
            max_channels=m_cfg.get("max_channels", 512),
            use_attention=m_cfg.get("use_attention", False),
        )

    # Prefer ema_generator weights if present
    weights = state.get("ema_generator", state.get("generator"))
    generator.load_state_dict(weights)
    generator.to(device).eval()
    return generator, config


def main():
    parser = argparse.ArgumentParser(description="StyleForge Generator CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, default="samples", choices=["samples", "interpolate", "style_mix", "truncation"], help="Generation task")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--n", type=int, default=16, help="Number of random samples to generate")
    parser.add_argument("--steps", type=int, default=10, help="Steps for interpolation")
    parser.add_argument("--psi", type=float, default=0.7, help="Truncation psi")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator, config = load_generator_from_checkpoint(args.checkpoint, device)
    latent_dim = config.get("model", {}).get("latent_dim", 512)
    explorer = LatentExplorer(generator, device=device)

    print(f"Loaded generator from {args.checkpoint}. Running mode: {args.mode}")

    if args.mode == "samples":
        z = torch.randn(args.n, latent_dim, device=device)
        with torch.no_grad():
            if hasattr(generator, "style_dim"):
                imgs = generator(z, truncation_psi=args.psi)
            else:
                imgs = generator(z)
        imgs = (imgs * 0.5 + 0.5).clamp(0.0, 1.0)
        save_path = out_dir / "samples.png"
        save_image(imgs, save_path, nrow=int(args.n ** 0.5) or 4)
        print(f"Generated {args.n} samples saved to: {save_path}")

    elif args.mode == "interpolate":
        imgs = explorer.interpolate_z(steps=args.steps, method="slerp", latent_dim=latent_dim, truncation_psi=args.psi)
        imgs = (imgs * 0.5 + 0.5).clamp(0.0, 1.0)
        save_path = out_dir / "interpolation_slerp.png"
        save_image(imgs, save_path, nrow=args.steps)
        print(f"Interpolation strip saved to: {save_path}")

    elif args.mode == "truncation":
        psi_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        imgs = explorer.truncation_sweep(psi_values=psi_vals, latent_dim=latent_dim)
        imgs = (imgs * 0.5 + 0.5).clamp(0.0, 1.0)
        save_path = out_dir / "truncation_sweep.png"
        save_image(imgs, save_path, nrow=len(psi_vals))
        print(f"Truncation sweep (psi={psi_vals}) saved to: {save_path}")

    elif args.mode == "style_mix":
        grid = explorer.style_mixing_matrix(num_src_a=4, num_src_b=4, mix_mode="coarse", latent_dim=latent_dim, truncation_psi=args.psi)
        save_path = out_dir / "style_mixing_matrix.png"
        save_image(grid, save_path, nrow=5)
        print(f"Style mixing matrix saved to: {save_path}")


if __name__ == "__main__":
    main()
