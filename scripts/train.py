"""StyleForge Training CLI script.

Usage:
    python scripts/train.py --config configs/stylegan_adain.yaml
    python scripts/train.py --config configs/quick_run.yaml
    python scripts/train.py --config configs/baseline_wgan_gp.yaml --device cuda
"""

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import get_dataloader
from models.wgan_baseline import WGANBaselineGenerator, WGANBaselineCritic
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.trainer import GANTrainer


def set_seed(seed: int = 42, deterministic: bool = False):
    """Sets global random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_models(config):
    m_cfg = config.get("model", {})
    m_type = m_cfg.get("type", "stylegan")
    img_size = m_cfg.get("img_size", 64)
    img_channels = m_cfg.get("img_channels", 3)
    latent_dim = m_cfg.get("latent_dim", 512)
    use_attention = m_cfg.get("use_attention", False)
    use_spectral_norm = m_cfg.get("use_spectral_norm", False)

    if m_type == "wgan_baseline":
        feature_dim = m_cfg.get("feature_dim", 64)
        generator = WGANBaselineGenerator(
            latent_dim=latent_dim,
            img_channels=img_channels,
            feature_dim=feature_dim,
            img_size=img_size,
            use_attention=use_attention,
        )
        critic = WGANBaselineCritic(
            img_channels=img_channels,
            feature_dim=feature_dim,
            img_size=img_size,
            use_spectral_norm=use_spectral_norm,
            use_attention=use_attention,
        )
    elif m_type == "stylegan":
        style_dim = m_cfg.get("style_dim", 512)
        num_mapping_layers = m_cfg.get("num_mapping_layers", 8)
        base_channels = m_cfg.get("base_channels", 512)
        max_channels = m_cfg.get("max_channels", 512)
        use_minibatch_std = m_cfg.get("use_minibatch_std", True)

        generator = StyleGenerator(
            latent_dim=latent_dim,
            style_dim=style_dim,
            num_mapping_layers=num_mapping_layers,
            img_channels=img_channels,
            img_size=img_size,
            base_channels=base_channels,
            max_channels=max_channels,
            use_attention=use_attention,
        )
        critic = StyleCritic(
            img_channels=img_channels,
            img_size=img_size,
            base_channels=64,
            max_channels=max_channels,
            use_minibatch_std=use_minibatch_std,
            use_spectral_norm=use_spectral_norm,
            use_attention=use_attention,
        )
    else:
        raise ValueError(f"Unknown model type: {m_type}")

    return generator, critic


def main():
    parser = argparse.ArgumentParser(description="StyleForge GAN Training CLI")
    parser.add_argument("--config", type=str, default="configs/quick_run.yaml", help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable strict cuDNN determinism")
    args = parser.parse_args()

    config = load_config(args.config)

    # Overrides
    if args.steps is not None:
        config["training"]["total_steps"] = args.steps

    seed = args.seed or config.get("experiment", {}).get("seed", 42)
    set_seed(seed, deterministic=args.deterministic)
    config["experiment"]["seed"] = seed

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================")
    print(f"StyleForge Training Pipeline")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Model Type: {config['model']['type']}")
    print(f"Resolution: {config['model']['img_size']}x{config['model']['img_size']}")
    print(f"Seed: {seed} (Deterministic: {args.deterministic})")
    print(f"==================================================")

    # DataLoader
    d_cfg = config.get("data", {})
    dataloader = get_dataloader(
        dataset_path=d_cfg.get("dataset_path"),
        split="train",
        img_size=d_cfg.get("img_size", 64),
        batch_size=d_cfg.get("batch_size", 32),
        num_workers=d_cfg.get("num_workers", 0),
        use_synthetic=d_cfg.get("use_synthetic", False),
        synthetic_samples=d_cfg.get("synthetic_samples", 500),
        seed=seed,
    )

    # Models
    generator, critic = build_models(config)

    # Trainer
    trainer = GANTrainer(
        generator=generator,
        critic=critic,
        dataloader=dataloader,
        config=config,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    total_steps = config["training"]["total_steps"]
    sample_every = config["training"].get("sample_every", 250)
    checkpoint_every = config["training"].get("checkpoint_every", 1000)

    trainer.train(
        total_steps=total_steps,
        sample_every=sample_every,
        checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    main()
