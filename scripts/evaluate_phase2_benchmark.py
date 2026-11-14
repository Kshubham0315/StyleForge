"""Phase 2 Research Evaluation, Benchmarking & Visual Artifacts Pipeline.

Evaluates trained GAN generators on authentic CelebA data:
- Official TorchMetrics Inception-v3 FID with matched real/fake sample sizes (N=614)
- Unbiased Inception-v3 Kernel Inception Distance (KID, subsets=20, subset_size=50)
- Inception-v3 feature-space Precision and Recall (k=3 manifold estimation)
- Pixel-space diversity (Pairwise L1/L2 and batch pixel variance)
- Nearest-neighbor memorization analysis against real training images
- Visual artifact generation: 64-sample grid, SLERP interpolation, truncation sweep, style mixing matrix
"""

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from data.dataset import get_dataloader
from models.wgan_baseline import WGANBaselineGenerator
from models.style_generator import StyleGenerator
from evaluation.diversity import compute_precision_recall
from evaluation.latent_explorer import LatentExplorer


def to_uint8_images(tensor: torch.Tensor) -> torch.Tensor:
    """Converts [-1, 1] float tensor to [0, 255] uint8 tensor."""
    tensor = (tensor * 0.5 + 0.5).clamp(0.0, 1.0)
    return (tensor * 255.0).to(torch.uint8)


def load_generator_checkpoint(config: dict, ckpt_path: str, device: torch.device, force_no_ema: bool = False) -> nn.Module:
    m_cfg = config.get("model", {})
    m_type = m_cfg.get("type", "stylegan")
    img_size = m_cfg.get("img_size", 64)
    img_channels = m_cfg.get("img_channels", 3)
    latent_dim = m_cfg.get("latent_dim", 512)
    use_attention = m_cfg.get("use_attention", False)

    if m_type == "wgan_baseline":
        feature_dim = m_cfg.get("feature_dim", 64)
        generator = WGANBaselineGenerator(
            latent_dim=latent_dim,
            img_channels=img_channels,
            feature_dim=feature_dim,
            img_size=img_size,
            use_attention=use_attention,
        )
    else:
        style_dim = m_cfg.get("style_dim", 512)
        num_mapping_layers = m_cfg.get("num_mapping_layers", 8)
        base_channels = m_cfg.get("base_channels", 512)
        max_channels = m_cfg.get("max_channels", 512)
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

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer EMA weights if available and not forced off
    if not force_no_ema and "ema_generator" in state:
        generator.load_state_dict(state["ema_generator"])
        print(f"Loaded EMA generator weights from {ckpt_path}")
    elif "generator" in state:
        generator.load_state_dict(state["generator"])
        print(f"Loaded regular generator weights (no EMA) from {ckpt_path}")
    else:
        generator.load_state_dict(state)
        print(f"Loaded raw generator state_dict from {ckpt_path}")

    generator = generator.to(device).eval()
    return generator


def run_full_evaluation(
    config: dict,
    ckpt_path: str,
    output_dir: Path,
    device: torch.device,
    seed: int = 42,
    force_no_ema: bool = False,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    m_cfg = config.get("model", {})
    m_type = m_cfg.get("type", "stylegan")
    latent_dim = m_cfg.get("latent_dim", 512 if m_type == "stylegan" else 128)
    img_size = m_cfg.get("img_size", 64)

    # 1. Load generator
    generator = load_generator_checkpoint(config, ckpt_path, device, force_no_ema=force_no_ema)

    # 2. Validation DataLoader
    val_loader = get_dataloader(
        dataset_path="data/celeba/img_align_celeba",
        split="val",
        img_size=img_size,
        batch_size=32,
        num_workers=0,
        seed=seed,
    )
    N_val = len(val_loader.dataset)
    print(f"Validation dataset size: {N_val} authentic CelebA images.")

    # 3. Setup Inception Metrics
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_metric = KernelInceptionDistance(feature=2048, subsets=20, subset_size=50, normalize=False).to(device)

    # Access Inception feature extractor from fid_metric
    inception_net = fid_metric.inception.to(device).eval()

    # Accumulate real images & features
    print("Extracting Inception features for real validation images...")
    real_features_list = []
    real_images_list = []
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            real_images_list.append(batch.cpu())
            uint8_b = to_uint8_images(batch)
            fid_metric.update(uint8_b, real=True)
            kid_metric.update(uint8_b, real=True)

            feats = inception_net(uint8_b)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            real_features_list.append(feats.view(feats.size(0), -1).cpu())

    real_features = torch.cat(real_features_list, dim=0)
    real_images_tensor = torch.cat(real_images_list, dim=0)
    print(f"  Real features extracted: {real_features.shape}")

    # 4. Generate matched number of synthetic images (N_val)
    print(f"Generating {N_val} synthetic images using fixed seed={seed}...")
    torch.manual_seed(seed)
    fake_features_list = []
    fake_images_list = []

    count = 0
    batch_size = 32
    with torch.no_grad():
        while count < N_val:
            cur_bs = min(batch_size, N_val - count)
            z = torch.randn(cur_bs, latent_dim, device=device)
            if hasattr(generator, "style_dim"):
                fake = generator(z, truncation_psi=0.8)
            else:
                fake = generator(z)
            fake_images_list.append(fake.cpu())
            uint8_f = to_uint8_images(fake)
            fid_metric.update(uint8_f, real=False)
            kid_metric.update(uint8_f, real=False)

            feats = inception_net(uint8_f)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            fake_features_list.append(feats.view(feats.size(0), -1).cpu())
            count += cur_bs

    fake_features = torch.cat(fake_features_list, dim=0)
    fake_images_tensor = torch.cat(fake_images_list, dim=0)
    print(f"  Fake features extracted: {fake_features.shape}")

    # 5. Compute Inception FID & KID
    print("Computing Fréchet Inception Distance (FID)...")
    fid_score = float(fid_metric.compute().item())
    print(f"  Inception-v3 FID: {fid_score:.2f}")

    print("Computing Kernel Inception Distance (KID)...")
    kid_mean, kid_std = kid_metric.compute()
    kid_m = float(kid_mean.item())
    kid_s = float(kid_std.item())
    print(f"  Inception-v3 KID: {kid_m:.5f} +/- {kid_s:.5f}")

    # 6. Compute Feature-Space Precision and Recall
    print("Computing Inception Feature Precision & Recall (k=3)...")
    precision, recall = compute_precision_recall(real_features, fake_features, k=3)
    print(f"  Inception Precision: {precision:.4f} (Fidelity)")
    print(f"  Inception Recall:    {recall:.4f} (Coverage)")

    # 7. Compute Pixel Diversity Metrics
    flat_fakes = fake_images_tensor.view(fake_images_tensor.size(0), -1)
    batch_variance = float(torch.var(flat_fakes, dim=0).mean().item())

    # Pairwise L1 and L2
    idx1 = torch.randint(0, N_val, (1000,))
    idx2 = torch.randint(0, N_val, (1000,))
    mask = idx1 != idx2
    diff = flat_fakes[idx1[mask]] - flat_fakes[idx2[mask]]
    l1_dist = float(torch.abs(diff).mean().item())
    l2_dist = float(torch.norm(diff, p=2, dim=1).mean().item())

    results = {
        "model_type": m_type,
        "checkpoint_path": str(ckpt_path),
        "dataset": "CelebA Align&Cropped (Photographic)",
        "resolution": f"{img_size}x{img_size}",
        "sample_count_real": N_val,
        "sample_count_fake": N_val,
        "fid": round(fid_score, 4),
        "kid_mean": round(kid_m, 6),
        "kid_std": round(kid_s, 6),
        "inception_precision": round(precision, 4),
        "inception_recall": round(recall, 4),
        "pairwise_l1_diversity": round(l1_dist, 4),
        "pairwise_l2_diversity": round(l2_dist, 4),
        "batch_pixel_variance": round(batch_variance, 6),
        "timestamp": time.time(),
    }

    # Save metrics JSON & CSV
    with open(output_dir / "eval_phase2_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame([results]).to_csv(output_dir / "eval_phase2_metrics.csv", index=False)
    print(f"Saved evaluation metrics to {output_dir / 'eval_phase2_metrics.json'}")

    # 8. Visual Artifact Generation
    print("Generating visual research artifacts...")
    # (A) 64-sample fixed grid
    grid_64 = (fake_images_tensor[:64] * 0.5 + 0.5).clamp(0.0, 1.0)
    vutils.save_image(grid_64, output_dir / "samples_64.png", nrow=8)

    # (B) SLERP Latent Interpolation
    explorer = LatentExplorer(generator, device=device)
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    interp_strip = explorer.interpolate_z(z_start=z1, z_end=z2, steps=10, method="slerp")
    interp_strip = (interp_strip.cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
    vutils.save_image(interp_strip, output_dir / "interpolation_slerp.png", nrow=10)

    # (C) StyleForge specific: Truncation sweep and Style mixing
    if hasattr(generator, "style_dim") and hasattr(generator, "mapping"):
        trunc_imgs = []
        fixed_z = torch.randn(4, latent_dim, device=device)
        for psi in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            with torch.no_grad():
                out = generator(fixed_z, truncation_psi=psi)
                trunc_imgs.append((out.cpu() * 0.5 + 0.5).clamp(0.0, 1.0))
        trunc_grid = torch.cat(trunc_imgs, dim=0)
        vutils.save_image(trunc_grid, output_dir / "truncation_sweep.png", nrow=4)

        # Style mixing matrix (4x4)
        style_matrix = explorer.style_mixing_matrix(num_src_a=4, num_src_b=4, mix_mode="coarse")
        style_matrix = (style_matrix.cpu() * 0.5 + 0.5).clamp(0.0, 1.0)
        vutils.save_image(style_matrix, output_dir / "style_mixing_matrix.png", nrow=5)

    # (D) Nearest-Neighbor Memorization Diagnostic
    # For 8 fake samples, find nearest real training sample in pixel space
    train_loader = get_dataloader(
        dataset_path="data/celeba/img_align_celeba",
        split="train",
        img_size=img_size,
        batch_size=64,
        num_workers=0,
        max_samples=256,
        seed=seed,
    )
    train_imgs = []
    for b in train_loader:
        if isinstance(b, (list, tuple)):
            b = b[0]
        train_imgs.append(b.cpu())
    train_imgs = torch.cat(train_imgs, dim=0)  # (M, 3, 64, 64)

    flat_train = train_imgs.view(train_imgs.size(0), -1)
    nn_pairs = []
    for i in range(8):
        fake_i = flat_fakes[i : i + 1]
        dist = torch.norm(flat_train - fake_i, p=2, dim=1)
        nn_idx = torch.argmin(dist).item()
        nn_pairs.append(fake_images_tensor[i])
        nn_pairs.append(train_imgs[nn_idx])

    nn_tensor = (torch.stack(nn_pairs) * 0.5 + 0.5).clamp(0.0, 1.0)
    vutils.save_image(nn_tensor, output_dir / "nearest_neighbors_gen_vs_real.png", nrow=2)
    print(f"Nearest-neighbor visualization saved to {output_dir / 'nearest_neighbors_gen_vs_real.png'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="StyleForge Phase 2 Benchmarking")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Output destination folder")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    parser.add_argument("--force_no_ema", action="store_true", help="Force evaluating non-EMA raw generator weights")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_full_evaluation(config, args.checkpoint, Path(args.output_dir), device, seed=args.seed, force_no_ema=args.force_no_ema)


if __name__ == "__main__":
    import yaml
    main()
