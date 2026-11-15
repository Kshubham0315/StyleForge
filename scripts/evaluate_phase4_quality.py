"""StyleForge Phase 4 Generation Quality & Objective Evaluation.

Performs rigorous objective evaluation comparing:
- Baseline Checkpoint (Step 1,500)
- Phase 4 Continuation Checkpoint

Computes:
1. Fréchet Feature Distance (perceptual distribution match)
2. Generative Precision & Recall (k=3 manifold estimation)
3. Pairwise L1 / L2 Diversity & Intra-batch Variance (mode collapse check)
4. Laplacian Variance (perceptual sharpness)
5. Generates side-by-side fixed-seed comparison images
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from data.dataset import CelebADataset, FaceTransform
from models.style_generator import StyleGenerator
from evaluation.fid_kid import PerceptualFeatureBank, calculate_frechet_distance
from evaluation.diversity import compute_precision_recall, compute_pairwise_distances



def compute_laplacian_sharpness(images: torch.Tensor) -> float:
    """Computes mean Laplacian variance across images as sharpness metric."""
    # Convert [-1, 1] tensor to [0, 255] numpy uint8 grayscale
    imgs_np = ((images * 0.5 + 0.5).clamp(0, 1) * 255.0).cpu().numpy().astype(np.uint8)
    variances = []
    for i in range(imgs_np.shape[0]):
        # RGB to grayscale
        gray = cv2.cvtColor(np.transpose(imgs_np[i], (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        variances.append(var)
    return float(np.mean(variances))


def compute_diversity_metrics(images: torch.Tensor) -> dict:
    """Computes pairwise L1 distance and intra-batch pixel variance."""
    N = images.size(0)
    flat = images.view(N, -1)
    # L1 pairwise distance
    l1_dists = torch.cdist(flat, flat, p=1.0) / flat.size(1)
    mask = ~torch.eye(N, dtype=torch.bool, device=images.device)
    mean_l1 = float(l1_dists[mask].mean().item())
    
    # Intra-batch pixel variance
    pixel_var = float(flat.var(dim=0).mean().item())
    return {
        "pairwise_l1_diversity": mean_l1,
        "pixel_variance": pixel_var,
    }


def evaluate_checkpoint(
    ckpt_path: str,
    feature_bank: PerceptualFeatureBank,
    val_features: torch.Tensor,
    real_mu: np.ndarray,
    real_sigma: np.ndarray,
    device: torch.device,
    num_samples: int = 512,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    """Evaluates a single checkpoint on objective quality metrics."""
    print(f"\n--- Evaluating Checkpoint: {ckpt_path} ---")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    generator = StyleGenerator(
        latent_dim=512,
        style_dim=512,
        num_mapping_layers=8,
        img_size=64,
        base_channels=512,
        max_channels=512,
    ).to(device)
    
    # Prefer ema_generator if present, else generator
    if "ema_generator" in state:
        generator.load_state_dict(state["ema_generator"])
        used_weights = "EMA Generator"
    else:
        generator.load_state_dict(state["generator"])
        used_weights = "Standard Generator"
    
    generator.eval()
    step = state.get("global_step", 0)
    print(f"Loaded {used_weights} at global step: {step}")

    # Generate samples deterministically
    torch.manual_seed(seed)
    all_fake_images = []
    all_fake_features = []

    batches = (num_samples + batch_size - 1) // batch_size
    with torch.no_grad():
        for b in range(batches):
            cur_bs = min(batch_size, num_samples - b * batch_size)
            z = torch.randn(cur_bs, 512, device=device)
            fake_img = generator(z)
            feats = feature_bank(fake_img)
            all_fake_images.append(fake_img)
            all_fake_features.append(feats)

    fake_images = torch.cat(all_fake_images, dim=0)[:num_samples]
    fake_features = torch.cat(all_fake_features, dim=0)[:num_samples]

    # 1. Frechet Feature Distance
    fake_np = fake_features.cpu().numpy()
    fake_mu = np.mean(fake_np, axis=0)
    fake_sigma = np.cov(fake_np, rowvar=False)
    ffd = calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

    # 2. Precision & Recall (k=3)
    precision, recall = compute_precision_recall(val_features, fake_features, k=3)

    # 3. Diversity & Sharpness
    div = compute_diversity_metrics(fake_images)
    sharpness = compute_laplacian_sharpness(fake_images)

    results = {
        "checkpoint": ckpt_path,
        "global_step": step,
        "weights_type": used_weights,
        "num_evaluated_samples": num_samples,
        "frechet_feature_distance": round(ffd, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "pairwise_l1_diversity": round(div["pairwise_l1_diversity"], 4),
        "pixel_variance": round(div["pixel_variance"], 4),
        "laplacian_sharpness": round(sharpness, 2),
    }

    print(f"Results for Step {step}:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results, fake_images[:16]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"StyleForge Phase 4 Generation Quality Evaluator on {device}")

    # 1. Validation Dataset (strictly unaugmented)
    val_dataset = CelebADataset(
        "data/celeba/img_align_celeba",
        split="val",
        rotation=False,
        color_jitter=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Loaded Validation Dataset: {len(val_dataset)} images")

    # 2. Feature Bank
    feature_bank = PerceptualFeatureBank().to(device).eval()

    # Extract real validation features
    print("Extracting validation perceptual features...")
    real_feats_list = []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)
            feats = feature_bank(imgs)
            real_feats_list.append(feats)
    val_features = torch.cat(real_feats_list, dim=0)
    real_np = val_features.cpu().numpy()
    real_mu = np.mean(real_np, axis=0)
    real_sigma = np.cov(real_np, rowvar=False)

    # 3. Candidate Checkpoints across the training trajectory
    candidate_list = [
        "experiments/phase3_convergence/checkpoints/checkpoint_final.pt",
        "experiments/phase4_training_continuation/checkpoints/checkpoint_step_002000.pt",
        "experiments/phase4_training_continuation/checkpoints/checkpoint_step_005000.pt",
        "experiments/phase4_training_continuation/checkpoints/checkpoint_step_008000.pt",
        "experiments/phase4_training_continuation/checkpoints/checkpoint_step_010000.pt",
    ]
    eval_targets = [p for p in candidate_list if os.path.exists(p)]

    all_metrics = []
    sample_grids = {}

    for ckpt in eval_targets:
        res, sample_grid = evaluate_checkpoint(
            ckpt_path=ckpt,
            feature_bank=feature_bank,
            val_features=val_features,
            real_mu=real_mu,
            real_sigma=real_sigma,
            device=device,
            num_samples=min(512, len(val_dataset)),
            batch_size=32,
            seed=42,
        )
        all_metrics.append(res)
        sample_grids[res["global_step"]] = sample_grid

    # Save metrics JSON
    os.makedirs("reports", exist_ok=True)
    out_json = "reports/phase4_generation_quality.json"
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[+] Saved evaluation metrics to {out_json}")

    # Multi-checkpoint progression grid
    if len(sample_grids) >= 2:
        steps = sorted(list(sample_grids.keys()))
        num_seeds = 4
        progression_rows = []
        for seed_idx in range(num_seeds):
            for s in steps:
                progression_rows.append(sample_grids[s][seed_idx:seed_idx+1])
        composite = torch.cat(progression_rows, dim=0)

        out_img_path = "reports/visual_audit/phase4_quality_comparison.png"
        os.makedirs("reports/visual_audit", exist_ok=True)
        save_image(
            composite,
            out_img_path,
            nrow=len(steps),
            normalize=True,
            value_range=(-1, 1),
            padding=4,
        )
        print(f"[+] Saved visual progression grid ({len(steps)} steps x {num_seeds} seeds) to {out_img_path}")


if __name__ == "__main__":
    main()
