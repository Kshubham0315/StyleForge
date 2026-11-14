"""Comprehensive Nearest-Neighbor Memorization Audit in Pixel & Inception Space."""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from data.dataset import get_dataloader
from models.style_generator import StyleGenerator

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return ((t * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def run_memorization_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load generator
    ckpt_path = "experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device).eval()
    gen.load_state_dict(ckpt.get("ema_generator", ckpt.get("generator")))

    # 2. Inception feature extractor
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    inc = fid_metric.inception.eval()

    # 3. Generate N=64 query samples
    print("Generating 64 synthetic query faces...")
    torch.manual_seed(42)
    z = torch.randn(64, 512, device=device)
    with torch.no_grad():
        fake_imgs = gen(z, truncation_psi=0.8)
        fake_uint8 = to_uint8(fake_imgs)
        fake_feats = inc(fake_uint8).view(64, -1)
    fake_flat = fake_imgs.view(64, -1)

    # 4. Extract Real Train & Val Features
    def extract_dataset(split_name: str, max_batches: int = 40):
        loader = get_dataloader("data/celeba/img_align_celeba", split=split_name, img_size=64, batch_size=64, num_workers=0)
        imgs_list, feats_list = [], []
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= max_batches:
                    break
                b = b.to(device)
                imgs_list.append(b)
                feats_list.append(inc(to_uint8(b)).view(b.size(0), -1))
        return torch.cat(imgs_list, dim=0), torch.cat(feats_list, dim=0)

    print("Extracting real training set (up to 2,560 images)...")
    train_imgs, train_feats = extract_dataset("train", max_batches=40)
    print("Extracting real validation set (614 images)...")
    val_imgs, val_feats = extract_dataset("val", max_batches=10)

    train_flat = train_imgs.view(train_imgs.size(0), -1)
    val_flat = val_imgs.view(val_imgs.size(0), -1)

    # 5. Compute Distances
    # Pixel space L2 (normalized by dimension)
    d_pixel_train = torch.cdist(fake_flat, train_flat, p=2) / (fake_flat.size(1) ** 0.5)
    d_pixel_val = torch.cdist(fake_flat, val_flat, p=2) / (fake_flat.size(1) ** 0.5)

    # Inception feature space L2
    d_feat_train = torch.cdist(fake_feats, train_feats, p=2)
    d_feat_val = torch.cdist(fake_feats, val_feats, p=2)

    # Feature space cosine similarity
    norm_fake = fake_feats / fake_feats.norm(dim=1, keepdim=True)
    norm_train = train_feats / train_feats.norm(dim=1, keepdim=True)
    norm_val = val_feats / val_feats.norm(dim=1, keepdim=True)
    cos_sim_train = torch.mm(norm_fake, norm_train.T)
    cos_sim_val = torch.mm(norm_fake, norm_val.T)

    # Closest neighbors
    min_pix_train, min_pix_train_idx = d_pixel_train.min(dim=1)
    min_pix_val, min_pix_val_idx = d_pixel_val.min(dim=1)
    min_feat_train, min_feat_train_idx = d_feat_train.min(dim=1)
    min_feat_val, min_feat_val_idx = d_feat_val.min(dim=1)
    max_cos_train, max_cos_train_idx = cos_sim_train.max(dim=1)
    max_cos_val, max_cos_val_idx = cos_sim_val.max(dim=1)

    results = {
        "pixel_dist_train": {
            "min": float(min_pix_train.min().item()),
            "mean": float(min_pix_train.mean().item()),
            "max": float(min_pix_train.max().item()),
        },
        "pixel_dist_val": {
            "min": float(min_pix_val.min().item()),
            "mean": float(min_pix_val.mean().item()),
            "max": float(min_pix_val.max().item()),
        },
        "feature_dist_train": {
            "min": float(min_feat_train.min().item()),
            "mean": float(min_feat_train.mean().item()),
            "max": float(min_feat_train.max().item()),
        },
        "feature_dist_val": {
            "min": float(min_feat_val.min().item()),
            "mean": float(min_feat_val.mean().item()),
            "max": float(min_feat_val.max().item()),
        },
        "cosine_sim_train": {
            "min": float(max_cos_train.min().item()),
            "mean": float(max_cos_train.mean().item()),
            "max": float(max_cos_train.max().item()),
        },
        "cosine_sim_val": {
            "min": float(max_cos_val.min().item()),
            "mean": float(max_cos_val.mean().item()),
            "max": float(max_cos_val.max().item()),
        },
    }

    with open(out_dir / "memorization_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    # 6. Plot Nearest Neighbor Pairs (Inception Feature Space)
    # Find 4 queries with highest cosine similarity and 4 queries with lowest
    sorted_cos, sorted_cos_idx = torch.sort(max_cos_train, descending=True)
    selected_queries = list(sorted_cos_idx[:4]) + list(sorted_cos_idx[-4:])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, q_idx in enumerate(selected_queries):
        row = i // 2
        col_offset = (i % 2) * 2

        q_img = (fake_imgs[q_idx] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        nn_idx = min_feat_train_idx[q_idx].item()
        nn_img = (train_imgs[nn_idx] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        cos_score = max_cos_train[q_idx].item()
        feat_l2 = min_feat_train[q_idx].item()

        # Query
        axes[row, col_offset].imshow(q_img)
        axes[row, col_offset].axis("off")
        axes[row, col_offset].set_title(f"Gen Face #{q_idx}", fontsize=10, fontweight="bold")

        # Nearest Real
        axes[row, col_offset + 1].imshow(nn_img)
        axes[row, col_offset + 1].axis("off")
        axes[row, col_offset + 1].set_title(f"Train NN (Cos: {cos_score:.3f}, L2: {feat_l2:.1f})", fontsize=9)

    plt.suptitle("StyleForge Phase 3: Nearest-Neighbor Memorization Audit\nLeft: Generated Face | Right: Closest Real Training Face (Inception Space)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pair_plot = out_dir / "memorization_audit_pairs.png"
    plt.savefig(pair_plot, dpi=180, bbox_inches="tight")
    plt.close()

    # 7. Plot Distance Distribution Histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(min_feat_train.cpu().numpy(), bins=15, alpha=0.7, label="Train NN Distances", color="navy")
    ax1.hist(min_feat_val.cpu().numpy(), bins=15, alpha=0.6, label="Val NN Distances", color="crimson")
    ax1.set_title("Inception Feature-Space Distance to Nearest Real", fontsize=12, fontweight="bold")
    ax1.set_xlabel("L2 Distance in R^2048")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.hist(max_cos_train.cpu().numpy(), bins=15, alpha=0.7, label="Train NN Cosine Sim", color="navy")
    ax2.hist(max_cos_val.cpu().numpy(), bins=15, alpha=0.6, label="Val NN Cosine Sim", color="crimson")
    ax2.set_title("Inception Feature-Space Cosine Similarity to Nearest Real", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cosine Similarity")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    hist_plot = out_dir / "memorization_distance_distributions.png"
    plt.savefig(hist_plot, dpi=180, bbox_inches="tight")
    plt.close()

    print("\n=== MEMORIZATION AUDIT SUMMARY ===")
    print(f"Feature Distance to Train NN: Min={results['feature_dist_train']['min']:.2f}, Mean={results['feature_dist_train']['mean']:.2f}, Max={results['feature_dist_train']['max']:.2f}")
    print(f"Feature Distance to Val NN:   Min={results['feature_dist_val']['min']:.2f}, Mean={results['feature_dist_val']['mean']:.2f}, Max={results['feature_dist_val']['max']:.2f}")
    print(f"Cosine Sim to Train NN:       Min={results['cosine_sim_train']['min']:.3f}, Mean={results['cosine_sim_train']['mean']:.3f}, Max={results['cosine_sim_train']['max']:.3f}")
    print(f"Cosine Sim to Val NN:         Min={results['cosine_sim_val']['min']:.3f}, Mean={results['cosine_sim_val']['mean']:.3f}, Max={results['cosine_sim_val']['max']:.3f}")
    print(f"Artifacts saved to {pair_plot} and {hist_plot}")

if __name__ == "__main__":
    run_memorization_audit()
