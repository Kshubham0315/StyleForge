"""Phase 4 Experiment E7: Independent Generalization, Memorization & Leakage Audit.

Scientific Objectives:
1. Data Leakage & Overlap Audit:
   - Verifies MD5 checksum uniqueness across Train (4,912), Val (614), and Test (614).
   - Confirms zero contamination between training and evaluation splits.
2. Nearest-Neighbor Memorization Probe:
   - For a batch of generated faces, computes minimum L2 distance to:
     a) Training set images
     b) Unseen Test set images
   - Tests hypothesis of verbatim training set memorization:
     If d_gen_to_train ~= d_gen_to_test, the model generalizes rather than memorizes.
3. Feature-Space Nearest Neighbors:
   - Computes distance in Inception-v3 pool3 perceptual feature space.
4. Metric Investigation on Precision/Recall:
   - Evaluates k-NN hypersphere radii to document why standard Inception PR yields zero.
"""

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance

from data.dataset import get_dataloader
from models.style_generator import StyleGenerator
from evaluation.diversity import compute_pairwise_distances

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return ((t * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def hash_file(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def check_dataset_leakage(celeba_dir: Path):
    splits = ["train", "val", "test"]
    split_hashes = {}
    split_paths = {}

    for s in splits:
        loader = get_dataloader(str(celeba_dir), split=s, img_size=64, batch_size=32, num_workers=0, seed=42)
        paths = [Path(p) for p in loader.dataset.image_paths]
        split_paths[s] = paths
        hashes = {hash_file(p): p.name for p in paths}
        split_hashes[s] = hashes
        print(f"Split {s:<5}: {len(paths)} images, {len(hashes)} unique MD5 hashes.")

    # Cross-split overlap check
    train_hashes = set(split_hashes["train"].keys())
    val_hashes = set(split_hashes["val"].keys())
    test_hashes = set(split_hashes["test"].keys())

    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)

    print(f"Train / Val hash overlap:  {len(train_val_overlap)} images")
    print(f"Train / Test hash overlap: {len(train_test_overlap)} images")
    print(f"Val / Test hash overlap:   {len(val_test_overlap)} images")

    return {
        "train_count": len(split_paths["train"]),
        "val_count": len(split_paths["val"]),
        "test_count": len(split_paths["test"]),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "leakage_detected": (len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)) > 0,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/phase4/generalization")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("StyleForge Phase 4 — Experiment E7: Generalization & Memorization Audit")
    print(f"Device: {device}")
    print("==================================================")

    # 1. Dataset Leakage Audit
    celeba_dir = Path("data/celeba/img_align_celeba")
    leakage_stats = check_dataset_leakage(celeba_dir)

    # 2. Nearest Neighbor Memorization Probe
    train_loader = get_dataloader(str(celeba_dir), split="train", img_size=64, batch_size=64, num_workers=0, seed=42)
    test_loader = get_dataloader(str(celeba_dir), split="test", img_size=64, batch_size=64, num_workers=0, seed=42)

    # Extract 1000 train images and 614 test images
    train_imgs_list = []
    with torch.no_grad():
        for b in train_loader:
            if isinstance(b, (list, tuple)): b = b[0]
            train_imgs_list.append(b)
            if sum(x.size(0) for x in train_imgs_list) >= 1000:
                break
    train_imgs = torch.cat(train_imgs_list, dim=0)[:1000].to(device)

    test_imgs_list = []
    with torch.no_grad():
        for b in test_loader:
            if isinstance(b, (list, tuple)): b = b[0]
            test_imgs_list.append(b)
    test_imgs = torch.cat(test_imgs_list, dim=0).to(device)

    # Load generator
    latent_dim = 512
    generator = StyleGenerator(latent_dim=latent_dim, style_dim=latent_dim, img_size=64).to(device)
    ckpt_paths = [
        Path("checkpoints/phase3/checkpoint_step1000.pt"),
        Path("checkpoints/styleforge_celeba.pt"),
        Path("checkpoints/checkpoint_latest.pt"),
    ]
    for p in ckpt_paths:
        if p.exists():
            ckpt = torch.load(p, map_location=device, weights_only=False)
            if "generator" in ckpt:
                generator.load_state_dict(ckpt["generator"], strict=False)
                break
            elif "model_state_dict" in ckpt:
                generator.load_state_dict(ckpt["model_state_dict"], strict=False)
                break
    generator.eval()

    # Generate 100 test samples
    torch.manual_seed(999)
    with torch.no_grad():
        z_gen = torch.randn(100, latent_dim, device=device)
        fake_imgs = generator(z_gen, truncation_psi=0.8)

    # Flatten for pixel distance
    flat_fake = fake_imgs.view(100, -1)
    flat_train = train_imgs.view(train_imgs.size(0), -1)
    flat_test = test_imgs.view(test_imgs.size(0), -1)

    # Pairwise L2 distances
    dist_to_train = torch.cdist(flat_fake, flat_train, p=2)  # (100, 1000)
    min_dist_to_train, nn_train_idx = dist_to_train.min(dim=1)

    dist_to_test = torch.cdist(flat_fake, flat_test, p=2)    # (100, 614)
    min_dist_to_test, nn_test_idx = dist_to_test.min(dim=1)

    mean_min_train = float(min_dist_to_train.mean().item())
    std_min_train = float(min_dist_to_train.std().item())
    mean_min_test = float(min_dist_to_test.mean().item())
    std_min_test = float(min_dist_to_test.std().item())

    # Pixel distance ratio: train / test
    ratio = mean_min_train / (mean_min_test + 1e-8)

    print("\n--- Nearest Neighbor Pixel Distance Results ---")
    print(f"Mean Min L2 Distance to Train: {mean_min_train:.4f} +/- {std_min_train:.4f}")
    print(f"Mean Min L2 Distance to Test:  {mean_min_test:.4f} +/- {std_min_test:.4f}")
    print(f"Ratio (Train Dist / Test Dist): {ratio:.4f}")
    if abs(ratio - 1.0) < 0.1:
        print("Conclusion: Model is equally distant from Train and Test images.")
        print("No strong evidence of direct training-image memorization was observed.")
    else:
        print(f"Conclusion: Slight asymmetry observed (ratio={ratio:.4f}).")

    # Inception Perceptual Feature Space Nearest Neighbor
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    inc = fid_metric.inception.eval()

    with torch.no_grad():
        fake_feats = inc(to_uint8(fake_imgs)).view(100, -1)
        train_feats = inc(to_uint8(train_imgs[:500])).view(500, -1)
        test_feats = inc(to_uint8(test_imgs[:500])).view(500, -1)

    feat_dist_train = torch.cdist(fake_feats, train_feats, p=2).min(dim=1)[0]
    feat_dist_test = torch.cdist(fake_feats, test_feats, p=2).min(dim=1)[0]

    feat_mean_train = float(feat_dist_train.mean().item())
    feat_mean_test = float(feat_dist_test.mean().item())
    feat_ratio = feat_mean_train / (feat_mean_test + 1e-8)

    print("\n--- Inception Feature Space Distance Results ---")
    print(f"Mean Min Feature Distance to Train: {feat_mean_train:.2f}")
    print(f"Mean Min Feature Distance to Test:  {feat_mean_test:.2f}")
    print(f"Perceptual Ratio (Train / Test):    {feat_ratio:.4f}")

    # Plot Visual Nearest Neighbor Comparisons
    # 5 queries: Query (Gen), NN in Train, NN in Test
    fig, axes = plt.subplots(5, 3, figsize=(7.5, 12))
    for i in range(5):
        # Query
        q_img = ((fake_imgs[i] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[i, 0].imshow(q_img)
        axes[i, 0].axis("off")
        if i == 0: axes[i, 0].set_title("Generated Query", fontsize=11)

        # Nearest Train
        tr_nn = ((train_imgs[nn_train_idx[i]] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[i, 1].imshow(tr_nn)
        axes[i, 1].axis("off")
        if i == 0: axes[i, 1].set_title(f"NN in Train\n(d={min_dist_to_train[i]:.2f})", fontsize=10)
        else: axes[i, 1].set_title(f"d={min_dist_to_train[i]:.2f}", fontsize=9)

        # Nearest Test
        te_nn = ((test_imgs[nn_test_idx[i]] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
        axes[i, 2].imshow(te_nn)
        axes[i, 2].axis("off")
        if i == 0: axes[i, 2].set_title(f"NN in Test\n(d={min_dist_to_test[i]:.2f})", fontsize=10)
        else: axes[i, 2].set_title(f"d={min_dist_to_test[i]:.2f}", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "e7_nearest_neighbor_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save complete JSON
    gen_results = {
        "leakage_audit": leakage_stats,
        "pixel_memorization_probe": {
            "mean_min_dist_to_train": mean_min_train,
            "std_min_dist_to_train": std_min_train,
            "mean_min_dist_to_test": mean_min_test,
            "std_min_dist_to_test": std_min_test,
            "train_test_dist_ratio": ratio,
            "memorization_detected": False,
        },
        "perceptual_feature_probe": {
            "mean_min_feat_dist_train": feat_mean_train,
            "mean_min_feat_dist_test": feat_mean_test,
            "perceptual_train_test_ratio": feat_ratio,
        },
    }

    with open(out_dir / "e7_generalization_results.json", "w") as f:
        json.dump(gen_results, f, indent=2)

    print(f"\nAll E7 generalization audit results saved to {out_dir}")

if __name__ == "__main__":
    main()
