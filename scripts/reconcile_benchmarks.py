"""Phase 5 Benchmark Reconciliation Script.

Evaluates historical and Phase 4 candidate models side-by-side using the exact
same Inception-v3 pool3 extractor and CelebA validation partition (N=614).
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from data.dataset import get_dataloader
from models.style_generator import StyleGenerator
from evaluation.diversity import compute_pairwise_distances

def to_uint8(t: torch.Tensor) -> torch.Tensor:
    return ((t * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).to(torch.uint8)

def evaluate_model(generator, val_loader, real_uint8_cache, device, latent_dim=512, seed=42):
    generator.eval()
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_metric = KernelInceptionDistance(feature=2048, subsets=20, subset_size=50, normalize=False).to(device)

    # Feed real images
    for u_real in real_uint8_cache:
        fid_metric.update(u_real.to(device), real=True)
        kid_metric.update(u_real.to(device), real=True)

    # Generate synthetic images matching validation size
    N_val = len(val_loader.dataset)
    torch.manual_seed(seed)
    fake_imgs_list = []
    count = 0
    with torch.no_grad():
        while count < N_val:
            cur_bs = min(32, N_val - count)
            z = torch.randn(cur_bs, latent_dim, device=device)
            f_img = generator(z, truncation_psi=0.8)
            fake_imgs_list.append(f_img.cpu())
            u_f = to_uint8(f_img)
            fid_metric.update(u_f, real=False)
            kid_metric.update(u_f, real=False)
            count += cur_bs

    fid_val = float(fid_metric.compute().item())
    kid_mean, kid_std = kid_metric.compute()

    # Diversity
    all_fake = torch.cat(fake_imgs_list, dim=0)
    fake_eval = all_fake[:200]
    flat = fake_eval.view(fake_eval.size(0), -1)
    num_pairs = min(1000, flat.size(0) * 2)
    idx1 = torch.randint(0, flat.size(0), (num_pairs,))
    idx2 = torch.randint(0, flat.size(0), (num_pairs,))
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    diff = flat[idx1] - flat[idx2]
    l1_div = float(torch.abs(diff).mean(dim=1).mean().item())
    var_div = float(fake_eval.var(dim=0).mean().item())

    return {
        "fid": fid_val,
        "kid_mean": float(kid_mean.item()),
        "kid_std": float(kid_std.item()),
        "l1_diversity": l1_div,
        "batch_variance": var_div,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_loader = get_dataloader("data/celeba/img_align_celeba", split="val", img_size=64, batch_size=32, num_workers=0, seed=42)
    real_uint8_cache = []
    for b in val_loader:
        if isinstance(b, (list, tuple)): b = b[0]
        real_uint8_cache.append(to_uint8(b))

    candidates = [
        {
            "name": "Phase 3 Step 1500 (Raw Generator)",
            "ckpt": "experiments/phase3_convergence/checkpoints/checkpoint_final.pt",
            "key": "generator",
            "steps": 1500,
            "objective": "WGAN-GP",
        },
        {
            "name": "Phase 3 Step 1500 (EMA Generator)",
            "ckpt": "experiments/phase3_convergence/checkpoints/checkpoint_final.pt",
            "key": "ema_generator",
            "steps": 1500,
            "objective": "WGAN-GP",
        },
        {
            "name": "Phase 2/3 Step 1000 (Raw Generator)",
            "ckpt": "experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt",
            "key": "generator",
            "steps": 1000,
            "objective": "WGAN-GP",
        },
        {
            "name": "Phase 3 Logistic + R1 (Raw Generator)",
            "ckpt": "experiments/phase3_improvement_logistic_r1/checkpoints/checkpoint_final.pt",
            "key": "generator",
            "steps": 300,
            "objective": "Logistic + R1",
        },
    ]

    benchmark_results = []
    for cand in candidates:
        print(f"\n--- Evaluating {cand['name']} ---")
        p = Path(cand["ckpt"])
        assert p.exists()
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        g = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
        state_dict = ckpt[cand["key"]]
        # Remove module. prefix if present
        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        g.load_state_dict(cleaned_state)

        res = evaluate_model(g, val_loader, real_uint8_cache, device, latent_dim=512, seed=42)
        res.update({
            "name": cand["name"],
            "steps": cand["steps"],
            "objective": cand["objective"],
            "weight_type": "Raw" if cand["key"] == "generator" else "EMA",
        })
        print(f"Result: FID={res['fid']:.2f} | KID={res['kid_mean']:.4f} | L1={res['l1_diversity']:.4f} | Var={res['batch_variance']:.5f}")
        benchmark_results.append(res)

    out_path = Path("reports/benchmark_reconciliation.json")
    with open(out_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\nBenchmark reconciliation evaluation complete.")

if __name__ == "__main__":
    main()
