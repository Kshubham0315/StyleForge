"""Phase 5 Reproducibility & Checkpoint Verification Script."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from models.style_generator import StyleGenerator
from data.dataset import get_dataloader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpts = {
        "Phase3_Step1500": "experiments/phase3_convergence/checkpoints/checkpoint_final.pt",
        "Phase3_LogisticR1": "experiments/phase3_improvement_logistic_r1/checkpoints/checkpoint_final.pt",
        "Phase2_AdaIN_1000": "experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt",
    }

    results = {}
    for name, path in ckpts.items():
        p = Path(path)
        if not p.exists():
            print(f"FAILED: {name} not found at {path}")
            continue

        data = torch.load(p, map_location="cpu", weights_only=False)
        g = StyleGenerator(latent_dim=512, style_dim=512, img_size=64)
        g.load_state_dict(data["generator"])
        g = g.to(device).eval()

        step = data.get("global_step")

        # Test deterministic reproduction
        torch.manual_seed(42)
        z = torch.randn(4, 512, device=device)
        with torch.no_grad():
            out1 = g(z, truncation_psi=0.8)

        torch.manual_seed(42)
        z = torch.randn(4, 512, device=device)
        with torch.no_grad():
            out2 = g(z, truncation_psi=0.8)

        max_diff = float((out1 - out2).abs().max().item())
        has_ema = "ema_generator" in data

        print(f"[{name}] Step: {step} | Max Diff: {max_diff:.1e} | Has EMA: {has_ema} | Params: {sum(p.numel() for p in g.parameters()):,}")
        results[name] = {
            "step": step,
            "max_diff": max_diff,
            "has_ema": has_ema,
            "verified": max_diff == 0.0,
        }

    # Verify dataloader seed determinism
    loader1 = get_dataloader("data/celeba/img_align_celeba", split="val", img_size=64, batch_size=16, num_workers=0, seed=42)
    loader2 = get_dataloader("data/celeba/img_align_celeba", split="val", img_size=64, batch_size=16, num_workers=0, seed=42)
    b1 = next(iter(loader1))[0]
    b2 = next(iter(loader2))[0]
    data_diff = float((b1 - b2).abs().max().item())
    print(f"[Dataset Determinism] Val batch diff across fresh loaders: {data_diff:.1e}")

    return results

if __name__ == "__main__":
    main()
