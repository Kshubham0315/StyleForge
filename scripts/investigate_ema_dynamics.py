"""Script to inspect Raw vs EMA parameters, output differences, and bias dynamics."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.style_generator import StyleGenerator
from evaluation.diversity import compute_pairwise_distances, compute_precision_recall

def analyze_checkpoint_ema(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    raw_sd = ckpt["generator"]
    ema_sd = ckpt.get("ema_generator", None)
    if ema_sd is None:
        print("No ema_generator found in checkpoint!")
        return

    print("=== PARAMETER LEVEL AUDIT ===")
    total_params = 0
    l2_diff_sum = 0.0
    cos_sims = []

    for name in raw_sd.keys():
        p_raw = raw_sd[name].float()
        p_ema = ema_sd[name].float()
        total_params += p_raw.numel()
        diff = torch.norm(p_raw - p_ema, p=2).item()
        l2_diff_sum += diff ** 2
        
        flat_raw = p_raw.view(-1)
        flat_ema = p_ema.view(-1)
        if flat_raw.norm() > 1e-7 and flat_ema.norm() > 1e-7:
            cos = torch.cosine_similarity(flat_raw.unsqueeze(0), flat_ema.unsqueeze(0)).item()
            cos_sims.append((name, cos, diff))

    total_l2_diff = (l2_diff_sum) ** 0.5
    avg_cos_sim = sum(c[1] for c in cos_sims) / len(cos_sims)
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Parameter L2 Norm Difference: {total_l2_diff:.4f}")
    print(f"Average Layer-wise Cosine Similarity: {avg_cos_sim:.4f}")
    
    # Sort layers by lowest cosine similarity (greatest divergence between raw and EMA)
    cos_sims.sort(key=lambda x: x[1])
    print("\nTop 5 Most Diverged Layers (Lowest Cosine Similarity):")
    for name, cos, diff in cos_sims[:5]:
        print(f"  {name:50s} | Cos Sim: {cos:.4f} | L2 Diff: {diff:.4f}")

    print("\nTop 5 Most Aligned Layers (Highest Cosine Similarity):")
    for name, cos, diff in cos_sims[-5:]:
        print(f"  {name:50s} | Cos Sim: {cos:.4f} | L2 Diff: {diff:.4f}")

    # Theoretical Bias analysis
    step = ckpt.get("step", 1000)
    print(f"\n=== THEORETICAL BIAS ANALYSIS (Step {step}) ===")
    for decay in [0.99, 0.995, 0.999]:
        init_weight_fraction = (decay ** step)
        effective_moving_window = 1.0 / (1.0 - decay)
        print(f"Decay {decay:5.3f}: Initial Weight Residual = {init_weight_fraction*100:6.2f}% | Effective Window = {effective_moving_window:6.1f} steps")

    # Output Difference on Fixed Latents
    print("\n=== OUTPUT DIFFERENCE ON FIXED LATENTS (N=64) ===")
    gen_raw = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device).eval()
    gen_ema = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device).eval()
    gen_raw.load_state_dict(raw_sd)
    gen_ema.load_state_dict(ema_sd)

    torch.manual_seed(42)
    z = torch.randn(64, 512, device=device)
    with torch.no_grad():
        out_raw = gen_raw(z, truncation_psi=0.8)
        out_ema = gen_ema(z, truncation_psi=0.8)

    pixel_l1 = torch.abs(out_raw - out_ema).mean().item()
    pixel_l2 = torch.norm(out_raw - out_ema, p=2, dim=[1,2,3]).mean().item()
    max_pixel_diff = torch.abs(out_raw - out_ema).max().item()
    print(f"Pixel L1 Difference:     {pixel_l1:.4f} (on [-1, 1] range)")
    print(f"Pixel L2 Difference:     {pixel_l2:.4f}")
    print(f"Max Pixel Difference:    {max_pixel_diff:.4f}")

    out_dict = {
        "step": step,
        "total_params": total_params,
        "total_l2_diff": total_l2_diff,
        "avg_cos_sim": avg_cos_sim,
        "pixel_l1_diff": pixel_l1,
        "pixel_l2_diff": pixel_l2,
        "max_pixel_diff": max_pixel_diff,
        "diverged_layers": [(x[0], x[1], x[2]) for x in cos_sims[:5]],
    }
    return out_dict

if __name__ == "__main__":
    analyze_checkpoint_ema("experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt")
