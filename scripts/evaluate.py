"""StyleForge Quantitative Evaluation CLI.

Evaluates trained checkpoint on:
- Frechet Inception Distance (FID)
- Kernel Inception Distance (KID)
- Pairwise Pixel Diversity (L1 / L2)
- Batch Pixel Variance (Mode collapse check)

Usage:
    python scripts/evaluate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --num_samples 200
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import torch

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import get_dataloader
from evaluation.fid_kid import MetricsEvaluator
from evaluation.diversity import DiversityEvaluator
from scripts.generate import load_generator_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="StyleForge Quantitative Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to real dataset (optional)")
    parser.add_argument("--out_json", type=str, default=None, help="Path to save output JSON metrics")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator, config = load_generator_from_checkpoint(args.checkpoint, device)

    m_cfg = config.get("model", {})
    latent_dim = m_cfg.get("latent_dim", 512)
    img_size = m_cfg.get("img_size", 64)

    dataset_path = args.dataset_path or config.get("data", {}).get("dataset_path")
    use_synthetic = config.get("data", {}).get("use_synthetic", False)

    print("==================================================")
    print(f"StyleForge Evaluation: {args.checkpoint}")
    print(f"Evaluating {args.num_samples} samples on {device}...")
    print("==================================================")

    # 1. Diversity & Mode Collapse
    print("1. Computing diversity & mode collapse metrics...")
    div_eval = DiversityEvaluator(device=device)
    div_results = div_eval.evaluate_diversity(
        generator,
        num_samples=min(args.num_samples, 256),
        latent_dim=latent_dim,
    )

    print(f"  - Pairwise L1 Distance: {div_results['pairwise_l1_distance']:.4f}")
    print(f"  - Pairwise L2 Distance: {div_results['pairwise_l2_distance']:.4f}")
    print(f"  - Batch Pixel Variance: {div_results['batch_pixel_variance']:.4f}")
    print(f"  - Diversity Score:     {div_results['diversity_score']:.2f}")

    # 2. FID & KID
    print("\n2. Computing FID & KID against real/reference distribution...")
    dataloader = get_dataloader(
        dataset_path=dataset_path,
        img_size=img_size,
        batch_size=32,
        use_synthetic=use_synthetic,
        synthetic_samples=max(args.num_samples, 200),
    )

    metrics_eval = MetricsEvaluator(device=device, feature_dim=2048)
    fid_results = metrics_eval.evaluate(
        generator=generator,
        real_dataloader=dataloader,
        num_samples=args.num_samples,
        latent_dim=latent_dim,
    )

    print(f"  - FID:      {fid_results['fid']:.3f}")
    print(f"  - KID Mean: {fid_results['kid_mean']:.5f} (±{fid_results['kid_std']:.5f})")

    combined_results = {
        "checkpoint": args.checkpoint,
        "config": config,
        "metrics": {
            **div_results,
            **fid_results,
        }
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(combined_results, f, indent=2)
        print(f"\nFull evaluation report saved to: {out_path}")


if __name__ == "__main__":
    main()
