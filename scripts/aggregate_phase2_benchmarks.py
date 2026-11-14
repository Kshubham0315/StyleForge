"""Aggregates all Phase 2 evaluation metrics into comparison tables (Markdown, JSON, CSV)."""

import json
from pathlib import Path
import pandas as pd


def aggregate():
    base_dir = Path("experiments")
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("Baseline WGAN-GP (1000 steps)", base_dir / "phase2_baseline_wgan_gp" / "eval" / "eval_phase2_metrics.json"),
        ("StyleForge AdaIN (EMA, 1000 steps)", base_dir / "phase2_stylegan_adain" / "eval" / "eval_phase2_metrics.json"),
        ("StyleForge AdaIN (Raw / No EMA)", base_dir / "phase2_ablation_no_ema" / "eval" / "eval_phase2_metrics.json"),
        ("Ablation A: Self-Attention (16x16)", base_dir / "phase2_ablation_self_attention" / "eval" / "eval_phase2_metrics.json"),
        ("Ablation B: Spectral Norm Critic", base_dir / "phase2_ablation_spectral_norm" / "eval" / "eval_phase2_metrics.json"),
    ]

    records = []
    for label, metric_file in experiments:
        if metric_file.exists():
            with open(metric_file, "r") as f:
                data = json.load(f)
            data["experiment_label"] = label
            records.append(data)
        else:
            print(f"Metrics not found for {label}: {metric_file}")

    if not records:
        print("No evaluation records found yet.")
        return

    df = pd.DataFrame(records)

    # Columns of interest
    cols = [
        "experiment_label",
        "sample_count_real",
        "sample_count_fake",
        "fid",
        "kid_mean",
        "kid_std",
        "inception_precision",
        "inception_recall",
        "pairwise_l1_diversity",
        "pairwise_l2_diversity",
        "batch_pixel_variance",
    ]
    avail_cols = [c for c in cols if c in df.columns]
    summary_df = df[avail_cols]

    # Save to JSON & CSV
    summary_df.to_json(results_dir / "phase2_model_comparison.json", orient="records", indent=2)
    summary_df.to_csv(results_dir / "phase2_model_comparison.csv", index=False)

    # Print markdown table
    print("\n=== Phase 2 Model Benchmark Comparison Table ===\n")
    print(summary_df.to_markdown(index=False))


if __name__ == "__main__":
    aggregate()
