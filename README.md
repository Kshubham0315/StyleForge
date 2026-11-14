# StyleForge

A research-grade StyleGAN-inspired face generation and latent exploration system built with PyTorch, focused on GAN training stability, style-based generation, latent space disentanglement, and quantitative evaluation.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests: 36 Passing](https://img.shields.io/badge/Tests-36%2F36%20Passing-brightgreen.svg)](tests/)
[![Phase: 4 Milestone](https://img.shields.io/badge/Phase%204-Architecture%20Optimization%20%26%20Generalization-success.svg)](reports/PHASE4_FINAL_REPORT.md)

---

## Highlights

- **Authentic Photographic CelebA Benchmark**: Trained and evaluated on 6,140 verified authentic photographic human faces ($178\times 218$) from the aligned & cropped CelebA dataset, with strictly disjoint deterministic Train (4,912), Validation (614), and Test (614) partitions (zero leakage).
- **Dual Generator Architectures**: Classical convolutional WGAN-GP baseline and full StyleGAN-style generator (8-layer Mapping Network $f: \mathcal{Z} \to \mathcal{W}$ with PixelNorm and equalized learning rate He scaling, learned constant spatial input $4 \times 4 \times 512$, AdaIN blocks, and per-channel stochastic noise injection).
- **Contrast Collapse Discovery**: Empirical proof of mode and contrast collapse in the convolutional baseline (pixel range narrows to $[0, 141]$, $\sigma=29.6$, batch variance $0.00153$), whereas StyleForge AdaIN maintains full $[0, 255]$ dynamic range with $242\times$ higher batch variance ($0.37069$).
- **Stabilized Training Pipeline**: Decoupled Two-Sided Gradient Penalty (WGAN-GP) and R1 regularizers, Minibatch Standard Deviation to prevent mode collapse, Exponential Moving Average (EMA) shadow weights, and calibrated pure FP32 execution preventing gradient underflow.
- **Architectural Ablations**: Controlled configurations for SAGAN Self-Attention blocks ($16\times 16$), Spectral Normalization in the Multi-scale Residual Critic, and EMA vs Instantaneous weights.
- **Latent Space Exploration**: Spherical Linear Interpolation (SLERP) in $\mathcal{Z}$-space, Linear Interpolation (LERP) in $\mathcal{W}$-space, multi-scale style mixing across coarse/middle/fine layers, and Truncation Trick ($\psi \in [0.0, 1.0]$).
- **Quantitative Evaluation Suite**: Pairwise pixel diversity (L1/L2), batch pixel variance (mode collapse diagnostics), manifold precision and recall ($k$-NN manifold approximation), and official TorchMetrics Inception-V3 FID/KID metrics evaluated on matched sample sets ($N=614$).
- **Hardware-Optimized**: Calibrated and profiled for consumer GPUs (tested on NVIDIA GeForce RTX 4050 6GB VRAM, using 3.36 GB peak reserved VRAM at $64 \times 64$).

---

## Phase 2 Experimental Benchmark: Authentic CelebA (1,000 Steps, N=614)

All models evaluated on the complete, disjoint CelebA photographic validation partition ($N=614$, $64 \times 64$, normalized to $[-1, 1]$) against an equal number of generated samples under identical evaluation seed ($42$):

| Metric | Baseline WGAN-GP | StyleForge AdaIN (EMA) | StyleForge AdaIN (No EMA) | Ablation A (Self-Attn 16x16) | Ablation B (Spectral Norm) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Steps** | 1,000 | 1,000 | 1,000 | 300 | 300 |
| **Inception-v3 FID $\downarrow$** | 344.90 | 383.71 | **305.99** | 404.19 | 417.51 |
| **KID Mean $\downarrow$** | $0.4225 \pm 0.0098$ | $0.5246 \pm 0.0171$ | **$0.3531 \pm 0.0167$** | $0.5855 \pm 0.0130$ | $0.6132 \pm 0.0210$ |
| **Pairwise L1 Distance $\uparrow$** | 0.0410 | 0.6512 | 0.5794 | 0.6918 | **0.7123** |
| **Pairwise L2 Distance $\uparrow$** | 5.91 | 94.00 | 84.45 | 97.81 | **102.37** |
| **Batch Pixel Variance $\uparrow$** | 0.00153 | 0.37069 | 0.31068 | 0.39405 | **0.43339** |
| **Dynamic Contrast Range** | Collapsed `[0, 141]` | **Full `[0, 255]`** | **Full `[0, 255]`** | **Full `[0, 255]`** | **Full `[0, 255]`** |
| **Peak Reserved VRAM** | **271 MB** | 3,363 MB | 3,363 MB | 3,510 MB | 3,486 MB |

Detailed Phase 2 reports, audit logs, and comparative charts:
- [docs/phase2_audit.md](docs/phase2_audit.md) — Comprehensive dataset and metric audit
- [docs/scaled_training_results.md](docs/scaled_training_results.md) — 1,000-step training dynamics and convergence
- [docs/reliable_evaluation.md](docs/reliable_evaluation.md) — Mathematical evaluation of FID, KID, and Precision/Recall
- [docs/visual_quality_analysis.md](docs/visual_quality_analysis.md) — SLERP, LERP, truncation, and style mixing
- [docs/ablation_study.md](docs/ablation_study.md) — Controlled ablations (Attention, Spectral Norm, EMA)
- [docs/phase2_limitations.md](docs/phase2_limitations.md) — Transparent research boundaries and hardware limits

---

## Phase 3 Experimental Benchmark: Quality Audit & Optimization (N=614)

All evaluations conducted on the full, disjoint CelebA validation set ($N=614$) with official TorchMetrics Inception-v3:

### 1. Extended Convergence Trajectory (Steps 1,000 to 1,500)
| Checkpoint | Steps | Inception-v3 FID $\downarrow$ | KID Mean $\downarrow$ ($\pm \text{std}$) | Pairwise L1 Diversity $\uparrow$ | Batch Pixel Variance $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **StyleForge AdaIN (EMA)** | 1,000 | 383.71 | $0.5246 \pm 0.0171$ | 0.6512 | 0.37069 |
| **StyleForge AdaIN (EMA)** | 1,250 | 358.29 | $0.4635 \pm 0.0156$ | 0.6286 | 0.35165 |
| **StyleForge AdaIN (EMA)** | 1,500 | **336.98** | **$0.4167 \pm 0.0156$** | 0.6086 | 0.33447 |
| **StyleForge AdaIN (Raw)** | 1,000 | 305.99 | $0.3531 \pm 0.0167$ | 0.5794 | 0.31068 |
| **StyleForge AdaIN (Raw)** | 1,500 | **297.76** | **$0.3285 \pm 0.0159$** | 0.5635 | 0.28934 |

### 2. Multi-EMA Sweep & Objective Optimization (300 Steps)
| Configuration | Objective & Regularizer | Initial Weight Residual | Inception FID $\downarrow$ | KID Mean $\downarrow$ | Pairwise L1 $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Raw Instantaneous** | WGAN-GP ($\lambda=10.0$) | 0.00% | 381.61 | 0.5178 | 0.6354 |
| **EMA decay = 0.990** | WGAN-GP ($\lambda=10.0$) | 4.90% | 393.73 | 0.5468 | 0.6605 |
| **EMA decay = 0.995** | WGAN-GP ($\lambda=10.0$) | 22.23% | 409.50 | 0.5885 | 0.6864 |
| **EMA decay = 0.999** | WGAN-GP ($\lambda=10.0$) | 74.06% | 419.60 | 0.6186 | **0.7159** |
| **Phase F: Logistic + R1 (Raw)** | **Logistic + R1 ($\gamma=10.0$)** | 0.00% | **373.66** | **0.4975** | **0.6624** |

Detailed Phase 3 research reports:
- [docs/phase3_metric_audit.md](docs/phase3_metric_audit.md) — Mathematical verification of Precision/Recall & Inception geometry
- [docs/phase3_ema_analysis.md](docs/phase3_ema_analysis.md) — Parameter lag, weight residual, and decay sweep analysis
- [docs/phase3_visual_quality.md](docs/phase3_visual_quality.md) — Anatomical audit across facial landmarks and artifacts
- [docs/phase3_memorization_audit.md](docs/phase3_memorization_audit.md) — Train vs Val nearest-neighbor analysis in pixel and feature space
- [docs/phase3_training_results.md](docs/phase3_training_results.md) — Extended 1,500-step convergence and sub-300 FID milestone

---

## Phase 4 Experimental Benchmark: Architecture Optimization & Generalization

All evaluations conducted on the full, disjoint CelebA photographic validation partition ($N=614, 64 \times 64$) with official TorchMetrics Inception-v3:

### 1. Master Milestone Benchmark

| Experiment / Milestone | Configuration | Inception-v3 FID $\downarrow$ | KID Mean $\downarrow$ ($\times 10^3$) | Pairwise L1 Diversity $\uparrow$ | Batch Pixel Variance $\uparrow$ | Hardware / Memory | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1: Raw Instantaneous** | Instantaneous weights | $381.75$ | $517.1$ | $0.6355$ | $0.36877$ | $1.42\text{ GB}$ | Verified |
| **E1: Uncorrected EMA** | $\beta=0.999$, static | $419.32$ | $618.4$ | $0.7158$ | $0.43513$ | $1.42\text{ GB}$ | Inertia Lag |
| **E1: Debiased EMA** | Model-init debiased | $404.24$ | $574.3$ | $0.6666$ | $0.39945$ | $1.42\text{ GB}$ | -15.1 pts vs uncorr. |
| **E1: Warmup EMA (Best)** | Dynamic warmup schedule | **$383.01$** | **$518.1$** | $0.6419$ | $0.37611$ | $1.42\text{ GB}$ | **-36.3 pts vs uncorr.** |
| **E2: WGAN-GP** | Multi-seed ($N=3$) | $394.25 \pm 6.86$ | $552.5 \pm 13.3$ | **$0.6499 \pm 0.020$** | **$0.39957 \pm 0.011$** | $1.42\text{ GB}$ | Highly Robust |
| **E2: Logistic (Unreg.)** | Multi-seed ($N=3$) | **$369.06 \pm 18.22$** | **$491.9 \pm 37.6$** | $0.3924 \pm 0.133$ | $0.21875 \pm 0.093$ | $1.08\text{ GB}$ | High Seed Variance |
| **E2: Logistic + $R_1$** | Multi-seed ($N=3$) | $374.54 \pm 12.13$ | $505.2 \pm 26.9$ | $0.4227 \pm 0.150$ | $0.23783 \pm 0.107$ | $1.42\text{ GB}$ | **Optimal Quality/Stability** |
| **E4: Scaled Critic** | Base channels 128 | $393.78$ | $551.0$ | $0.6450$ | $0.39100$ | $2.28\text{ GB}$ | $+120\%$ latency |
| **E6: 128×128 (bs=8)** | 128×128 Resolution | — | — | — | — | $1.84\text{ GB}$ | $4.15\text{ GB}$ Headroom |
| **E6: 128×128 (bs=16)** | 128×128 Resolution | — | — | — | — | $3.41\text{ GB}$ | $2.59\text{ GB}$ Headroom |
| **E7: Gen. Audit** | Nearest-Neighbor Probe | — | — | Pixel Ratio: $0.9806$ | Feat Ratio: $0.9996$ | — | **No Memorization** |

### 2. Comprehensive Phase 4 Research Reports
- **[reports/PHASE4_FINAL_REPORT.md](reports/PHASE4_FINAL_REPORT.md)** — **Master Phase 4 Final Research Report**
- [reports/PHASE4_EMA_ANALYSIS.md](reports/PHASE4_EMA_ANALYSIS.md) — E1 EMA Warmup & Debiasing Optimization
- [reports/PHASE4_OBJECTIVE_COMPARISON.md](reports/PHASE4_OBJECTIVE_COMPARISON.md) — E2 Multi-Seed Controlled Objective Comparison
- [reports/PHASE4_ARTIFACT_ANALYSIS.md](reports/PHASE4_ARTIFACT_ANALYSIS.md) — E3 Water-Droplet & 2D FFT Spectral Frequency Decomposition
- [reports/PHASE4_CAPACITY_ANALYSIS.md](reports/PHASE4_CAPACITY_ANALYSIS.md) — E4 & E5 Architecture Capacity & Resource Scaling
- [reports/PHASE4_RESOLUTION_ANALYSIS.md](reports/PHASE4_RESOLUTION_ANALYSIS.md) — E6 128×128 Progressive Scaling Feasibility
- [reports/PHASE4_GENERALIZATION_REPORT.md](reports/PHASE4_GENERALIZATION_REPORT.md) — E7 Generalization, Memorization & Leakage Audit

---

## Project Structure

```text
StyleForge/
├── configs/
│   ├── baseline_wgan_gp.yaml        # Reference WGAN-GP baseline configuration
│   ├── stylegan_adain.yaml          # Style-based generator with AdaIN & mapping network
│   ├── ablation_spectral_norm.yaml  # Spectral normalization critic ablation
│   ├── ablation_self_attention.yaml # Self-attention SAGAN ablation
│   └── quick_run.yaml               # Rapid smoke-test configuration
├── data/
│   ├── __init__.py
│   └── dataset.py                   # CelebA loader, center crop, deterministic train/val splits
├── docs/
│   ├── research_audit.md            # Comprehensive repository and metric audit
│   ├── architecture.md              # Detailed mathematical specifications
│   ├── training.md                  # Losses, AMP stability, EMA, and VRAM management
│   ├── training_results.md          # 600-step empirical training trajectories
│   ├── metrics_analysis.md          # FID, KID, diversity, and precision/recall analysis
│   ├── model_comparison.md          # Fair model comparison and trade-offs
│   └── limitations.md               # Honest research limitations and next experiments
├── models/
│   ├── __init__.py
│   ├── common.py                    # AdaIN, PixelNorm, MinibatchStdDev, SelfAttention, Equalized layers
│   ├── wgan_baseline.py             # Classical DCGAN/WGAN-GP generator & critic
│   ├── style_generator.py           # MappingNetwork, SynthesisNetwork, and StyleGenerator
│   └── critic.py                    # Multi-scale residual critic with MinibatchStdDev & SN
├── training/
│   ├── __init__.py
│   ├── losses.py                    # Decoupled WGAN-GP, R1 penalty, and logistic losses
│   └── trainer.py                   # GANTrainer with EMA, AMP, metrics logging & checkpointing
├── evaluation/
│   ├── __init__.py
│   ├── fid_kid.py                   # Official Inception-V3 FID & KID evaluator
│   ├── latent_explorer.py           # SLERP, LERP, truncation sweep & style mixing matrix
│   └── diversity.py                 # Mode collapse, pairwise diversity, and precision/recall
├── scripts/
│   ├── train.py                     # Training CLI with checkpoint resume
│   ├── generate.py                  # Image generation & latent exploration CLI
│   ├── evaluate.py                  # Quantitative metrics CLI
│   ├── verify_dataset.py            # Dataset integrity and zero-leakage validator
│   ├── validate_and_generate_research.py # Checkpoint validation & artifact generator
│   └── download_celeba.py           # CelebA terms helper & archive unpacker
├── tests/
│   ├── test_models.py               # Unit tests for models & layers
│   ├── test_losses.py               # Unit tests for losses & GP/R1 penalties
│   ├── test_latent_explorer.py      # Unit tests for SLERP, LERP & style mixing
│   └── test_integration.py         # Integration tests: dataset, resume, EMA, seed reproducibility
├── outputs/                         # Research sample grids, SLERP strips, truncation sweeps
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Kshubham0315/StyleForge.git
cd StyleForge

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Test Suite (24/24 Passing)

Run the comprehensive unit and integration test suite:

```bash
python -m pytest tests/ -v
```

### 3. Dataset Pipeline & Verification

1. Place CelebA images in `data/celeba/img_align_celeba/` (or use `scripts/download_celeba.py`).
2. Run the automated dataset verification script to certify image integrity, channel dimensions, and guarantee zero train-val leakage:

```bash
python scripts/verify_dataset.py --data_dir data/celeba/img_align_celeba --val_ratio 0.1
```

---

## Training Experiments

### Train StyleGAN AdaIN (Main Architecture)

```bash
python scripts/train.py --config configs/stylegan_adain.yaml
```

To resume an existing training run:

```bash
python scripts/train.py --config configs/stylegan_adain.yaml --resume experiments/stylegan_adain/checkpoints/checkpoint_latest.pt
```

Checkpoints will be saved automatically to `experiments/stylegan_adain/checkpoints/` and training curves to `experiments/stylegan_adain/training_curves.png`.

### Train Baseline WGAN-GP

```bash
python scripts/train.py --config configs/baseline_wgan_gp.yaml
```

---

## Quantitative Evaluation (Official Inception-v3 FID & KID)

Evaluate any trained checkpoint for diversity, mode collapse, manifold precision/recall, and Inception-V3 FID/KID:

```bash
python scripts/evaluate.py \
    --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_final.pt \
    --num_samples 64 \
    --out_json experiments/stylegan_adain/eval_metrics.json \
    --out_csv experiments/stylegan_adain/eval_metrics.csv
```

---

## Research Validation & Artifact Generation

Run the certified research validation script to load checkpoints, verify reproducibility, and generate publication-ready artifact panels:

```bash
python scripts/validate_and_generate_research.py
```

Generated artifact outputs:
- `outputs/research/baseline/samples_64.png`
- `outputs/research/baseline/interpolation_slerp.png`
- `outputs/research/styleforge/samples_64.png`
- `outputs/research/styleforge/interpolation_slerp.png`
- `outputs/research/styleforge/truncation_sweep.png`
- `outputs/research/styleforge/style_mixing_matrix.png`
- `outputs/research/comparisons/baseline_vs_styleforge.png`
- `experiments/results/model_comparison.png`

---

## Research Documentation

- [Research Audit (docs/research_audit.md)](docs/research_audit.md): Complete audit of implementations, data split, and metric definitions.
- [Training Results (docs/training_results.md)](docs/training_results.md): Extended 600-step training trajectories and gradient statistics.
- [Metrics Analysis (docs/metrics_analysis.md)](docs/metrics_analysis.md): Inception-v3 FID, KID, and pixel diversity vs perceptual quality.
- [Model Comparison (docs/model_comparison.md)](docs/model_comparison.md): Fair comparison and architectural trade-offs.
- [Limitations & Next Experiments (docs/limitations.md)](docs/limitations.md): Transparent analysis of dataset scale, training steps, and future extensions.
- [Architecture Specification (docs/architecture.md)](docs/architecture.md): Mathematical formulations for EqualizedLinear, AdaIN, Mapping Network, and Multi-Scale Critic.

---

## License & Dataset Terms

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
CelebA dataset usage is restricted to non-commercial research purposes in accordance with the official [CelebA Dataset Terms](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
