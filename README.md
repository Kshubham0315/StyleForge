# StyleForge

A research-grade StyleGAN-inspired face generation and latent exploration system built with PyTorch, focused on GAN training stability, style-based generation, latent space disentanglement, and quantitative evaluation.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests: 24 Passing](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen.svg)](tests/)

---

## Highlights

- **Dual Generator Architectures**: Classical convolutional WGAN-GP baseline and full StyleGAN-style generator (8-layer Mapping Network $f: \mathcal{Z} \to \mathcal{W}$ with PixelNorm and equalized learning rate He scaling, learned constant spatial input $4 \times 4 \times 512$, AdaIN blocks, and per-channel stochastic noise injection).
- **Stabilized Training Pipeline**: Decoupled Two-Sided Gradient Penalty (WGAN-GP) and R1 regularizers, Minibatch Standard Deviation to prevent mode collapse, Exponential Moving Average (EMA) shadow weights, and mixed-precision (FP16 AMP) acceleration with numerical FP32 autograd safeguards.
- **Architectural Ablations**: Controlled configurations for SAGAN Self-Attention blocks ($16\times 16$) and Spectral Normalization in the Multi-scale Residual Critic.
- **Latent Space Exploration**: Spherical Linear Interpolation (SLERP) in $\mathcal{Z}$-space, Linear Interpolation (LERP) in $\mathcal{W}$-space, multi-scale style mixing across coarse/middle/fine layers, and Truncation Trick ($\psi \in [0.0, 1.0]$).
- **Quantitative Evaluation Suite**: Pairwise pixel diversity (L1/L2), batch pixel variance (mode collapse diagnostics), manifold precision and recall ($k$-NN manifold approximation), and official TorchMetrics Inception-V3 FID/KID metrics.
- **Hardware-Optimized**: Calibrated and profiled for consumer GPUs (tested on NVIDIA GeForce RTX 4050 6GB VRAM, using 3.5 GB peak VRAM at $64 \times 64$).

---

## Experimental Benchmark: Baseline vs StyleGAN AdaIN (Extended 600 Steps)

Both models evaluated on real CelebA validation images ($64 \times 64$, normalized to $[-1, 1]$) under identical evaluation seed ($42$):

| Metric | Baseline WGAN-GP (600 Steps) | StyleForge AdaIN (600 Steps) | StyleForge + Self-Attention (150 Steps) | Finding / Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Model Type** | Convolutional Transpose WGAN | Style-Based + AdaIN | Style-Based + SAGAN | Multi-scale modulation |
| **Inception-v3 FID $\downarrow$** | **349.15** | 470.89 | **445.12** | Self-attention drops FID by 25.7 pts |
| **KID Mean $\downarrow$** | **0.50802** ($\pm 0.007$) | 0.70576 ($\pm 0.018$) | 0.66166 ($\pm 0.012$) | Standard Inception-v3 polynomial kernel |
| **Pairwise L1 Distance** | 0.0415 | **0.6109** | **0.7010** | **$14.7\times$ higher diversity** |
| **Batch Pixel Variance** | 0.0017 | **0.3519** | **0.3994** | **$207\times$ higher variance (No collapse)** |
| **Manifold Recall** | 0.0000 | **1.0000** | 0.9600 | Full manifold support coverage |
| **VRAM Footprint** | 743 MB | 3,475 MB (~3.5 GB) | 3,461 MB (~3.5 GB) | Fits safely in 6GB VRAM |
| **Throughput** | 5.62 it/s | 1.12 it/s | 1.05 it/s | Baseline is $5\times$ faster |

Detailed metrics, logs, and comparative charts can be found in [docs/model_comparison.md](docs/model_comparison.md) and [docs/metrics_analysis.md](docs/metrics_analysis.md).

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
