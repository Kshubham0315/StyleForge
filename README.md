# StyleForge

A research-grade StyleGAN-inspired face generation and latent exploration system built with PyTorch, focused on GAN training stability, style-based generation, latent space disentanglement, and quantitative evaluation.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests: 20 Passing](https://img.shields.io/badge/Tests-20%2F20%20Passing-brightgreen.svg)](tests/)

---

## Highlights

- **Dual Generator Architectures**: Classical convolutional WGAN-GP baseline and full StyleGAN-style generator (8-layer Mapping Network $f: \mathcal{Z} \to \mathcal{W}$ with PixelNorm and equalized learning rate He scaling, learned constant spatial input $4 \times 4 \times 512$, AdaIN blocks, and per-channel stochastic noise injection).
- **Stabilized Training Pipeline**: Decoupled Two-Sided Gradient Penalty (WGAN-GP) and R1 regularizers, Minibatch Standard Deviation to prevent mode collapse, Exponential Moving Average (EMA) shadow weights, and mixed-precision (FP16 AMP) acceleration with numerical FP32 autograd safeguards.
- **Architectural Ablations**: Controlled configurations for SAGAN Self-Attention blocks and Spectral Normalization in the Multi-scale Residual Critic.
- **Latent Space Exploration**: Spherical Linear Interpolation (SLERP) in $\mathcal{Z}$-space, Linear Interpolation (LERP) in $\mathcal{W}$-space, multi-scale style mixing across coarse/middle/fine layers, and Truncation Trick ($\psi \in [0.0, 1.0]$).
- **Quantitative Evaluation Suite**: Pairwise pixel diversity (L1/L2), batch pixel variance (mode collapse diagnostics), manifold precision and recall ($k$-NN manifold approximation), and FID/KID metrics with offline perceptual feature bank fallback.
- **Hardware-Optimized**: Calibrated and profiled for consumer GPUs (tested on NVIDIA GeForce RTX 4050 6GB VRAM, using 3.2 GB peak VRAM at $64 \times 64$).

---

## Experimental Benchmark: Baseline vs StyleGAN AdaIN

Both models trained on real CelebA face images ($64 \times 64$, normalized to $[-1, 1]$) with identical seed ($42$):

| Metric | Experiment 1: Baseline WGAN-GP | Experiment 2: StyleGAN AdaIN | Delta / Finding |
| :--- | :--- | :--- | :--- |
| **Model Type** | Convolutional Transpose WGAN | Style-Based Synthesis + AdaIN | Multi-scale modulation |
| **Latent Mapping** | Direct $Z \in \mathbb{R}^{128}$ | $Z \in \mathbb{R}^{512} \to W \in \mathbb{R}^{512}$ | 8-layer MLP disentanglement |
| **Pairwise L1 Distance** | 0.0657 | **0.7163** | **+990%** diversity variety |
| **Pairwise L2 Distance** | 10.29 | **103.18** | **+902%** separation |
| **Batch Pixel Variance** | 0.0043 | **0.4326** | **+9960%** (100x mode collapse immunity) |
| **Diversity Score** | 0.656 | **7.163** | **10.9x higher variety** |
| **Manifold Recall** | 0.0000 | **1.0000** | Full manifold coverage |
| **VRAM Footprint** | 713 MB | 3,199 MB (~3.2 GB) | Fits safely in 6GB VRAM |
| **Throughput** | 2.5 it/s | 1.15 it/s | ~2.1x computation for AdaIN modulation |

Detailed metrics, logs, and comparative charts can be found in [docs/experiments.md](docs/experiments.md).

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
│   ├── architecture.md              # Detailed mathematical specifications
│   ├── training.md                  # Losses, AMP stability, EMA, and VRAM management
│   ├── experiments.md               # Empirical research and benchmark tables
│   └── evaluation.md                # Quantitative metrics and latent analysis
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
│   ├── fid_kid.py                   # Fréchet Inception Distance & KID evaluator
│   ├── latent_explorer.py           # SLERP, LERP, truncation sweep & style mixing matrix
│   └── diversity.py                 # Mode collapse, pairwise diversity, and precision/recall
├── scripts/
│   ├── train.py                     # Training CLI
│   ├── generate.py                  # Image generation & latent exploration CLI
│   ├── evaluate.py                  # Quantitative metrics CLI
│   ├── verify_dataset.py            # Dataset integrity and zero-leakage validator
│   └── download_celeba.py           # CelebA terms helper & archive unpacker
├── tests/
│   ├── test_models.py               # Unit tests for models & layers
│   ├── test_losses.py               # Unit tests for losses & GP/R1 penalties
│   ├── test_latent_explorer.py      # Unit tests for SLERP, LERP & style mixing
│   └── test_integration.py         # Full single-step training & checkpoint save/load integration tests
├── outputs/                         # Generated samples, SLERP strips, truncation sweeps, style mixes
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

### 2. Verify Test Suite (20/20 Passing)

Run the comprehensive unit and integration test suite:

```bash
python -m pytest tests/
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

Checkpoints will be saved automatically to `experiments/stylegan_adain/checkpoints/` and periodic sample grids to `experiments/stylegan_adain/samples/`.

### Train Baseline WGAN-GP

```bash
python scripts/train.py --config configs/baseline_wgan_gp.yaml
```

---

## Quantitative Evaluation

Evaluate any trained checkpoint for diversity, mode collapse, manifold precision/recall, and FID/KID:

```bash
python scripts/evaluate.py \
    --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_final.pt \
    --num_samples 64 \
    --out_json experiments/stylegan_adain/eval_metrics.json \
    --out_csv experiments/stylegan_adain/eval_metrics.csv
```

---

## Latent Space Exploration & Artifact Generation

StyleForge provides interactive CLI generation for latent research:

```bash
# 1. Generate 16 random sample grid with truncation psi=0.7
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_final.pt --mode samples --n 16 --psi 0.7 --out_dir outputs

# 2. Spherical Linear Interpolation (SLERP) in Z-space
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_final.pt --mode interpolate --steps 10 --out_dir outputs

# 3. Truncation Trick Sweep (psi = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_final.pt --mode truncation --out_dir outputs

# 4. Multi-Scale Style Mixing Matrix
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_final.pt --mode style_mix --out_dir outputs
```

All generated image artifacts are saved to `outputs/`:
- `outputs/samples.png`
- `outputs/interpolation_slerp.png`
- `outputs/truncation_sweep.png`
- `outputs/style_mixing_matrix.png`

---

## Documentation

- [Architecture Specification (docs/architecture.md)](docs/architecture.md): Mathematical formulations for EqualizedLinear, AdaIN, Mapping Network, and Multi-Scale Critic.
- [Training & Stability Guide (docs/training.md)](docs/training.md): Decoupled loss functions, AMP safeguards, EMA shadow weights, and RTX 4050 memory profiling.
- [Empirical Research & Benchmark (docs/experiments.md)](docs/experiments.md): Comparative analysis and ablation experiments.
- [Evaluation Protocol (docs/evaluation.md)](docs/evaluation.md): Details on FID, KID, Pairwise Diversity, and $k$-NN Manifold Precision/Recall.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
CelebA dataset usage is restricted to non-commercial research purposes in accordance with the official [CelebA Dataset Terms](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
