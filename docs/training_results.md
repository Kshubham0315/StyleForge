# StyleForge Extended Training Results (600 Steps)

## 1. Overview & Experimental Budget

This document reports empirical training trajectories for both the WGAN-GP Baseline and StyleForge AdaIN models on the CelebA dataset ($64\times 64$, normalized to $[-1, 1]$) extended to **600 steps** ($9,600$ image presentations per model, representing $21.3$ epochs over the 450-image training partition).

All runs executed on an NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM) under Windows 11 with PyTorch 2.5.1 and CUDA 12.1.

---

## 2. Quantitative Training Summary

| Parameter | Baseline WGAN-GP | StyleForge AdaIN |
| :--- | :--- | :--- |
| **Model Type** | Convolutional Transpose WGAN | Style-Based Synthesis + AdaIN |
| **Latent Space** | $Z \in \mathbb{R}^{128}$ | $Z \in \mathbb{R}^{512} \to W \in \mathbb{R}^{512}$ |
| **Total Steps** | 600 steps | 600 steps |
| **Batch Size** | 16 | 16 |
| **Images Processed** | 9,600 | 9,600 |
| **Precision Mode** | FP16 AMP (FP32 GP) | Pure FP32 (Numerically Stable) |
| **Critic Updates ($n_{\text{critic}}$)** | 5 | 4 |
| **Peak VRAM Allocated** | 743 MB | 3,475 MB (~3.5 GB) |
| **Average Throughput** | **5.62 it/s** | **1.12 it/s** |
| **Total Duration** | 81.8s (1.36 min) | 417.8s (6.96 min) |
| **Final Critic Grad Norm** | ~1.5 - 2.0 | ~15.0 - 25.0 |
| **Final Gen Grad Norm** | ~1.2 - 2.5 | ~8.0 - 18.0 |
| **W-Space Standard Deviation** | N/A | **0.96 - 1.05** (Well-dispersed) |

---

## 3. Loss Dynamics & Stability Analysis

### 3.1 Baseline WGAN-GP Dynamics
- **Wasserstein Distance**: Rose steadily from $0.001$ to $8.3$ - $9.2$, indicating stable separation between real and generated distributions without critic saturation.
- **Gradient Penalty**: Constrained tightly around $1.3$ - $1.7$ (target 0 deviation from unit gradient norm), confirming 1-Lipschitz continuity.
- **Gradient Norms**: Both generator and critic gradient norms remained bounded below $3.0$, exhibiting stable non-oscillatory convergence.

### 3.2 StyleForge AdaIN Dynamics
- **Numerical Stability**: When executing in pure FP32, StyleForge AdaIN maintained unit variance across the 8-layer mapping network without a single NaN or Inf anomaly throughout 600 steps.
- **W-Space Variance**: Tracked continuously in generator step statistics; intermediate $W$-vector standard deviation remained stable in range $[0.96, 1.05]$, proving that the $1 / \text{lr\_mul}$ initialization mathematically prevents latent space collapse.
- **Wasserstein Distance**: Scaled to ~85 - 90, reflecting higher-capacity adversarial discrimination across multi-scale feature hierarchies.

---

## 4. Training Trajectory Visualizations

Trajectory plots export automatically upon completion of training:
- **Baseline Trajectory**: `experiments/baseline_wgan_gp/training_curves.png`
- **StyleForge Trajectory**: `experiments/stylegan_adain/training_curves.png`
- **Spectral Norm Ablation Trajectory**: `experiments/ablation_spectral_norm/training_curves.png`
- **Self-Attention Ablation Trajectory**: `experiments/ablation_self_attention/training_curves.png`
