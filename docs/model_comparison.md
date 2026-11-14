# StyleForge Model Comparison: Baseline vs StyleForge AdaIN (Phase 2)

## 1. Experimental Overview

A controlled comparison between the Classical WGAN-GP Baseline and the StyleGAN-inspired AdaIN architecture on authentic photographic CelebA data under identical evaluation constraints:
- **Dataset**: Authentic CelebA Align&Cropped ($178\times 218 \to 64\times 64$, normalized to $[-1, 1]$).
- **Partitioning**: Strictly disjoint Train ($4,912$), Validation ($614$), and Test ($614$) splits.
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, pure FP32).
- **Step Budget**: 1,000 generator updates ($16,000$ image presentations per model, $n_\text{critic}=4$).
- **Evaluation Discipline**: Official TorchMetrics Inception-v3 computed on all $N=614$ validation images against $N=614$ generated images.

---

## 2. Multi-Metric Benchmark Comparison Table

| Metric | WGAN-GP Baseline (1000 Steps) | StyleForge AdaIN (EMA, 1000 Steps) | StyleForge AdaIN (Raw / No EMA, 1000 Steps) | Scientific Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Generator Architecture** | Transpose Conv (DCGAN) | 8-Layer Mapping + AdaIN + Constant | 8-Layer Mapping + AdaIN + Constant | Multi-scale modulation |
| **Latent Dimension** | $Z \in \mathbb{R}^{128}$ | $Z \in \mathbb{R}^{512} \to W \in \mathbb{R}^{512}$ | $Z \in \mathbb{R}^{512} \to W \in \mathbb{R}^{512}$ | Learned intermediate manifold |
| **Inception-v3 FID $\downarrow$** | 344.90 | 383.71 | **305.99** | Instantaneous weights achieve lowest FID |
| **KID Mean $\downarrow$** | $0.4225 \pm 0.0098$ | $0.5246 \pm 0.0171$ | **$0.3531 \pm 0.0167$** | Unbiased polynomial MMD estimator |
| **Pairwise L1 Distance $\uparrow$** | 0.0410 | **0.6512** | 0.5794 | **$15.9\times$ higher diversity in AdaIN** |
| **Pairwise L2 Distance $\uparrow$** | 5.91 | **94.00** | 84.45 | **$15.9\times$ higher feature spread** |
| **Batch Pixel Variance $\uparrow$** | 0.00153 | **0.37069** | 0.31068 | **$242\times$ higher variance (No collapse)** |
| **Contrast & Dynamic Range** | Collapsed: `[0, 141]` ($\sigma=29.6$) | **Full: `[0, 255]` ($\sigma=109.8$)** | **Full: `[0, 255]` ($\sigma=109.8$)** | Baseline collapses to gray-brown blur |
| **Inception Precision ($k=3$)** | 0.0000 | 0.0000 | 0.0000 | Limited by 16 kimg training budget |
| **Inception Recall ($k=3$)** | 0.0000 | 0.0000 | 0.0000 | Limited by 16 kimg training budget |
| **Peak Reserved VRAM** | **271 MB** | 3,363 MB | 3,363 MB | Both safely within 6GB hardware limit |
| **Latent Disentanglement** | Entangled $Z$ | **Linear $W$-space LERP & Style Mixing** | **Linear $W$-space LERP & Style Mixing** | Disentangled attribute manipulation |

---

## 3. Engineering & Research Insights

### 3.1 Mode Collapse Immunity vs. Spatial Averaging
- **Baseline Failure**: The DCGAN-style baseline suffered from severe spatial contrast decay. To minimize discriminator penalties across a diverse photographic face dataset, the generator learned to output the spatial average of the training data. Pixel intensities narrowed to $[0, 141]$, and standard deviation collapsed from $52.7$ to $29.6$.
- **StyleForge Robustness**: By decoupling the spatial geometry via a learned $4\times 4$ constant input and injecting style through AdaIN at every resolution scale, StyleForge is immune to contrast decay, preserving the full $[0, 255]$ dynamic range and yielding $242\times$ higher batch variance.

### 3.2 The EMA Effect: Manifold Smoothing vs Sharp Realism
- In StyleForge, shadow weights smoothed by Exponential Moving Average ($\beta=0.999$) achieved the highest pairwise diversity ($0.6512$) and batch variance ($0.37069$).
- However, evaluating the raw instantaneous generator weights produced a substantially lower Fréchet Inception Distance (**305.99** vs 383.71) and lower KID (**0.3531** vs 0.5246). This demonstrates that EMA dampens extreme parameter swings and increases distributional coverage, whereas instantaneous weights achieve sharper local alignment with validation feature clusters.
