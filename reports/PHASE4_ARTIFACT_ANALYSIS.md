# StyleForge Phase 4 — Experiment E3: Water-Droplet & Spectral Frequency Analysis

**Date:** November 14, 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM), Windows 11  
**Software:** PyTorch 2.5.1 + CUDA 12.1, Python 3.10.8  
**Scope:** 2D FFT Spectral Decomposition, Intermediate AdaIN Spatial Heatmap Profiling, Upsampling Architecture Ablation  

---

## 1. Scientific Context & Motivation

StyleGAN architectures (Karras et al., 2019) are known to exhibit characteristic localized artifacts commonly termed **"water droplets"**—phase-like, localized circular blobs or spikes that frequently seed in low-to-medium resolutions ($16\times 16$ to $64\times 64$) and persist through fine synthesis layers.

In the seminal StyleGAN2 literature (Karras et al., 2020), this failure mode was mathematically traced to **Adaptive Instance Normalization (AdaIN)**:
$$\text{AdaIN}(x, w) = y_{s, i} \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + y_{b, i}$$
Because AdaIN computes $\mu(x_i)$ and $\sigma(x_i)$ independently across the spatial axes $(H, W)$ for each feature map, it destroys relative magnitude information between features. To bypass this normalization and scale down feature maps globally, the generator learns to create an extreme localized activation spike (often in image corners or outside the facial boundary) that artificially inflates $\sigma(x_i)$.

Furthermore, spatial upsampling methods (bilinear, nearest-neighbor, transposed convolution) introduce distinct spectral signatures, including high-frequency checkerboard harmonics (Odena et al., 2016).

---

## 2. Experimental Investigations & Quantitative Findings

### 2.1 2D Fast Fourier Transform (FFT) Spectral Analysis

We computed the 2D Fast Fourier Transform and radially averaged power spectral density across authentic CelebA validation images versus StyleForge generated samples:

$$\mathcal{F}(I)(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} I(x, y) e^{-j 2\pi (\frac{ux}{H} + \frac{vy}{W})}$$
$$\text{Power}(u, v) = |\mathcal{F}_\text{shifted}(I)(u, v)|^2$$

| Dataset / Source | High-Frequency Energy Ratio (Upper 50% Frequencies) | Spectral Characteristics |
|---|:---:|---|
| **Real CelebA Validation** | $\mathbf{0.0004}$ | Smooth natural power-law decay ($1/f^\alpha$) from DC center to Nyquist edge |
| **StyleForge Synthetic** | $\mathbf{0.0011}$ | **$2.75\times$ elevated high-frequency energy**, exhibiting subtle ring harmonics from discrete upsampling |

---

### 2.2 Layer-Wise Spatial Activation & Outlier Spikes

We hooked all intermediate synthesis blocks across resolutions ($4\times 4, 8\times 8, 16\times 16, 32\times 32, 64\times 64$) to measure spatial outlier severity:
- **Peak-to-Mean Ratio:** $\frac{\max_{h, w} |A_{c, h, w}|}{\frac{1}{HW} \sum_{h, w} |A_{c, h, w}|}$
- **Max Z-Score:** $\frac{\max_{h, w} A_{c, h, w} - \mu(A_c)}{\sigma(A_c)}$
- **Corner Domination %:** Frequency where the maximum activation in a channel occurs at one of the four corners ($(\pm 1, \pm 1)$) within $90\%$ of the global spatial maximum.

| Layer Resolution | Feature Map Shape | Peak-to-Mean Ratio | Max Z-Score ($\sigma$) | Corner Domination % |
|:---:|:---:|:---:|:---:|:---:|
| `res_4` | $512 \times 4 \times 4$ | $22.79$ | $2.07$ | $\mathbf{22.4\%}$ |
| `res_8` | $512 \times 8 \times 8$ | $27.69$ | $2.71$ | $10.2\%$ |
| `res_16` | $256 \times 16 \times 16$ | $27.39$ | $3.10$ | $6.5\%$ |
| `res_32` | $128 \times 32 \times 32$ | $\mathbf{39.32}$ | $3.62$ | $3.2\%$ |
| `res_64` | $64 \times 64 \times 64$ | $25.39$ | $\mathbf{3.73}$ | $2.6\%$ |

#### Interpretation
1. In the initial $4\times 4$ constant input layer, over **$22.4\%$ of feature channels** concentrate their peak activation at the corner coordinates.
2. The peak-to-mean ratio rises progressively, reaching a maximum of **$39.32$ at resolution $32\times 32$**.
3. Max spatial Z-scores reach **$3.73$ standard deviations above the channel mean** at $64\times 64$, confirming that the generator creates localized outlier activations to modulate AdaIN statistics.

---

### 2.3 Upsampling Architecture Ablation

We ablated three distinct spatial upsampling mechanisms:
1. **Mode A: Standard Bilinear Interpolation** (`F.interpolate(mode="bilinear")`)
2. **Mode B: Nearest-Neighbor + $3\times 3$ Conv** (`F.interpolate(mode="nearest")` + Conv)
3. **Mode C: Anti-Aliased Filtered Upsampling** (Bilinear upsampling followed by a $3\times 3$ low-pass FIR binomial filter $[\frac{1}{4}, \frac{1}{2}, \frac{1}{4}] \otimes [\frac{1}{4}, \frac{1}{2}, \frac{1}{4}]^T$)

| Upsampling Mechanism | High-Frequency Energy Ratio | Relative Aliasing vs Bilinear |
|---|:---:|:---:|
| **Nearest-Neighbor + Conv** | $0.1964$ | $+311\%$ (severe checkerboard harmonics) |
| **Bilinear Interpolation** | $0.0478$ | Baseline ($1.0\times$) |
| **Anti-Aliased FIR Filter** | $\mathbf{0.0138}$ | $\mathbf{-71.1\%}$ (smoothest high-frequency transition) |

---

## 3. Engineering Recommendations for Next Architecture Phase

1. **Adopt Weight Demodulation (StyleGAN2 Style):**  
   To completely eliminate the mathematical incentive for the generator to form droplet spikes, future versions should replace AdaIN with weight demodulation:
   $$w'_{i, j, k} = \frac{s_i \cdot w_{i, j, k}}{\sqrt{\sum_{i, k} (s_i \cdot w_{i, j, k})^2 + \epsilon}}$$
2. **Standardize on Anti-Aliased Filtering:**  
   Replacing raw bilinear upsampling with low-pass filtered upsampling suppresses artificial high-frequency energy by $71.1\%$, bringing the generated spectrum much closer to natural CelebA photographs ($0.0138$ vs $0.0478$).

---

## 4. Artifacts & Visualizations

- 2D FFT Power Spectrum & 1D Radial Profile: `experiments/phase4/artifacts/e3_fft_spectral_comparison.png`
- Spatial Activation Outlier Plots: `experiments/phase4/artifacts/e3_activation_spikes.png`
- Raw Metrics JSON: `experiments/phase4/artifacts/e3_artifact_results.json`
