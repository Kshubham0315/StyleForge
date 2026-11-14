# StyleForge Phase 3: Comprehensive EMA Dynamics & Decay Sweep Analysis

**Author**: StyleForge AI/ML Research Engineering Team  
**Date**: November 14, 2026  
**Repository**: [https://github.com/Kshubham0315/StyleForge](https://github.com/Kshubham0315/StyleForge)  
**Milestone**: Phase 3 — Generative Quality Audit & Optimization  
**Artifacts**:
- `experiments/phase3/ema_sweep/ema_decay_visual_comparison.png`
- `experiments/phase3/ema_sweep/ema_sweep_results.json`

---

## 1. Executive Summary

A pervasive assumption in modern generative adversarial networks is that Exponential Moving Average (Polyak-Ruppert parameter averaging) uniformly improves sample quality and lowers Fréchet Inception Distance (FID).

In Phase 2, an apparent paradox emerged:
- **Raw Instantaneous Weights**: Inception FID $= \mathbf{305.99}$, KID $= \mathbf{0.3531}$
- **EMA Shadow Weights ($\beta=0.999$)**: Inception FID $= 383.71$, KID $= 0.5246$
- **EMA Pairwise Diversity**: $\mathbf{0.6512}$ ($12.4\%$ higher than Raw $0.5794$)
- **EMA Batch Pixel Variance**: $\mathbf{0.37069}$ ($19.3\%$ higher than Raw $0.31068$)

To rigorously resolve this paradox, Phase 3 conducted:
1. **Mathematical Initialization & Bias Audit**: Analyzing parameter lag and initial weight retention.
2. **Layer-wise Parameter Divergence Audit**: Inspecting cosine similarity across all 14.5M generator parameters.
3. **Controlled Multi-EMA Sweep**: Simultaneously updating three EMA shadows ($\beta = 0.990, 0.995, 0.999$) alongside the raw generator on identical minibatches and optimizer steps.

---

## 2. Mathematical Bias & Initialization Inertia

In `training/trainer.py`, the shadow parameters $\theta_\text{EMA}$ evolve according to:
$$\theta_\text{EMA}^{(t)} = \beta \theta_\text{EMA}^{(t-1)} + (1 - \beta) \theta^{(t)}$$
Expanding this recurrence from initialization $\theta^{(0)}$ yields:
$$\theta_\text{EMA}^{(t)} = \beta^t \theta^{(0)} + (1 - \beta) \sum_{i=1}^t \beta^{t-i} \theta^{(i)}$$

### Initial Weight Residual Analysis:
At step $t$, the fraction of the initial untrained random weights $\theta^{(0)}$ still contaminating $\theta_\text{EMA}$ is given by $\beta^t$:

| EMA Decay $\beta$ | Effective Window $\frac{1}{1 - \beta}$ | Residual at Step 300 ($\beta^{300}$) | Residual at Step 1,000 ($\beta^{1000}$) | Residual at Step 5,000 ($\beta^{5000}$) |
| :---: | :---: | :---: | :---: | :---: |
| **0.990** | 100 steps | **$4.90\%$** | **$0.00\%$** | $0.00\%$ |
| **0.995** | 200 steps | **$22.23\%$** | **$0.67\%$** | $0.00\%$ |
| **0.999** | 1,000 steps | **$74.06\%$** | **$36.77\%$** | **$0.67\%$** |

### Mathematical Deduction:
- At 1,000 training steps, an un-debiased EMA with $\beta = 0.999$ still retains **$36.77\%$ of its initial untrained parameter state**!
- At 300 steps, it retains **$74.06\%$ of initial random weights**!
- Because the raw generator has undergone 1,000 pure gradient descent updates, its parameters are unencumbered by initial initialization bias, allowing it to fit the high-frequency features of CelebA faces much more sharply.

---

## 3. Parameter-Level & Output Divergence Audit

Inspecting the 1,000-step checkpoint (`experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt`):
- **Total Generator Parameters**: $14,536,387$
- **Total Parameter $L_2$ Difference**: $\|\theta_\text{raw} - \theta_\text{ema}\|_2 = 19.9714$
- **Average Layer-wise Cosine Similarity**: $0.9739$

### Most Diverged vs Most Aligned Layers:
- **Highest Divergence (Stochastic Noise Injection)**:
  - `synthesis.blocks.res_16_b1.noise.weight`: Cos Sim $= \mathbf{0.8967}$, $L_2$ Diff $= 0.0290$
  - `synthesis.blocks.res_8_b1.noise.weight`: Cos Sim $= \mathbf{0.8976}$, $L_2$ Diff $= 0.0417$
  - *Finding*: Noise scales undergo high-frequency gradient adaptations that EMA aggressively smooths out.
- **Highest Alignment (Style Mapping Network)**:
  - `mapping.net.4.weight`: Cos Sim $= \mathbf{1.0000}$
  - `mapping.net.8.weight`: Cos Sim $= \mathbf{1.0000}$
  - *Finding*: The 8-layer MLP mapping manifold evolves smoothly along a stable direction.

### Output Difference on Fixed Latent Codes ($N=64$):
- **Mean Pixel $L_1$ Difference**: $0.3309$ (on $[-1, 1]$ scale)
- **Mean Pixel $L_2$ Difference**: $53.0410$
- **Maximum Pixel Difference**: $1.9972$ (near the theoretical maximum of $2.0$)
- *Conclusion*: Raw and EMA weights produce structurally distinct facial syntheses for identical latent codes $z$.

---

## 4. Controlled Multi-EMA Sweep Results (300 Steps)

To isolate the causal effect of $\beta$, a controlled experiment was run where **one single training run** concurrently updated the raw generator and three EMA shadows on the exact same minibatches:

| Configuration | Initial Residual | Inception FID $\downarrow$ | KID Mean $\downarrow$ ($\pm \text{std}$) | Pairwise L1 Diversity $\uparrow$ | Batch Pixel Variance $\uparrow$ | Core Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Instantaneous** | 0.00% | **381.61** | **$0.5178 \pm 0.0161$** | 0.6354 | 0.36889 | **Lowest FID & KID** |
| **EMA decay = 0.990** | 4.90% | 393.73 | $0.5468 \pm 0.0201$ | 0.6605 | 0.39300 | Fast tracking, low lag |
| **EMA decay = 0.995** | 22.23% | 409.50 | $0.5885 \pm 0.0219$ | 0.6864 | 0.41215 | Intermediate smoothing |
| **EMA decay = 0.999** | 74.06% | 419.60 | $0.6186 \pm 0.0185$ | **0.7159** | **0.43518** | **Highest diversity & variance** |

### Perfect Monotonic Empirical Progression:
Across the controlled sweep:
1. **FID strictly increases monotonically with decay**: $381.61 \to 393.73 \to 409.50 \to 419.60$.
2. **KID strictly increases monotonically with decay**: $0.5178 \to 0.5468 \to 0.5885 \to 0.6186$.
3. **Pairwise diversity strictly increases monotonically with decay**: $0.6354 \to 0.6605 \to 0.6864 \to 0.7159$.
4. **Batch variance strictly increases monotonically with decay**: $0.36889 \to 0.39300 \to 0.41215 \to 0.43518$.

---

## 5. Scientific Conclusion: Does EMA Improve Quality or Change Variance?

**Scientific Verdict:**
1. **EMA does NOT unconditionally improve visual sharpness or lower FID in early/medium GAN training.**
2. Instead, Polyak averaging acts as a **distributional low-pass filter**: it averages out high-frequency parameter oscillations, preventing mode collapse and broadening sample variance, but simultaneously introducing parameter inertia that delays alignment with the target feature distribution.
3. For training budgets under 5,000 steps, standard $\beta = 0.999$ carries significant initialization drag ($36.8\%$ at 1,000 steps). If EMA is used in short training regimes, either **$\beta \le 0.990$** or an explicit warmup schedule $\beta(t) = \min(\beta, 1 - \frac{1}{t})$ should be applied.
4. For maximum sample sharpness and lowest FID at 1,000 steps, the instantaneous **Raw generator weights** outperform EMA.
