# StyleForge Phase 2: Systematic Ablation Studies

**Date**: 2026-11-14  
**Author**: StyleForge AI/ML Research Engineering Team  
**Milestone**: Phase 2 — Scale, Convergence & Generalization  
**Repository**: [https://github.com/Kshubham0315/StyleForge](https://github.com/Kshubham0315/StyleForge)  

---

## 1. Executive Summary

To isolate the causal factors underlying StyleForge's training stability and generative capacity, Phase 2 conducts three strictly controlled ablation studies on authentic CelebA data:
1. **Ablation A: Self-Attention (Non-Local Modeling)** — Testing whether self-attention at intermediate feature resolutions ($16\times 16$) improves global geometric facial coherence.
2. **Ablation B: Spectral Normalization in the Critic** — Testing whether spectral normalization ($L_2$ operator norm bound $\sigma(W) \le 1$) stabilizes critic Lipschitz continuity alongside gradient penalty.
3. **Ablation C: Exponential Moving Average (EMA)** — Directly isolating the effect of Polyak-Ruppert parameter averaging ($\beta = 0.999$) on generated image sharpness and metric variance.

All ablation experiments share identical hyperparameters, optimizer settings ($\text{Adam}, \beta_1=0.0, \beta_2=0.99$), learning rates ($\eta_g=10^{-4}, \eta_d=10^{-4}$), batch size ($16$), and data partitions.

---

## 2. Theoretical Motivation for Each Ablation

### 2.1 Ablation A: Self-Attention at $16\times 16$
- **Hypothesis**: Standard convolutional layers have localized receptive fields that grow linearly with network depth. In human face synthesis, distant anatomical structures (e.g., matching eye colors, symmetric cheekbones, aligned jawlines) require non-local feature coordination.
- **Mechanism**: Implements Self-Attention 2D (Zhang et al., 2019) at spatial resolution $16\times 16$:
  $$\beta_{j, i} = \frac{\exp(s_{i j})}{\sum_{i=1}^N \exp(s_{i j})}, \quad \text{where } s_{i j} = f(x_i)^\top g(x_j)$$
  $$o_j = v\left(\sum_{i=1}^N \beta_{j, i} h(x_i)\right)$$
  $$y_i = \gamma o_i + x_i$$
  where $\gamma$ is a learned scalar parameter initialized to $0.0$, allowing the network to gradually incorporate non-local context without disrupting early training.

### 2.2 Ablation B: Spectral Normalization in Critic
- **Hypothesis**: WGAN-GP enforces 1-Lipschitz continuity via a soft penalty on the gradient norm along random linear interpolations:
  $$\mathcal{L}_\text{GP} = \lambda \mathbb{E}_{\hat{x}}\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right]$$
  However, this penalty is only enforced almost-everywhere along the interpolation lines. Spectral Normalization (Miyato et al., 2018) provides an analytical, global Lipschitz bound across the entire input space by dividing each layer's weight matrix by its maximum singular value $\sigma(W)$ estimated via power iteration:
  $$\bar{W}_\text{SN} = \frac{W}{\sigma(W)}$$
- **Research Question**: Does combining Spectral Normalization with Equalized Learning Rate provide superior critic gradient conditioning, or does it over-constrain the discriminator's capacity and slow convergence?

### 2.3 Ablation C: Exponential Moving Average (EMA)
- **Hypothesis**: GAN training trajectories do not converge to a static point in parameter space, but rather orbit a limit cycle or suffer from high-frequency parameter oscillations due to non-zero-sum game dynamics (Mescheder et al., 2018).
- **Mechanism**: Maintains shadow weights $\theta_\text{EMA}$ updated as:
  $$\theta_\text{EMA}^{(t)} = \beta \theta_\text{EMA}^{(t-1)} + (1 - \beta) \theta^{(t)}, \quad \beta = 0.999$$
- **Research Question**: By comparing the shadow weights $\theta_\text{EMA}$ against the raw generator weights $\theta$ from the exact same training run, what is the precise numerical benefit of EMA on Fréchet Inception Distance, Precision, and Recall?

---

## 3. Experimental Configurations & Checkpoint Paths

| Experiment | Config Path | Active Component | Total Steps | Checkpoint Path |
| :--- | :--- | :--- | :---: | :--- |
| **Full StyleForge** | `configs/phase2_stylegan_adain.yaml` | AdaIN + Noise + EqualizedLR + EMA | 1,000 | `experiments/phase2_stylegan_adain/checkpoints/` |
| **Ablation A** | `configs/phase2_ablation_self_attention.yaml` | + Self-Attention at $16\times 16$ | 400 | `experiments/phase2_ablation_self_attention/checkpoints/` |
| **Ablation B** | `configs/phase2_ablation_spectral_norm.yaml` | + Spectral Normalization in Critic | 400 | `experiments/phase2_ablation_spectral_norm/checkpoints/` |
| **Ablation C** | `configs/phase2_ablation_no_ema.yaml` | Raw Generator ($\beta=0.0$) vs EMA | 1,000 | Direct evaluation of raw vs EMA weights |
