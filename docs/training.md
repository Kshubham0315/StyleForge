# StyleForge Training Dynamics & Optimization Guide

## 1. Objective Functions & Regularization

StyleForge supports decoupled adversarial objectives configurable via YAML.

### 1.1 Wasserstein Loss with Two-Sided Gradient Penalty (WGAN-GP)
Under the Kantorovich-Rubinstein duality, the Wasserstein-1 distance is:
$$W(p_r, p_g) = \sup_{\|D\|_L \le 1} \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})]$$

The critic loss with two-sided gradient penalty (Gulrajani et al., 2017) is:
$$\mathcal{L}_D = \mathbb{E}_{\tilde{x}}[D(\tilde{x})] - \mathbb{E}_{x}[D(x)] + \lambda_{\text{GP}} \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$
where $\hat{x} = \epsilon x + (1 - \epsilon)\tilde{x}$ with $\epsilon \sim U[0, 1]$ represents points sampled along straight lines between real and generated distributions.

The generator loss minimizes:
$$\mathcal{L}_G = -\mathbb{E}_{\tilde{x}}[D(\tilde{x})]$$

### 1.2 R1 Zero-Centered Gradient Penalty
Alternatively, StyleForge implements the Mescheder et al. (2018) R1 penalty applied strictly to real data points:
$$R_1(D) = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{\text{data}}}\left[\|\nabla_x D(x)\|^2_2\right]$$
Combined with non-saturating logistic loss:
$$\mathcal{L}_D^{\text{logistic}} = \mathbb{E}_{x}[\text{softplus}(-D(x))] + \mathbb{E}_{\tilde{x}}[\text{softplus}(D(\tilde{x}))] + R_1(D)$$

---

## 2. Mixed Precision (AMP) & Numerical Stability Safeguards

GAN training with gradient penalties involves second-order derivatives $\nabla_{\hat{x}} D(\hat{x})$ with `create_graph=True`. In native FP16 automatic mixed precision (AMP), computing gradients of gradients frequently encounters numerical underflow or overflow, triggering NaN loss scales and dynamic scaler skips.

### Safeguard Protocol in StyleForge:
```python
# Critic forward pass runs in AMP for maximum VRAM throughput
with torch.amp.autocast("cuda", enabled=self.amp):
    d_real = self.critic(real_images)
    d_fake = self.critic(fake_images.detach())
    w_loss = d_fake.mean() - d_real.mean()

# Gradient Penalty calculation: FORCE FP32 and DISABLE autocast
with torch.amp.autocast("cuda", enabled=False):
    penalty = compute_gradient_penalty(
        self.critic,
        real_images.float(),
        fake_images.detach().float(),
        device=self.device
    )
```

Furthermore, explicit NaN/Inf gradient auditing checks parameter gradients after backward passes, falling back gracefully if an optimizer step is corrupted.

---

## 3. Exponential Moving Average (EMA)

Adversarial training suffers from cyclical parameter oscillations around saddle points. StyleForge maintains an EMA shadow copy of the generator parameters $\theta_{\text{EMA}}$ updated after each generator step:
$$\theta_{\text{EMA}} \leftarrow \beta \theta_{\text{EMA}} + (1 - \beta) \theta, \quad \beta = 0.999$$

Benefits:
- Evaluated and saved independently in checkpoints (`ema_generator`).
- Dampens high-frequency mode transitions.
- Yields cleaner, artifact-free visual synthesis.

---

## 4. Hardware Optimization for RTX 4050 (6GB VRAM)

Training runs on NVIDIA GeForce RTX 4050 Laptop GPU under Windows 11 with the following budget:

| Component | Baseline WGAN-GP | StyleGAN AdaIN |
| :--- | :--- | :--- |
| **Image Resolution** | $64 \times 64$ | $64 \times 64$ |
| **Batch Size** | 16 | 16 |
| **VRAM Footprint** | 713 MB | 3,199 MB (~3.2 GB) |
| **Throughput** | 2.1 – 2.5 it/s | 1.15 – 1.45 it/s |
| **Critic Steps per G ($n_{\text{critic}}$)** | 5 | 4 |
| **Mapping Network** | N/A (Linear projection) | 8 layers ($512 \to 512$) |
| **Peak VRAM Cap** | ~12% of 6GB | ~53% of 6GB |

Safety margin: Over 2.7 GB of VRAM remains unallocated, preventing CUDA out-of-memory (OOM) triggers during autograd backward graph expansions.
