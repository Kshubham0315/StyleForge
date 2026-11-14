# StyleForge Model Card
## Style-Based Face Generation System (v1.0 Production)

**Model Release:** StyleForge v1.0  
**Model Date:** November 2026  
**License:** MIT License  
**Model Type:** Generative Adversarial Network (StyleGAN-Inspired)  
**Developers:** StyleForge Research Team  

---

## 1. Model Overview

StyleForge is a style-based generative adversarial network engineered for human face synthesis, latent manifold exploration, and generative stability benchmarking. It maps 512-dimensional Gaussian latent vectors $z \in \mathcal{Z}$ through an 8-layer MLP Mapping Network into an intermediate disentangled space $w \in \mathcal{W}$, progressively modulating spatial convolutions via Adaptive Instance Normalization (AdaIN) from a learned $4\times 4\times 512$ constant tensor up to $64\times 64$ RGB images.

---

## 2. Intended Use

### Primary Use Cases
- Academic and educational exploration of GAN architectures and adversarial dynamics.
- Controlled research on latent space disentanglement, SLERP/LERP interpolation, truncation tricks, and style mixing.
- Benchmarking generative metrics (FID, KID, pairwise diversity, batch variance) under consumer hardware constraints.
- Educational demonstrations in generative AI engineering portfolios and technical presentations.

### Out-of-Scope & Non-Intended Uses
- Biometric facial recognition or identification.
- Generating deceptive deepfakes or non-consensual synthetic representations of real individuals.
- Commercial automated surveillance or law enforcement systems.
- High-stakes automated decision making.

---

## 3. Model Architecture & Specifications

| Parameter | Specification |
|---|---|
| **Input Latent Space ($z$)** | 512-dimensional vector, $z \sim \mathcal{N}(0, I)$ |
| **Intermediate Style Space ($w$)** | 512-dimensional vector ($f: \mathcal{Z} \to \mathcal{W}$) |
| **Output Spatial Resolution** | $64\times 64\times 3$ (RGB) |
| **Mapping Network** | 8-layer MLP with PixelNorm, EqualizedLinear (gain $\sqrt{2}$), LeakyReLU ($\alpha=0.2$) |
| **Synthesis Network** | 5 resolution stages ($4\times 4 \to 64\times 64$), AdaIN modulation, learned constant input |
| **Critic / Discriminator** | Multi-scale residual critic with Minibatch Standard Deviation |
| **Generator Parameters** | $14,535,875$ |
| **Critic Parameters** | $12,174,721$ |
| **Exported Checkpoint** | `checkpoints/styleforge_final_model.pt` ($110.95\text{ MB}$, stripped optimizers) |

---

## 4. Quantitative Performance Metrics

All metrics evaluated against authentic photographic CelebA validation images ($N=614, 64\times 64$):

| Configuration | Global Steps | Inception-v3 FID ↓ | KID $\times 10^3$ ↓ | Pairwise $L_1$ Div ↑ | Batch Pixel Var ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| **StyleForge Raw (Production)** | **1,500** | **297.76** | **0.3285** | $0.5779$ | $0.28987$ |
| **StyleForge EMA (Stabilized)** | **1,500** | $336.98$ | $0.4167$ | $0.6223$ | $0.33502$ |
| **Convolutional WGAN-GP Baseline** | 1,000 | $344.90$ | $0.4225$ | $0.0410$ (Collapsed) | $0.00153$ (Collapsed) |

---

## 5. Generalization & Memorization Safeguards

- **Data Contamination:** Cryptographic MD5 hash audit confirmed $0$ overlapping images across Train, Validation, and Test partitions.
- **Memorization Probing:** Generated samples exhibit equal distance to training and unseen test samples:
  - Pixel $L_2$ Ratio ($\frac{d_\text{train}}{d_\text{test}}$): **$0.9806$** (Parity within $1.9\%$)
  - Inception pool3 Ratio ($\frac{d_\text{train}}{d_\text{test}}$): **$0.9996$** (Parity within $0.04\%$)
- Under tested nearest-neighbor protocols, no evidence of verbatim training exemplar memorization was detected.

---

## 6. Ethical Considerations & Demographic Biases

- **Training Distribution Bias:** CelebA consists primarily of photographs of public figures, actors, and celebrities taken under professional lighting conditions. Consequently, the model reflects demographic skews inherent to CelebA (predominantly young adults, Eurocentric facial traits, stylized hairstyles, and makeup).
- **Underrepresentation:** Representation of diverse age groups (elderly, young children) and varied geographic populations is constrained by the training distribution.
- **Artifacts:** At $\psi > 1.0$, truncation sweeps can produce localized high-frequency contrast saturations.

---

## 7. How to Load and Run

```python
import torch
from inference.infer import StyleForgeInference

# Initialize engine (loads checkpoints/styleforge_final_model.pt by default)
engine = StyleForgeInference()

# Generate 4 diverse faces
images = engine.generate(seed=42, num_samples=4, truncation_psi=0.8)
images[0].save("face_0.png")
```
