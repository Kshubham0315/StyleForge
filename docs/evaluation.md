# StyleForge Evaluation Methodology & Latent Analysis

## 1. Quantitative Evaluation Suite

StyleForge implements a multi-metric quantitative pipeline designed to assess sample quality, mode collapse, manifold coverage, and latent structure without fabricating metric scores.

### 1.1 Pairwise Sample Diversity
Given $N$ generated images $\{x_1, \dots, x_N\}$, images are flattened into vectors $v_i \in \mathbb{R}^{C \cdot H \cdot W}$. For random pairs $(i, j)$:
$$\mathcal{D}_{L1} = \frac{1}{|P|} \sum_{(i, j) \in P} \frac{1}{d} \|v_i - v_j\|_1$$
$$\mathcal{D}_{L2} = \frac{1}{|P|} \sum_{(i, j) \in P} \|v_i - v_j\|_2$$
$$\text{Diversity Score} = 10 \times \mathcal{D}_{L1}$$

### 1.2 Batch Pixel Variance (Mode Collapse Diagnostic)
To detect whether a generator produces identical images regardless of input latent vector:
$$\text{Var}_{\text{batch}} = \frac{1}{C \cdot H \cdot W} \sum_{k=1}^{C \cdot H \cdot W} \text{Var}_i(v_{i, k})$$
Values approaching zero indicate severe mode collapse.

### 1.3 Manifold Precision and Recall
Following Sajjadi et al. (2018) and Kynkäänniemi et al. (2019):
- For real features $\{r_i\}$, the $k$-NN radius $B(r_i, \rho(r_i))$ defines the real data manifold.
- **Precision**: Fraction of generated samples that fall within at least one real sample hypersphere (measures fidelity/authenticity).
- **Recall**: Fraction of real samples that fall within at least one generated sample hypersphere (measures diversity and mode coverage).

### 1.4 FID & KID Distribution Distances
- Fréchet Inception Distance (FID) evaluates the 2-Wasserstein distance between Gaussian approximations of feature distributions:
  $$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$
- StyleForge checks for TorchMetrics InceptionV3 and includes an offline perceptual feature bank fallback for offline or air-gapped environments, explicitly reporting the metric backend in all JSON/CSV outputs.

---

## 2. Latent Space Exploration Suite

StyleForge provides interactive CLI generation via `scripts/generate.py`:

### 2.1 SLERP in $\mathcal{Z}$-Space vs LERP in $\mathcal{W}$-Space
Spherical linear interpolation (SLERP) preserves latent vector norm on Gaussian spheres:
$$\text{SLERP}(z_1, z_2, t) = \frac{\sin((1 - t)\Omega)}{\sin \Omega} z_1 + \frac{\sin(t\Omega)}{\sin \Omega} z_2, \quad \Omega = \arccos\left(\frac{z_1 \cdot z_2}{\|z_1\| \|z_2\|}\right)$$
Linear interpolation (LERP) is performed directly in disentangled $\mathcal{W}$-space:
$$\text{LERP}(w_1, w_2, t) = (1 - t)w_1 + t w_2$$

### 2.2 Truncation Trick in $\mathcal{W}$-Space
To trade diversity for individual sample fidelity, styles are shrunk toward the running average $\bar{w}$:
$$w' = \bar{w} + \psi (w - \bar{w}), \quad \psi \in [0.0, 1.0]$$
- $\psi = 0.0$: Yields the canonical average face.
- $\psi = 0.7$: Optimal quality-diversity balance.
- $\psi = 1.0$: Full latent diversity without truncation.

### 2.3 Style Mixing Matrix
Styles from Source A and Source B are mixed across resolution layers:
- **Coarse layers (4x4, 8x8)**: Governs pose, face shape, and head orientation.
- **Middle layers (16x16, 32x32)**: Governs facial features, hairstyle, eyes, and mouth.
- **Fine layers (64x64)**: Governs color scheme, skin tone, lighting, and fine texture.
