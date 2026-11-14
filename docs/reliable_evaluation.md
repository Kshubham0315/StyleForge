# StyleForge Phase 2: Reliable Quantitative Evaluation & Generative Metrics

**Date**: 2026-11-14  
**Author**: StyleForge AI/ML Research Engineering Team  
**Milestone**: Phase 2 — Scale, Convergence & Generalization  
**Repository**: [https://github.com/Kshubham0315/StyleForge](https://github.com/Kshubham0315/StyleForge)  

---

## 1. Executive Summary

Evaluating generative adversarial networks requires disciplined statistical methodology. In Phase 1, metrics were computed over small subsets ($N_\text{real}=50, N_\text{gen}=64$). Under such regimes, Fréchet Inception Distance (FID) is mathematically uninterpretable due to severe rank deficiency and finite-sample bias.

In Phase 2, StyleForge establishes a rigorous, multidimensional quantitative benchmarking suite:
1. **Matched Inception-v3 Sample Counts**: Official TorchMetrics Inception-v3 evaluation utilizing the full disjoint validation split ($N=614$) against an equal number of generated samples ($N=614$).
2. **Unbiased Kernel Inception Distance (KID)**: Polynomial maximum mean discrepancy (MMD) estimator computed across 20 subsets of size 50.
3. **Inception Feature-Space Precision & Recall**: $k$-nearest neighbor manifold estimation ($k=3$) following Kynkäänniemi et al. (NeurIPS 2019).
4. **Pixel-Space Diversity & Mode Collapse Indicators**: Pairwise $L_1/L_2$ distance distributions and batch pixel variance.
5. **Memorization Diagnostics**: Nearest-neighbor $L_2$ retrieval against real training samples to establish genuine generalization.

---

## 2. Statistical Analysis of Generative Metrics

### 2.1 The Fréchet Inception Distance (FID) Finite-Sample Bias

The Fréchet distance between two Gaussians in Inception-v3 feature space ($\mathbb{R}^d$, where $d=2048$) is defined as:
$$\text{FID}(P_r, P_g) = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

#### The Small-Sample Problem
When the sample size $N < d$ ($N=50 \ll 2048$):
1. **Rank Deficiency**: The sample covariance matrices $\widehat{\Sigma}_r$ and $\widehat{\Sigma}_g$ have rank at most $N-1 \le 49$. The remaining 1,999 eigenvalues are zero.
2. **Eigenvalue Distortions**: As proven by Bińkowski et al. (2018), sample covariance estimation systematically inflates the Fréchet distance by an asymptotic bias term:
   $$\mathbb{E}[\widehat{\text{FID}}_N] = \text{FID}_\infty + \frac{C}{N} + \mathcal{O}\left(\frac{1}{N^2}\right)$$
   For $N=50$, this $\frac{C}{N}$ term dominates the calculation, rendering raw FID scores artificially inflated ($>300-450$) regardless of image quality.
3. **Remediation in Phase 2**: Evaluating on the full validation partition ($N=614$) significantly stabilizes covariance estimation and reduces finite-sample variance.

### 2.2 Unbiased Kernel Inception Distance (KID)

Kernel Inception Distance (Bińkowski et al., 2018) calculates the squared Maximum Mean Discrepancy (MMD) between Inception feature representations using a polynomial kernel:
$$k(x, y) = \left(\frac{1}{d} x^\top y + 1\right)^3$$

The unbiased estimator computes:
$$\text{KID} = \mathbb{E}_{x, x' \sim P_r, y, y' \sim P_g}\left[k(x, x') + k(y, y') - 2k(x, y)\right]$$
- **Key Advantage**: Because KID is an unbiased U-statistic, its expectation does not scale with sample size ($\mathbb{E}[\widehat{\text{KID}}_N] = \text{KID}_\infty$).
- Subsampling 20 blocks of 50 samples provides both an unbiased point estimate and an empirical standard error ($\mu \pm \sigma$).

---

## 3. Manifold Precision & Recall for Generative Models

While FID aggregates quality and diversity into a single scalar, it conflates mode dropping with sample distortion. StyleForge implements the $k$-NN manifold estimation proposed by Kynkäänniemi et al. (2019).

### 3.1 Mathematical Definition

Given real feature set $X_r = \{x_r^{(i)}\}_{i=1}^{N_r}$ and generated feature set $Y_g = \{y_g^{(j)}\}_{j=1}^{N_g}$:
1. For each real feature $x_r^{(i)}$, compute the Euclidean distance to its $k$-th nearest real neighbor:
   $$R_k(x_r^{(i)}) = \text{dist}(x_r^{(i)}, \text{NN}_k(x_r^{(i)}))$$
   The union of hyper-spheres $B(x_r^{(i)}, R_k(x_r^{(i)}))$ forms the estimated real data manifold $\mathcal{M}_r$.
2. Similarly, the generated data manifold $\mathcal{M}_g$ is formed by hyper-spheres around generated samples with radii $R_k(y_g^{(j)})$.

$$\text{Precision} = \frac{1}{N_g} \sum_{j=1}^{N_g} \mathbb{I}\left(y_g^{(j)} \in \mathcal{M}_r\right) = \frac{1}{N_g} \sum_{j=1}^{N_g} \max_{i} \mathbb{I}\left(\|y_g^{(j)} - x_r^{(i)}\|_2 \le R_k(x_r^{(i)})\right)$$

$$\text{Recall} = \frac{1}{N_r} \sum_{i=1}^{N_r} \mathbb{I}\left(x_r^{(i)} \in \mathcal{M}_g\right) = \frac{1}{N_r} \sum_{i=1}^{N_r} \max_{j} \mathbb{I}\left(\|x_r^{(i)} - y_g^{(j)}\|_2 \le R_k(y_g^{(j)})\right)$$

### 3.2 Interpretation for GAN Diagnosis

- **High Precision, Low Recall**: Severe mode collapse. The generator produces realistic samples, but only covers a tiny cluster of the dataset (e.g., generating only blonde women or only center-looking faces).
- **Low Precision, High Recall**: The generator covers the whole space, but produces unrealistic, noisy artifacts outside the manifold.
- **High Precision, High Recall**: Optimal generative modeling with both photorealism and comprehensive distributional diversity.

---

## 4. Diversity & Memorization Metrics

### 4.1 Pixel Diversity Metrics
- **Pairwise Distances**: Evaluated over $M=1,000$ randomly selected sample pairs:
  $$d_{L_1}(x, y) = \frac{1}{C \times H \times W} \sum_{c, h, w} |x_{c, h, w} - y_{c, h, w}|$$
  $$d_{L_2}(x, y) = \sqrt{\frac{1}{C \times H \times W} \sum_{c, h, w} (x_{c, h, w} - y_{c, h, w})^2}$$
- **Batch Pixel Variance**: Evaluates intra-batch spread across all pixels:
  $$\text{Var}_\text{batch} = \frac{1}{C \times H \times W} \sum_{c, h, w} \text{Var}_i\left(x_{i, c, h, w}\right)$$
  When a model collapses to a single mode, $\text{Var}_\text{batch} \to 0$.

### 4.2 Nearest-Neighbor Memorization Test
To verify that the generator does not merely memorize training instances:
- For each generated image $y_g$, find the minimum $L_2$ distance to any image in the training partition:
  $$\text{dist}_\text{train\_NN}(y_g) = \min_{x_r \in \mathcal{D}_\text{train}} \|y_g - x_r\|_2$$
- If the distribution of $\text{dist}_\text{train\_NN}$ is significantly greater than zero and comparable to the real validation set's distance to the training set, the generator is confirmed to perform non-trivial generative synthesis rather than copy-paste replication.
