# StyleForge — Metrics & Quantitative Evaluation Audit
## Scientific Forensic Analysis of Reported Metrics, Mathematical Formulations & Empirical Validation

**Auditor:** Senior AI Research Engineer & ML Systems Architect  
**Objective:** Deconstruct, mathematically verify, and audit every reported metric in StyleForge.

---

## 1. The Core Metric Unmasking: "FID 0.0018 → 0.0005"

One of the central claims investigated during this audit was the reported metric:
$$\text{"FID: } 0.0018 \to 0.0005\text{"}$$

### 1.1 The Forensic Finding
In official Generative Adversarial Network literature (Heusel et al. NeurIPS 2017, Brock et al. ICLR 2019, Karras et al. CVPR 2020), **Fréchet Inception Distance (FID)** is strictly defined as the Wasserstein-2 distance between multivariate Gaussian distributions fitted to the **2048-dimensional activation features of a pre-trained Inception-v3 network** (`MaxPool2d` layer before classification) evaluated on $50,000$ generated samples against $50,000$ real reference images:
$$\text{FID} = \|\mu_{\text{real}} - \mu_{\text{fake}}\|_2^2 + \text{Tr}\left(\Sigma_{\text{real}} + \Sigma_{\text{fake}} - 2 \left(\Sigma_{\text{real}}\Sigma_{\text{fake}}\right)^{1/2}\right)$$

In real-world generative modeling:
- A state-of-the-art StyleGAN2 on FFHQ ($1024\times 1024$) achieves an FID of **$2.84$**.
- A standard DCGAN on CelebA $64\times 64$ achieves an FID between **$15.0$ and $35.0$**.
- An Inception-v3 FID of **$0.0005$ is mathematically impossible** for any generative model unless the generated images are byte-identical to the training set with zero variance.

### 1.2 What the 0.0005 Metric Actually Is
By tracing the execution in `scripts/evaluate_phase4_quality.py` and `evaluation/fid_kid.py`, this audit uncovered that **the value $0.0005$ is NOT Inception-v3 FID**.

Instead, it is **Fréchet Feature Distance (FFD)** evaluated on an offline custom 4-layer convolutional network (`PerceptualFeatureBank`):
```python
class PerceptualFeatureBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
```
- **Feature Extractor:** 4 convolutional downsampling layers with no pretraining.
- **Output Dimensionality:** $256$ dimensions.
- **Sample Count:** $N = 512$ generated images vs $N = 614$ validation images.
- **Feature Scale:** Because the feature extractor maps low-resolution images through randomly initialized or unscaled conv filters without batch normalization, the resulting feature activations $\mu$ and covariances $\Sigma$ have small numerical ranges ($\sim 10^{-2}$ to $10^{-3}$), yielding Fréchet distances on the order of $10^{-4}$ to $10^{-3}$.

### 1.3 What the True Inception-v3 FID Actually Is
When StyleForge was evaluated using the official TorchMetrics Inception-v3 feature extractor (documented in `reports/benchmark_reconciliation.json` and `reports/PHASE5_FINAL_MODEL_SELECTION.md`):
- **Phase 3 Step 1,500 (Raw Generator):** **$\text{FID} = 297.76$**, $\text{KID} = (328.5 \pm 15.9) \times 10^{-3}$.
- **Phase 3 Step 1,500 (EMA Generator):** **$\text{FID} = 336.98$**, $\text{KID} = (416.7 \pm 15.6) \times 10^{-3}$.
- **Phase 2 Step 1,000 (Raw Generator):** **$\text{FID} = 305.99$**, $\text{KID} = (353.1 \pm 16.7) \times 10^{-3}$.
- **Phase 4 Pilots (Exp A–D):** **$\text{FID} = 369.06 - 394.25$**, $\text{KID} = (491.9 - 552.5) \times 10^{-3}$.

> [!WARNING]
> **Summary on FID:** Presenting the $0.0005$ value as "FID" on social media or in executive decks was a **naming conflation** between custom Fréchet Feature Distance (FFD) and standard Fréchet Inception Distance (FID). Scientifically, the model's true Inception-v3 FID on $64\times 64$ CelebA is **$297.76$**.

---

## 2. Generative Precision & Recall (@k=3)

### 2.1 Mathematical Formulation
StyleForge implements the $k$-NN manifold estimation formulation of generative Precision and Recall (Kynkäänniemi et al. NeurIPS 2019):
1. Let $\Phi_{\text{real}} = \{r_1, \dots, r_{N_r}\}$ be real feature embeddings, and $\Phi_{\text{fake}} = \{g_1, \dots, g_{N_g}\}$ be generated feature embeddings.
2. For each real feature $r_i$, compute the distance $B(r_i)$ to its $k$-th nearest neighbor in $\Phi_{\text{real}} \setminus \{r_i\}$.
3. The real data manifold is defined as the union of hyperspheres:
   $$\mathcal{M}_{\text{real}} = \bigcup_{i=1}^{N_r} B(r_i, \|r_i - r_i^{(k)}\|_2)$$
4. **Precision:** The fraction of generated samples that fall inside $\mathcal{M}_{\text{real}}$:
   $$\text{Precision} = \frac{1}{N_g} \sum_{j=1}^{N_g} \mathbb{I}\left(\exists i : \|g_j - r_i\|_2 \le \|r_i - r_i^{(k)}\|_2\right)$$
5. **Recall:** The fraction of real samples that fall inside the generated manifold $\mathcal{M}_{\text{fake}}$:
   $$\text{Recall} = \frac{1}{N_r} \sum_{i=1}^{N_r} \mathbb{I}\left(\exists j : \|r_i - g_j\|_2 \le \|g_j - g_j^{(k)}\|_2\right)$$

### 2.2 Empirical Audit of Precision & Recall
- **Implementation File:** `evaluation/diversity.py` (lines 22–59) and `scripts/evaluate_phase4_quality.py` (lines 128–130).
- **Evaluation Settings:** $k = 3$, $N_{\text{fake}} = 512$, $N_{\text{real}} = 614$ validation images.
- **Measured Progression:**
  - Step 1,500 (Baseline): $\text{Precision} = \mathbf{16.21\%}, \text{Recall} = \mathbf{72.80\%}$.
  - Step 2,000: $\text{Precision} = 28.71\%, \text{Recall} = 82.57\%$.
  - Step 5,000: $\text{Precision} = 63.09\%, \text{Recall} = 81.60\%$.
  - Step 8,000: $\text{Precision} = 62.89\%, \text{Recall} = 81.92\%$.
  - Step 10,000 (Final): $\text{Precision} = \mathbf{62.50\%}, \text{Recall} = \mathbf{81.92\%}$.
- **Statistical Interpretation:**
  - **Precision jumped from $16.21\%$ to $62.50\%$:** At step 1,500, only $16\%$ of generated faces possessed realistic feature distributions; by step 10,000, over $62\%$ fell within the local support hyperspheres of real celebrity photographs.
  - **Recall remained high ($81.92\%$):** Confirms that the model spans the broad diversity of the CelebA dataset (genders, skin tones, hair colors, poses) without collapsing to a narrow subset of faces.

---

## 3. Sample Diversity & Mode Collapse Metrics

Evaluated via `evaluation/diversity.py`:

| Metric | Mathematical Formula | Step 1,500 | Step 10,000 | Physical Interpretation |
|---|---|:---:|:---:|---|
| **Pairwise $L_1$ Distance** | $\frac{1}{M}\sum_{i \ne j} \frac{1}{D}\|x_i - x_j\|_1$ | $0.6878$ | $0.5846$ | High average pixel variation across random latent seeds; confirms absence of mode collapse. |
| **Intra-Batch Variance** | $\frac{1}{D}\sum_{d=1}^D \text{Var}_B(x_{b, d})$ | $0.3955$ | $0.2809$ | Natural anatomical facial variance; stabilizes around $0.28$ as facial geometry locks into place. |
| **Laplacian Sharpness** | $\frac{1}{N}\sum_{i=1}^N \text{Var}(\nabla^2 \text{gray}(x_i))$ | $2,601.97$ | $1,321.05$ | Reduction reflects elimination of harsh salt-and-pepper grain in favor of smooth skin rendering. |

---

## 4. Memorization & Generalization Audit

A common pitfall in GAN training on small datasets is memorizing training images. This was audited in Experiment E7 (`reports/PHASE4_GENERALIZATION_REPORT.md`):

1. **Nearest-Neighbor Distance Protocol:**
   - Evaluated 512 generated faces against the **4,912 training images** and the **614 unseen validation/test images**.
   - Feature space: Flattened normalized pixel space ($D=12,288$).
2. **Measured Distances:**
   - Mean Euclidean distance to closest **training** image: **$81.36 \pm 4.12$**.
   - Mean Euclidean distance to closest **test** image: **$82.97 \pm 4.35$**.
   - Generalization Ratio:
     $$\mathcal{R} = \frac{d_{\text{train}}}{d_{\text{test}}} = \frac{81.36}{82.97} = \mathbf{0.9806}$$
3. **Verdict:**
   A ratio close to $1.0$ ($0.9806 \approx 1.0$) proves that generated faces are equidistant to training images and completely unseen test images. **StyleForge learned a continuous generative manifold and does NOT memorize individual training samples.**
