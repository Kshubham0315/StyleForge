# StyleForge Phase 4 — Experiment E7: Generalization, Memorization & Leakage Audit

**Date:** November 14, 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM), Windows 11  
**Software:** PyTorch 2.5.1 + CUDA 12.1, Python 3.10.8  
**Scope:** Cryptographic Dataset Contamination Check, Pixel-Level & Feature-Level Nearest-Neighbor Retrieval, Memorization vs. Generalization Diagnosis  

---

## 1. Scientific Context & Objectives

A critical question in deep generative modeling (particularly GANs trained on small-to-medium subsets of CelebA) is whether the generator has learned continuous latent manifold representations or is merely memorizing training exemplars.

Furthermore, any empirical claim of generalization is invalid if data leakage exists between the training split and validation/test splits.

Experiment E7 conducts a two-fold scientific audit:
1. **Cryptographic Leakage Audit:** Exhaustive MD5 hash uniqueness and intersection check across Train ($4,912$), Validation ($614$), and Test ($614$) splits.
2. **Nearest-Neighbor Retrieval Probe:** Quantitative distance comparison from generated synthetic samples to their nearest neighbors in the training set versus unseen test images in both:
   - Raw Pixel $L_2$ Euclidean space
   - Deep Inception-v3 pool3 ($2048$-dim) perceptual feature space.

---

## 2. Experimental Results & Quantitative Findings

### 2.1 Cryptographic Dataset Leakage Audit

| Split | Sample Count | Unique MD5 Hashes | Overlap with Train | Overlap with Val | Overlap with Test |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | $4,912$ | $4,912$ | — | $0$ ($0.0\%$) | $0$ ($0.0\%$) |
| **Validation** | $614$ | $614$ | $0$ ($0.0\%$) | — | $0$ ($0.0\%$) |
| **Test** | $614$ | $614$ | $0$ ($0.0\%$) | $0$ ($0.0\%$) | — |

**Verdict:** **Zero Data Contamination.** All three splits are mathematically disjoint, guaranteeing that evaluation metrics reflect unseen generalization performance.

---

### 2.2 Nearest-Neighbor Memorization Probing

To test the hypothesis of verbatim memorization:
- If $d(\text{gen}, \text{train}) \ll d(\text{gen}, \text{test})$, the model is overfitting/memorizing training instances.
- If $d(\text{gen}, \text{train}) \approx d(\text{gen}, \text{test})$, the model generates samples with comparable statistical distance to both seen and unseen real distributions.

| Evaluation Metric / Space | Distance to Train Set (Mean $\pm$ Std) | Distance to Test Set (Mean $\pm$ Std) | Distance Ratio ($\frac{d_\text{train}}{d_\text{test}}$) | Memorization Diagnosis |
|---|:---:|:---:|:---:|---|
| **Raw Pixel $L_2$ Space** | $81.36 \pm 5.26$ | $82.97 \pm 5.44$ | $\mathbf{0.9806}$ | Parity within $1.9\%$ |
| **Inception-v3 Perceptual Space** | $19.23$ | $19.24$ | $\mathbf{0.9996}$ | Parity within $0.04\%$ |

---

## 3. Discussion & Scientific Conclusions

1. **Parity in Perceptual Distance:**  
   In the Inception feature manifold, the ratio of minimum distance to training samples versus unseen test samples is **$0.9996$**, demonstrating near-perfect statistical symmetry.
2. **Absence of Direct Memorization:**  
   Under the tested protocols, generated faces do not replicate individual training images. Visual inspection of nearest neighbors confirms that while overall facial poses, lighting conditions, and hair colors correlate with common modes, facial identities, fine geometries, and backgrounds are synthesized combinations rather than direct copies.
3. **Formal Scientific Statement:**  
   *No strong evidence of direct training-image memorization was observed under tested nearest-neighbor protocols.*

---

## 4. Visual Artifacts

- 5-Way Query vs Train NN vs Test NN Visual Grid: `experiments/phase4/generalization/e7_nearest_neighbor_grid.png`
- Raw Verification JSON: `experiments/phase4/generalization/e7_generalization_results.json`
