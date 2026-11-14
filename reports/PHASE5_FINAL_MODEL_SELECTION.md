# StyleForge Phase 5 — Final Model Selection

**Date:** November 14, 2026  
**Status:** Official Release Checkpoint Designated  
**Production Checkpoint:** `checkpoints/styleforge_final_model.pt` ($110.95\text{ MB}$)  
**Architecture:** StyleForge Style-Based Generator ($14,535,875$ parameters)  
**Target Resolution:** $64\times 64$  

---

## 1. Candidate Evaluation Matrix

To select the definitive model checkpoint for the open-source release, four primary candidate configurations were evaluated across quantitative metrics, visual structure, facial landmark coherence, and generalization:

| Candidate Model | Global Training Steps | Adversarial Objective | Inception-v3 FID ↓ | KID $\times 10^3$ ↓ | Diversity ($L_1$) ↑ | Visual Coherence | Selection Verdict |
|---|:---:|---|:---:|:---:|:---:|---|:---:|
| **Candidate A: Phase 3 Step 1500 (Raw)** | **1,500** | WGAN-GP | $\mathbf{297.76}$ | $\mathbf{328.5 \pm 15.9}$ | $0.5779$ | Sharp facial boundaries, distinct eyes, nose, hair | **PRIMARY PRODUCTION CHOICE** |
| **Candidate B: Phase 3 Step 1500 (EMA)** | **1,500** | WGAN-GP | $336.98$ | $416.7 \pm 15.6$ | $\mathbf{0.6223}$ | Smoother skin textures, softer contours | **SECONDARY EMA MODE** |
| **Candidate C: Phase 2 Step 1000 (Raw)** | 1,000 | WGAN-GP | $305.99$ | $353.1 \pm 16.7$ | $0.5953$ | Intermediate structural definition | Historical Predecessor |
| **Candidate D: Phase 3 Logistic + $R_1$** | 300 | Logistic + $R_1$ | $373.66$ | $497.5 \pm 19.8$ | $0.6692$ | High variety, early coarse geometry | Early Training Stage |

---

## 2. Selection Rationale

1. **Quantitative Performance:**  
   Candidate A achieved the lowest Fréchet Inception Distance (**$297.76$**) and lowest Kernel Inception Distance (**$0.3285$**) across all tested checkpoints, representing the optimal facial feature distribution relative to CelebA.
2. **Visual Feature Sharpness:**  
   Fixed-seed comparative analysis reveals that Candidate A synthesizes significantly more defined facial landmarks (pupil placement, mouth contours, hairline boundaries) than early-stage checkpoints ($t \le 300$).
3. **Generalization Proof:**  
   In Experiment E7, Candidate A demonstrated statistically identical distance to training images ($81.36$) and unseen test images ($82.97$) with a distance ratio of **$0.9806$**, confirming that its superior FID is driven by continuous manifold learning rather than training set memorization.
4. **Dual Weight Support:**  
   The exported production checkpoint (`checkpoints/styleforge_final_model.pt`) includes **both Candidate A (Raw weights) and Candidate B (EMA weights)**, allowing users to toggle between sharpness (Raw) and smooth interpolation stability (EMA) via the inference interface.

---

## 3. Production Checkpoint Specifications

- **File Path:** `checkpoints/styleforge_final_model.pt`
- **File Size:** $110.95\text{ MB}$ (optimizer states removed; $70\%$ storage reduction vs $361\text{ MB}$ raw checkpoint).
- **Contents:**
  - `generator`: State dictionary for Raw Generator (FID: $297.76$).
  - `ema_generator`: State dictionary for EMA Generator (FID: $336.98$).
  - `metadata`: Complete architectural parameters, resolution ($64$), latent dimensions ($512$).
- **License:** MIT License.
