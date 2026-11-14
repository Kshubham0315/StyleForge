# StyleForge Phase 4: Experiment E1 — EMA Initialization, Bias Correction & Optimization

**Document**: `reports/PHASE4_EMA_ANALYSIS.md`  
**Date**: November 14, 2026  
**Author**: StyleForge AI/ML Research Engineering Team  
**Milestone**: Phase 4 — Architecture Optimization & Generalization  
**Experiment ID**: `E1-EMA-OPTIM`  
**Artifact**: `experiments/phase4/ema/e1_ema_visual_comparison.png`  
**Metrics File**: `experiments/phase4/ema/e1_ema_results.json`  

---

## 1. Executive Summary & Mathematical Framework

In generative adversarial training, Exponential Moving Average (Polyak-Ruppert parameter averaging) is conventionally expected to smooth loss landscapes and lower Fréchet Inception Distance (FID). However, in Phase 2 and Phase 3, standard EMA ($\beta = 0.999$) exhibited higher FID than raw generator weights.

Experiment E1 resolves this behavior through mathematical derivation, code refactoring, automated unit tests, and a controlled 6-way empirical experiment executed on the NVIDIA RTX 4050 GPU.

### Mathematical Formulation of Initialization Bias
For a recurrent EMA update:
$$\theta_\text{EMA}^{(t)} = \beta \theta_\text{EMA}^{(t-1)} + (1 - \beta) \theta^{(t)}$$
Expanding the recurrence from initialization $\theta^{(0)}$ at step $t$:
$$\theta_\text{EMA}^{(t)} = \beta^t \theta^{(0)} + (1 - \beta)\sum_{i=1}^t \beta^{t-i} \theta^{(i)}$$

1. **Uncorrected Lag**: When initialized from model parameters $\theta^{(0)}$, a lingering residual $\beta^t \theta^{(0)}$ contaminates the moving average. For $\beta = 0.999$ at $t = 300$, $0.999^{300} = \mathbf{74.06\%}$ of the weights remain the initial random weights. At $t = 1,000$, $\beta^t = \mathbf{36.77\%}$.
2. **Model-Init Exact Debiasing**:
   Subtracting the decayed initial weights and dividing by total accumulated mass $(1 - \beta^t)$:
   $$\hat{\theta}_\text{debiased}^{(t)} = \frac{\theta_\text{EMA}^{(t)} - \beta^t \theta^{(0)}}{1 - \beta^t}$$
3. **Zero-Init Debiasing**:
   Initializing $\theta_\text{EMA}^{(0)} = 0$ and normalizing by accumulated mass:
   $$\hat{\theta}_\text{zero}^{(t)} = \frac{\theta_\text{EMA}^{(t)}}{1 - \beta^t}$$
4. **Warm-up Scheduling**:
   Instead of a constant $\beta$, dynamically ramping effective decay $\beta(t)$ as updates accumulate:
   $$\beta(t) = \min\left(\beta, \frac{1 + t}{10 + t}\right)$$
   At $t=1$, $\beta(1) = 2/11 \approx 0.1818$, rapidly incorporating early generator updates before asymptotically converging to target decay $0.999$.

---

## 2. Controlled Empirical Benchmark Table (300 Steps)

All 6 conditions were evaluated simultaneously within **one single 300-step training trajectory** ($4.8\text{ kimg}$ presentations) under identical minibatches, seeds, and optimizer steps on authentic CelebA data ($N_\text{val}=614$, official TorchMetrics Inception-v3):

| Condition / Strategy | Init Mode | Warmup | Debiasing | Initial Residual ($\beta^{300}$) | Inception FID $\downarrow$ | KID Mean $\downarrow$ ($\pm \text{std}$) | Pairwise L1 Diversity $\uparrow$ | Batch Pixel Variance $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Instantaneous** | N/A | None | None | 0.00% | **381.75** | **$0.5171 \pm 0.0190$** | 0.6355 | 0.36877 |
| **Standard EMA 0.999** | Model | None | None | 74.06% | 419.32 | $0.6184 \pm 0.0186$ | **0.7158** | **0.43513** |
| **Debiased EMA 0.999** | Model | None | Exact ($\beta^t \theta_0$) | 0.00% | 404.24 | $0.5743 \pm 0.0194$ | 0.6666 | 0.39945 |
| **Debiased EMA 0.999** | Zero | None | Mass ($1 - \beta^t$) | 0.00% | 404.34 | $0.5745 \pm 0.0193$ | 0.6663 | 0.39931 |
| **Warmup EMA 0.999** | Model | Scheduled | None | 0.00% | **383.01** | **$0.5181 \pm 0.0172$** | 0.6419 | 0.37611 |
| **Warmup EMA 0.990** | Model | Scheduled | None | 0.00% | **383.01** | **$0.5181 \pm 0.0172$** | 0.6419 | 0.37611 |

```mermaid
graph LR
    subgraph EMA Strategies (FID at 300 steps)
        S1["Standard EMA 0.999: 419.32 (High Lag)"] --> S2["Debiased EMA: 404.24 (-15.1 pts)"]
        S2 --> S3["Warmup EMA 0.999: 383.01 (-36.3 pts)"]
        R1["Raw Generator: 381.75"] -.-> S3
    end
```

---

## 3. Scientific Discoveries & Verification

1. **Exact Parity Between Zero-Init and Model-Init Debiasing**:
   Model-init debiased FID ($404.24$) and zero-init debiased FID ($404.34$) match within $0.10$ points, mathematically verifying that both formulations correctly eliminate initialization bias.
2. **Superiority of Warm-up Scheduling**:
   Warm-up scheduling ($\beta(t) = \min(0.999, \frac{1+t}{10+t})$) outperformed static debiasing by **$21.23$ FID points** ($383.01$ vs $404.24$), achieving virtual parity with the raw instantaneous generator ($381.75$) while providing moving average noise attenuation.
3. **Diversity vs Fidelity Trade-off**:
   Standard uncorrected EMA achieved the highest pairwise diversity ($0.7158$) and batch variance ($0.43513$), but at the cost of high FID ($419.32$) due to the heavy $74.1\%$ contamination of random initialization weights.
4. **Engineering Acceptance**:
   `training/trainer.py` now includes `num_updates` tracking, bias correction, warm-up scheduling, and full checkpoint serialization. 5 automated tests in `tests/test_ema_bias_correction.py` pass 100%.
