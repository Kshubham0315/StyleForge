# StyleForge Phase 4 — Experiment E4 & E5: Architecture Capacity & Resource Scaling

**Date:** November 14, 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM), Windows 11  
**Software:** PyTorch 2.5.1 + CUDA 12.1, Python 3.10.8  
**Scope:** Parameter Scaling, VRAM Footprint, Step Latency, and Convergence Trade-Offs at $64\times 64$  

---

## 1. Experimental Overview

To establish the optimal architectural configuration under the hardware budget (6GB VRAM on an RTX 4050 Laptop GPU), we evaluated the effects of scaling critic capacity:
1. **Standard Capacity (Base=64):** `base_channels = 64, max_channels = 512` for `StyleCritic`; `base_channels = 512` for `StyleGenerator`.
2. **Scaled Wide Capacity (Base=128):** `base_channels = 128, max_channels = 512` for `StyleCritic`, increasing low-level feature extraction capacity.

Both models were trained on authentic CelebA ($64\times 64$) under identical conditions ($n_\text{critic}=2$, WGAN-GP $\lambda=10.0$, batch size 16) for 150 steps and evaluated on the full 614-image validation set.

---

## 2. Quantitative Results & Trade-Off Matrix

| Configuration | Generator Params | Critic Params | Total Params | Peak VRAM (GB) | Step Latency (ms/step) | Validation FID (150 steps) ↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard (Base=64)** | $14,535,875$ | $12,174,721$ | $\mathbf{26,710,596}$ | $\mathbf{1.42\text{ GB}}$ | $\mathbf{481.4\text{ ms}}$ | $\mathbf{390.96}$ |
| **Scaled (Base=128)** | $14,535,875$ | $16,927,489$ | $31,463,364$ | $2.28\text{ GB}$ | $1060.4\text{ ms}$ | $393.78$ |
| **Delta (%)** | $0.0\%$ | $+39.0\%$ | $+17.8\%$ | $+60.6\%$ | $+120.3\%$ | $+2.82\text{ pts}$ |

---

## 3. Scientific Findings & Architecture Decision

1. **Severe Latency Penalty:**  
   Doubling the critic base channels from 64 to 128 increased per-step training latency by **$+120.3\%$** ($481.4\text{ ms} \to 1060.4\text{ ms}$), cutting training throughput by more than half.
2. **Memory Footprint:**  
   Peak VRAM increased from $1.42\text{ GB}$ to $2.28\text{ GB}$ ($+60.6\%$). While $2.28\text{ GB}$ remains safely within the 6.0 GB hardware ceiling, it substantially diminishes the memory headroom required for scaling to $128\times 128$ resolution.
3. **Critic-Generator Imbalance:**  
   The scaled critic achieved a slightly worse validation FID ($393.78$ vs $390.96$). With $+39\%$ more parameters, the critic learns faster than the generator, producing sharper gradients that can destabilize generator parameter updates without corresponding improvements in visual fidelity.
4. **Architectural Selection:**  
   `base_channels = 64` represents the Pareto-optimal configuration for $64\times 64$ synthesis on 6GB VRAM, providing optimal training speed ($>2$ steps/sec) and stable adversarial dynamics.

---

## 4. Visual Artifacts

- Resource & Performance Trade-Off Bar Charts: `experiments/phase4/capacity/e4_capacity_tradeoffs.png`
- Raw Metrics JSON: `experiments/phase4/capacity/e4_capacity_results.json`
