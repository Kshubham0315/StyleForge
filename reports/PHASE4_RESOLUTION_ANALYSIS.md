# StyleForge Phase 4 — Experiment E6: 128×128 Progressive Scaling Feasibility

**Date:** November 14, 2026  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM, Ada Lovelace), Windows 11  
**Software:** PyTorch 2.5.1 + CUDA 12.1, Python 3.10.8  
**Scope:** VRAM Profiling, Computational Graph Allocations, and Throughput Benchmarks across $64\times 64$ and $128\times 128$  

---

## 1. Scientific & Engineering Rationale

Scaling GAN synthesis from $64\times 64$ to $128\times 128$ quadruples the spatial dimensions ($4\times$ pixels, $16,384$ vs $4,096$ values per channel), increasing intermediate feature activations and backward gradient graphs. On consumer laptop hardware with a strict 6.00 GB physical VRAM boundary, unprofiled resolution scaling risks CUDA Out-Of-Memory (`CUDA OOM`) crashes during critic gradient penalty computation ($\nabla_x D(x)$).

Experiment E6 empirical measures:
1. Peak VRAM consumption including model weights, Adam optimizer states ($2\times$ parameter memory), forward activations, and backward computation graphs.
2. Safe batch size boundaries at $128\times 128$.
3. Per-step latency and hardware headroom.

---

## 2. Quantitative Profiling Benchmark

| Target Resolution | Batch Size | Peak VRAM Used (GB) | Hardware Headroom (GB) | Step Latency (ms) | Status | Feasibility Assessment |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **$64\times 64$** | 16 | $1.15\text{ GB}$ | $4.85\text{ GB}$ | $577.9\text{ ms}$ | **SUCCESS** | High throughput baseline |
| **$64\times 64$** | 32 | $2.07\text{ GB}$ | $3.92\text{ GB}$ | $559.3\text{ ms}$ | **SUCCESS** | Ample headroom, optimal batch size |
| **$128\times 128$** | 4 | $1.05\text{ GB}$ | $4.95\text{ GB}$ | $350.1\text{ ms}$ | **SUCCESS** | Ultra-safe memory footprint |
| **$128\times 128$** | 8 | $1.84\text{ GB}$ | $4.15\text{ GB}$ | $565.2\text{ ms}$ | **SUCCESS** | **Recommended production setting** |
| **$128\times 128$** | 16 | $3.41\text{ GB}$ | $2.59\text{ GB}$ | $1008.3\text{ ms}$ | **SUCCESS** | Feasible; $2.59\text{ GB}$ safety margin |

---

## 3. Engineering Recommendations for $128\times 128$ Execution

1. **Recommended Batch Size:**  
   - **Batch size 8** ($1.84\text{ GB}$ peak VRAM, $565.2\text{ ms/step}$) provides the ideal balance between gradient estimate quality and latency.
   - It leaves **$4.15\text{ GB}$ ($69.2\%$) of VRAM free**, completely insulating the system against memory fragmentation spikes during PyTorch CUDA memory management.
2. **Batch size 16 Feasibility:**  
   - Batch size 16 is fully operational ($3.41\text{ GB}$ peak VRAM), leaving $2.59\text{ GB}$ of headroom.
3. **Progressive Growing vs Direct Synthesis:**  
   - Progressive growing ($64\to 128$ with fade-in parameter $\alpha$) is mathematically viable because both $64\times 64$ and $128\times 128$ stay well under $3.5\text{ GB}$ VRAM.
   - However, given that direct $128\times 128$ synthesis operates comfortably at $1.84\text{ GB}$, direct training with warm-up EMA avoids the hyperparameter sensitivity and phase transitions of progressive growing.

---

## 4. Visual Artifacts

- VRAM Allocation & Hardware Headroom Chart: `experiments/phase4/resolution/e6_vram_feasibility.png`
- Raw Profiling Metrics JSON: `experiments/phase4/resolution/e6_resolution_profiling.json`
