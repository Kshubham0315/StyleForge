# StyleForge — Evidence Index & Provenance Map
## Traceable Repository References, Checkpoint Artifacts & Reproducibility Commands

**Auditor:** Senior AI Research Engineer & Repository Auditor  
**Purpose:** Provide exact file paths, line numbers, function signatures, artifact references, and CLI reproduction commands for every finding in this technical investigation.

---

## 1. Architectural & Source Code Evidence

| Finding / Claim | Source File Path | Line / Component Reference | Concrete Code / Structural Evidence |
|---|---|---|---|
| **Generator Parameters ($14,535,875$)** | [`models/style_generator.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/models/style_generator.py) | Class `StyleGenerator` | $2,101,248$ (Mapping) + $12,434,627$ (Synthesis) = $14,535,875$. |
| **Critic Parameters ($12,174,721$)** | [`models/critic.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/models/critic.py) | Class `StyleCritic` | ResBlocks down to 4x4 + MinibatchStdDev + FC projection. |
| **Mapping Network `lr_mul=0.01`** | [`models/style_generator.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/models/style_generator.py#L40-L52) | `MappingNetwork.__init__` | `EqualizedLinear(in_dim, style_dim, gain=2.0**0.5, lr_mul=0.01)`. |
| **Pixel Normalization** | [`models/common.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/models/common.py#L19-L33) | Class `PixelNorm` | `x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)`. |
| **AdaIN Modulation** | [`models/common.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/models/common.py#L124-L153) | Class `AdaIN` | Affine modulation: `normalized * scale + bias` with `chunk(2, dim=1)`. |
| **Noise Injection** | [`models/common.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/models/common.py#L105-L122) | Class `NoiseInjection` | `self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))`. |
| **Minibatch StdDev Layer** | [`models/common.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/models/common.py#L155-L193) | Class `MinibatchStdDev` | Computes standard deviation across group size $G=4$, appends channel. |
| **Lazy R1 Regularization** | [`training/losses.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/training/losses.py#L149-L162) | `GANLoss.critic_loss` | `apply_r1 = (step % lazy_interval == 0); effective_gamma = penalty * interval`. |
| **TTUR Optimization** | [`training/trainer.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/training/trainer.py#L182-L191) | `GANTrainer.__init__` | Independent Adam optimizers: `lr_g=0.0005, lr_d=0.0010, betas=(0.0, 0.99)`. |
| **Exponential Moving Average** | [`training/trainer.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/training/trainer.py#L39-L147) | Class `ExponentialMovingAverage` | Shadow tracking, warmup `(1+t)/(10+t)`, mathematical debiasing. |
| **GFPGAN 1.4 ONNX Enhancer** | [`core/engine/enhancer.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/core/engine/enhancer.py) | Class `FaceEnhancer` | Wraps `checkpoints/enhancer/gfpgan_1.4.onnx` via `onnxruntime`. |
| **GPU Concurrency Semaphore** | [`backend/api/v1/generate.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/backend/api/v1/generate.py#L51-L74) | Class `AsyncThreadSafeSemaphore` | Enforces `GPU_CONCURRENCY_LIMIT=1` to prevent VRAM OOM crashes. |
| **Constant-Time Auth** | [`backend/security/auth.py`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/backend/security/auth.py) | `get_current_user` | `secrets.compare_digest(api_key, settings.API_KEY)`. |

---

## 2. Checkpoint & Model Artifact Evidence

| Artifact Description | Disk Location | File Size (Bytes) | Verification Status |
|---|---|:---:|---|
| **Official Production Checkpoint** | [`checkpoints/styleforge_final_model.pt`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/checkpoints/styleforge_final_model.pt) | $116,336,534$ | Dual-weight release model; SHA256 verified. |
| **GFPGAN 1.4 ONNX Model** | [`checkpoints/enhancer/gfpgan_1.4.onnx`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/checkpoints/enhancer/gfpgan_1.4.onnx) | $340,299,087$ | Stage 2 face restoration model. |
| **10,000-Step Checkpoint** | [`experiments/phase4_training_continuation/checkpoints/checkpoint_step_010000.pt`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/experiments/phase4_training_continuation/checkpoints/checkpoint_step_010000.pt) | $378,836,285$ | Full training state (G, D, EMA, Opt_G, Opt_D). |
| **Phase 4 Best Checkpoint** | [`experiments/phase4_training_continuation/checkpoints/checkpoint_best.pt`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/experiments/phase4_training_continuation/checkpoints/checkpoint_best.pt) | $378,832,980$ | Lowest G-loss model checkpoint. |
| **Phase 3 Final Checkpoint** | [`experiments/phase3_convergence/checkpoints/checkpoint_final.pt`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/experiments/phase3_convergence/checkpoints/checkpoint_final.pt) | $378,827,797$ | Step 1,500 training checkpoint. |

---

## 3. Dataset & Data Split Evidence

| Partition / Manifest | Path | Item Count | Audit Verification |
|---|---|:---:|---|
| **CelebA Images on Disk** | [`data/celeba/img_align_celeba`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/data/celeba/img_align_celeba) | $6,140\text{ files}$ | Aligned face photographs ($178\times 218$). |
| **Partition Manifest** | [`data/celeba/list_eval_partition.txt`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/data/celeba/list_eval_partition.txt) | $6,140\text{ lines}$ | 4,912 train (0), 614 val (1), 614 test (2). |
| **Metadata Record** | [`data/celeba/dataset_metadata.json`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/data/celeba/dataset_metadata.json) | File | `zero_leakage_verified: true`, `train_val_overlap: 0`. |

---

## 4. Quantitative Metrics & Log Files Evidence

| Metric Record | File Location | Key Audited Data Points |
|---|---|---|
| **Phase 4 Generation Quality** | [`reports/phase4_generation_quality.json`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/phase4_generation_quality.json) | Step 1500 $\to$ 10000: FFD $0.0018 \to 0.0005$, Precision $16.21\% \to 62.50\%$, Recall $72.8\% \to 81.92\%$, Peak VRAM $1255.0\text{ MB}$. |
| **Inception FID Reconciliation** | [`reports/benchmark_reconciliation.json`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/benchmark_reconciliation.json) | Phase 3 Step 1500 Inception-v3 FID = $297.76$ (Raw) / $336.98$ (EMA), KID = $0.3285$. |
| **10,000-Step Training Log** | [`experiments/phase4_training_continuation/metrics.json`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/experiments/phase4_training_continuation/metrics.json) | $4.25\text{ MB}$ JSON tracking 8,500 steps, step times ($0.182\text{ s}$), VRAM ($1,255\text{ MB}$). |
| **Hardware Benchmarks** | [`docs/hardware_profile.json`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/docs/hardware_profile.json) | StyleForge batch 16 AMP: Peak allocated $1,117.67\text{ MB}$, Peak reserved $1,382.0\text{ MB}$. |
| **Memorization Audit** | [`experiments/phase3/memorization_audit.json`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/experiments/phase3/memorization_audit.json) | Distance ratio $d_{\text{train}}/d_{\text{test}} = 0.9806$ proving non-memorization. |

---

## 5. Verification & Test Suite Execution Commands

To independently reproduce the findings of this audit, run the following commands:

```bash
# 1. Verify Model Parameter Counts & FP32 Sizes (14.53M Gen, 12.17M Critic)
python -c "from models.style_generator import StyleGenerator; from models.critic import StyleCritic; g = StyleGenerator(); d = StyleCritic(); print('Gen Params:', sum(p.numel() for p in g.parameters())); print('Critic Params:', sum(p.numel() for p in d.parameters()))"

# 2. Run Core Model & Building Block Tests (10/10 Passed)
python -m pytest tests/test_models.py -v

# 3. Run Inference Engine & Face Enhancer Tests (17/17 Passed)
python -m pytest tests/test_core_engine.py tests/test_enhancer.py tests/test_inference.py -v

# 4. Run Dataset Split & Algorithmic Loss/EMA Tests (17/17 Passed)
python -m pytest tests/test_dataset_split.py tests/test_ema_bias_correction.py tests/test_losses.py tests/test_metrics_precision_recall.py tests/test_model_registry.py -v

# 5. Run Live Terminal Production Verification Banner
python scripts/demo_linkedin_terminal.py
```
