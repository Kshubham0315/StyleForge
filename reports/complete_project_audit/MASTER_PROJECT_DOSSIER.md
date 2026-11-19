# StyleForge — Master Project Dossier
## Comprehensive Technical Investigation, Architecture Audit & Due-Diligence Report

**Date of Audit:** November 15, 2026  
**Auditor:** Senior AI Research Engineer & ML Systems Architect  
**Project:** StyleForge (`c:\Users\Shubham Kumar\Desktop\Gaze_Connec\StyleForge`)  
**Target Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB GDDR6 VRAM), Windows 11 64-bit  
**Framework Environment:** PyTorch 2.5.1+cu121, CUDA 12.1, ONNX Runtime 1.20+, Python 3.10.8  

---

## 1. Executive Overview & Mission Statement

**StyleForge** is an open-source, edge-optimized Generative Adversarial Network (GAN) framework designed to synthesize human facial photographs on consumer-grade hardware. Traditional state-of-the-art generative face models (e.g., NVIDIA StyleGAN2, StyleGAN3, diffusion models) require high-end enterprise GPUs (such as NVIDIA A100/H100 with 40GB–80GB VRAM) for training and high-latency inference. 

StyleForge solves this constraint through a **two-stage hybrid architecture**:
1. **Stage 1 (Generative Core):** A custom StyleGAN-inspired PyTorch generator operating natively at $64\times 64$ resolution, featuring an 8-layer MLP Mapping Network ($Z \to W$), learned $4\times 4$ spatial constant tensor, Adaptive Instance Normalization (AdaIN) style modulation, stochastic noise injection, and multi-scale style mixing.
2. **Stage 2 (Super-Resolution & Face Restoration):** An ONNX Runtime-accelerated neural face restoration model (GFPGAN 1.4) that upscales and restores the coarse $64\times 64$ output to a photorealistic $512\times 512$ facial portrait.

Through this decomposition, StyleForge achieves complete training and real-time inference within a strict **6GB VRAM budget** (operating at peak memory of $1,117\text{ MB} - 1,259.8\text{ MB}$, well under $25\%$ of the RTX 4050's total capacity).

---

## 2. System Architecture & End-to-End Pipeline

The end-to-end synthesis pipeline operates in two distinct computational regimes:

```
[ Random Latent z ~ N(0, I) in R^512 ]
                 │
                 ▼
    [ Pixel Normalization (unit hypersphere) ]
                 │
                 ▼
   [ 8-Layer Mapping Network f: Z -> W ] (lr_mul=0.01, EqualizedLinear)
                 │
                 ▼
    [ Disentangled Style Vector w in R^512 ]
                 │
                 ▼ (Optional Truncation: w' = w_avg + psi * (w - w_avg))
                 │
                 ▼
    [ Multi-Layer Style Modulation (AdaIN) ]
                 │
    [ Learned Constant 4x4x512 ] ──► [ Block 4x4 ] ──► [ Res 8x8 ] ──► ... ──► [ Res 64x64 ] ──► [ ToRGB (Tanh) ]
                                          ▲                  ▲                        ▲
                                    Noise Inj.          Noise Inj.               Noise Inj.
                 │
                 ▼
    [ Native Generated Face: 64x64x3 in [-1, 1] ]
                 │
                 ▼  (Bilinear Resizing to 512x512 + RGB-to-BGR + Normalization)
                 │
    [ Stage 2: GFPGAN 1.4 ONNX Neural Restoration Engine ]
                 │
                 ▼
    [ Restored Final Portrait: 512x512x3 (PNG / JPEG) ]
```

### Stage 1: Generative Core Architecture
- **Latent Dimension:** $z \in \mathbb{R}^{512}$, sampled from standard Gaussian $\mathcal{N}(0, I)$.
- **Mapping Network:** 8 fully connected layers with LeakyReLU ($\alpha=0.2$) and runtime He-scaling (Equalized Learning Rate) with `lr_mul=0.01`. Maps entangled $Z$-space to linear $W$-space ($\mathbb{R}^{512}$).
- **Synthesis Backbone:** 
  - Learned spatial constant tensor $C \in \mathbb{R}^{1 \times 512 \times 4 \times 4}$.
  - Multi-scale synthesis stages: $4\times 4 \to 8\times 8 \to 16\times 16 \to 32\times 32 \to 64\times 64$.
  - Total of 10 style modulation injection points (2 per resolution level).
  - AdaIN: features normalized per instance and modulated via affine scale $y_s$ and bias $y_b$ derived from $w$.
  - Stochastic Noise Injection: learned per-channel scale parameter multiplying spatial Gaussian noise maps.
  - ToRGB: $1\times 1$ convolution projecting 64-channel features to 3 RGB channels with Tanh activation.
- **Generator Parameters:** $14,535,875$ parameters ($55.45\text{ MB}$ FP32).

### Critic / Discriminator Architecture
- **Input:** $3\times 64\times 64$ images in range $[-1.0, 1.0]$.
- **Backbone:** Convolutional downsampling blocks with residual $1\times 1$ skip connections ($64\times 64 \to 32\times 32 \to 16\times 16 \to 8\times 8 \to 4\times 4$).
- **Minibatch Standard Deviation:** Injected at $4\times 4$ resolution. Calculates standard deviation across the minibatch (group size $G=4$), averages over channels and spatial locations, and appends 1 extra channel to detect mode collapse.
- **Output:** Unbounded scalar prediction ($D(x) \in \mathbb{R}$).
- **Critic Parameters:** $12,174,721$ parameters ($46.44\text{ MB}$ FP32).

### Stage 2: Face Restoration (GFPGAN 1.4)
- **Model Format:** ONNX model (`checkpoints/enhancer/gfpgan_1.4.onnx`, $340,299,087\text{ bytes} \approx 324.53\text{ MB}$).
- **Runtime Backend:** ONNX Runtime (`CUDAExecutionProvider` with graceful fallback to `CPUExecutionProvider`).
- **Processing:** Takes the $64\times 64$ generated image, bilinearly upsamples to $512\times 512$, normalizes to $[-1, 1]$, converts RGB $\to$ BGR, evaluates through the facial prior network, denormalizes, and converts BGR $\to$ RGB.

---

## 3. Empirical Findings & Reality Reconciliation

The technical investigation cross-referenced all documented claims with codebase implementations, execution logs, and serializations:

| Reported Claim / Metric | Documented Claim | Verified Reality in Repository | Technical Explanation & Reconciliation |
|---|:---:|:---:|---|
| **Generator Parameters** | $14,535,875$ | **$14,535,875$** | **CONFIRMED EXACT.** $2,101,248$ in Mapping Network + $12,434,627$ in Synthesis Network. |
| **Critic Parameters** | $12,174,721$ | **$12,174,721$** | **CONFIRMED EXACT.** Backbone downsampling + Minibatch StdDev + final FC. |
| **Generator FP32 Size** | $\sim 55.45\text{ MB}$ | **$55.45\text{ MB}$** | **CONFIRMED EXACT.** $14,535,875 \times 4\text{ bytes} / (1024^2) = 55.451\text{ MB}$. |
| **Peak VRAM During Training** | $1,255\text{ MB}$ | **$1,117.7\text{ MB} - 1,259.8\text{ MB}$** | **CONFIRMED EXACT.** Peak allocated VRAM during Phase 4 continuation at batch size 16 under AMP is $1,255.0\text{ MB}$. |
| **Inference Latency** | $<0.4\text{ s}$ | **$110 - 280\text{ ms}$ (Warm GPU)** | **CONFIRMED.** Generator takes $\sim 15\text{ ms}$; GFPGAN ONNX takes $\sim 150-250\text{ ms}$. Total latency is well below $400\text{ ms}$. |
| **Training Steps** | 10,000 Steps | **10,000 Steps Serialized** | **CONFIRMED.** Checkpoint `checkpoint_step_010000.pt` exists in `experiments/phase4_training_continuation/checkpoints/` ($361.28\text{ MB}$). |
| **Reported "FID" Metric** | $0.0018 \to 0.0005$ | **FFD on 4-layer conv net** (NOT Inception-v3 FID) | **CRITICAL CLARIFICATION.** The value $0.0005$ is Fréchet Feature Distance on an unnormalized 256-dim custom `PerceptualFeatureBank`, not Inception-v3 FID. Official Inception-v3 FID is $297.76$ for Phase 3 Step 1500 and $369-394$ for Phase 4 pilots. |
| **Generative Precision@k=3** | $16.21\% \to 62.50\%$ | **$16.21\% \to 62.50\%$** | **CONFIRMED.** Evaluated using $k=3$ nearest neighbors on 256-dim feature vectors across $N=512$ generated samples vs $N=614$ validation samples. |
| **Generative Recall@k=3** | $72.8\% \to 81.92\%$ | **$72.80\% \to 81.92\%$** | **CONFIRMED.** Evaluated on same feature manifold; proves absence of complete mode collapse. |
| **Training Speedup Claim** | $6.5\times$ Speedup | **$1.192\text{ s} \to 0.182\text{ s}$ per step** | **CONFIRMED.** Moving from continuous Gradient Penalty ($k=1, n_{\text{critic}}=5$) to Lazy R1 ($k=16, n_{\text{critic}}=1$) yielded an exact $6.55\times$ step latency reduction. |

---

## 4. Dataset & Data Integrity

- **Source Dataset:** CelebFaces Attributes Dataset (CelebA) aligned & cropped subset (`data/celeba/img_align_celeba`).
- **Disk Sample Count:** Exactly **$6,140$ images** on disk.
- **Partition Manifest:** `data/celeba/list_eval_partition.txt` defines an exact disjoint split:
  - **Training Set (code 0):** $4,912$ images ($80.0\%$).
  - **Validation Set (code 1):** $614$ images ($10.0\%$).
  - **Test Set (code 2):** $614$ images ($10.0\%$).
- **Data Leakage:** **Zero overlap.** Verified by `tests/test_dataset_split.py` and `data/celeba/dataset_metadata.json` (`train_val_overlap = 0`).
- **Preprocessing Pipeline:** `transforms.CenterCrop(140)` $\to$ `transforms.Resize((64, 64))` $\to$ `transforms.ToTensor()` $\to$ `transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])`.

---

## 5. Production Serving, Backend & Ecosystem

StyleForge is not merely a model script; it possesses an enterprise-grade MLOps and serving ecosystem:
1. **Core Inference Engine (`core/engine/`):**
   - Thread-safe singleton `InferenceEngine`.
   - Device manager with dynamic VRAM telemetry.
   - Comprehensive configuration schemas (`GenerationConfig`, `InterpolationConfig`, `StyleMixConfig`).
   - Image postprocessor with base64 and PIL grid conversion.
2. **FastAPI REST API (`backend/`):**
   - Full REST API with endpoints `/api/v1/generate`, `/api/v1/interpolate`, `/api/v1/mix-styles`, `/api/v1/models`, `/health`, `/ready`, `/metrics`.
   - Security: Master API key validation via constant-time digest comparison (`secrets.compare_digest`).
   - Rate Limiting: In-memory sliding window rate limiter (`InMemoryRateLimiter`) with stale IP bounding.
   - Concurrency Control: `AsyncThreadSafeSemaphore` bounding simultaneous GPU execution to protect VRAM.
   - Database: SQLAlchemy ORM supporting SQLite (`data/styleforge.db`) and PostgreSQL for persistent generation records and model registry.
   - Telemetry: Prometheus client exporting request count, error count, and inference latency.
3. **Interactive Studio (`inference/app.py`):**
   - Gradio web interface providing live sliders for seed, sample count, truncation $\psi$, EMA toggle, latent interpolation, and multi-scale style mixing.
4. **React Frontend (`frontend/`):**
   - Modern React 18 + Vite + TypeScript dashboard (`GeneratorView`, `DashboardView`, `HistoryView`, `ModelInfoView`, `SystemStatusView`).
5. **Containerization & Deployment (`docker/`):**
   - Multi-stage Dockerfiles (`backend.Dockerfile`, `frontend.Dockerfile`), Nginx reverse proxy configuration, and `docker-compose.yml` / `docker-compose.gpu.yml`.

---

## 6. Audit Verdict & Project Maturity

- **Algorithmic Maturity:** Research & Engineering Grade (TRL 6–7). Core StyleGAN components (AdaIN, Mapping, Constant input, Noise injection, Lazy R1, TTUR, EMA) are genuinely implemented and verified.
- **Software Engineering Maturity:** Production Grade (Clean modular architecture, 100% test passing on models/core/enhancer/dataset/losses, Docker compose, FastAPI backend, Prometheus observability).
- **Generative Quality Reality:** Native $64\times 64$ faces are structurally sound and diverse, but coarse and noisy when unenhanced due to limited training steps ($1,500 - 10,000$ steps vs $100,000+$ in production GANs) and stochastic noise overdrive. Stage 2 (GFPGAN ONNX) carries the primary burden of rendering realistic high-frequency skin textures, eyes, and hair at $512\times 512$.
