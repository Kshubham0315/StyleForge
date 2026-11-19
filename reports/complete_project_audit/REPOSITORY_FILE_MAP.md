# StyleForge — Repository File Map & Implementation Directory

**Auditor:** Senior AI Research Engineer & ML Systems Architect  
**Scope:** Complete structural, functional, and dependency mapping of all code, configuration, checkpoint, and asset files.

---

## 1. High-Level Repository Tree

```
StyleForge/
├── .github/                      # CI/CD Workflows (tests, linting, docker builds)
├── artifacts/                    # Evaluation and visual comparison assets
│   ├── enhancer_test/            # GFPGAN test outputs
│   ├── independent_validation/   # Independent validation run artifacts
│   └── seed_scout/               # Qualitative seed exploration grids
├── assets/                       # Architectural diagrams & scorecards
├── backend/                      # FastAPI v2.0 Production Server
│   ├── api/v1/                   # REST API routes (generate, models, metrics)
│   ├── database/                 # SQLAlchemy ORM models, session & repositories
│   ├── metrics/                  # Prometheus instrumentation
│   ├── schemas/                  # Pydantic v2 request/response schemas
│   ├── security/                 # API Key authentication & rate limiting
│   ├── storage/                  # Local and S3 storage handlers
│   ├── config.py                 # Backend pydantic-settings
│   └── main.py                   # FastAPI application factory & lifespan
├── checkpoints/                  # Official release model weights
│   ├── enhancer/                 # GFPGAN 1.4 ONNX model (340 MB)
│   └── styleforge_final_model.pt # Exported dual-state release model (110.9 MB)
├── configs/                      # YAML training and ablation configurations
├── core/                         # Core inference engine & MLOps abstractions
│   ├── engine/                   # InferenceEngine, ModelLoader, Enhancer, Postprocessor
│   ├── mlops/                    # Tracking and experiment telemetry
│   └── registry/                 # Dynamic model registry
├── data/                         # CelebA dataset, partition file & SQLite DB
│   ├── celeba/                   # img_align_celeba (6,140 images) + partition
│   └── dataset.py                # PyTorch Dataset loaders & transforms
├── docker/                       # Dockerfiles & Nginx configurations
├── docs/                         # Engineering documentation & technical reports
├── evaluation/                   # Evaluation metrics (FID/KID, diversity, explorer)
├── experiments/                  # Experiment logs, checkpoints & sample grids
│   ├── baseline_wgan_gp/         # Initial WGAN-GP experiment
│   ├── phase2_stylegan_adain/    # Phase 2 StyleGAN baseline
│   ├── phase3_convergence/       # 1,500-step convergence run (source of release model)
│   ├── phase4_controlled_pilots/ # Pilots A, B, C, D (optimization studies)
│   └── phase4_training_continuation/ # 10,000-step training continuation
├── frontend/                     # React 18 + Vite + TypeScript web interface
├── inference/                    # Legacy & Gradio inference wrappers
│   ├── app.py                    # Gradio Interactive Studio
│   └── infer.py                  # High-level StyleForgeInference class
├── models/                       # PyTorch Generative Architectures
│   ├── common.py                 # AdaIN, PixelNorm, EqualizedConv2d, MinibatchStdDev
│   ├── critic.py                 # StyleCritic multi-scale discriminator
│   ├── style_generator.py        # StyleGenerator, MappingNetwork, SynthesisNetwork
│   └── wgan_baseline.py          # Conv baseline generator & critic
├── monitoring/                   # Prometheus & Grafana configuration
├── notebooks/                    # Jupyter walkthrough notebook
├── outputs/                      # Generated sample grids, sweeps & interpolations
├── reports/                      # Formal engineering reports & validation records
├── scripts/                      # CLI tools for training, evaluation & export
├── tests/                        # 14 PyTest test suites (unit, regression, integration)
└── training/                     # Trainer loop, loss functions, lazy R1 & augmentation
```

---

## 2. Exhaustive File-by-File Investigation

### A. Core Generative Models (`models/`)

#### 1. `models/style_generator.py`
- **Purpose:** Implements the complete StyleGAN-inspired generator.
- **Implemented Components:**
  - `MappingNetwork`: 8-layer MLP ($512 \to 512$) with PixelNorm, runtime He-scaling, and `lr_mul=0.01`.
  - `StyleBlock`: Single synthesis block combining `EqualizedConv2d`, `NoiseInjection`, `LeakyReLU(0.2)`, and `AdaIN`.
  - `SynthesisNetwork`: Multi-resolution generator starting from learned constant $4\times 4\times 512$ up to $64\times 64$, including `to_rgb` projection with Tanh.
  - `StyleGenerator`: Top-level model orchestrating mapping, synthesis, running style tracking ($\bar{w}$), truncation trick ($w' = \bar{w} + \psi(w - \bar{w})$), and style mixing.
- **Inputs & Outputs:** Input: Latent tensor $z \in \mathbb{R}^{B \times 512}$. Output: Image tensor $x \in \mathbb{R}^{B \times 3 \times 64 \times 64}$ in range $[-1.0, 1.0]$.
- **Dependencies:** `torch`, `torch.nn`, `models.common`.
- **Callers & Dependents:** Called by `training/trainer.py`, `core/engine/model_loader.py`, `inference/infer.py`, `scripts/train.py`, `scripts/evaluate_phase4_quality.py`.
- **Status:** **IMPLEMENTED & VERIFIED.** All forward passes, style mixing, and truncation verified by `tests/test_models.py`.
- **Production Use:** **YES.** Active generator backbone in both Stage 1 inference and training.

#### 2. `models/critic.py`
- **Purpose:** Implements the multi-scale downsampling Critic (Discriminator).
- **Implemented Components:**
  - `ConvBlock`: Residual downsampling block with $3\times 3$ convolutions, LeakyReLU, average pooling, and $1\times 1$ conv shortcut.
  - `StyleCritic`: Multi-scale network downsampling from $64\times 64$ to $4\times 4$, injecting `MinibatchStdDev`, followed by final $3\times 3$ conv and linear projection to an unbounded scalar score.
- **Inputs & Outputs:** Input: Image tensor $x \in \mathbb{R}^{B \times 3 \times 64 \times 64}$. Output: Scalar tensor $s \in \mathbb{R}^{B \times 1}$.
- **Dependencies:** `torch`, `torch.nn`, `models.common`.
- **Callers & Dependents:** Called by `training/trainer.py`, `training/losses.py`, `scripts/train.py`.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by `tests/test_models.py`.
- **Production Use:** Used during training and offline validation; not required for pure generation inference.

#### 3. `models/common.py`
- **Purpose:** Houses all shared building blocks for StyleGAN and baseline architectures.
- **Implemented Components:**
  - `PixelNorm`: Unit hypersphere normalization across channels: $y = x / \sqrt{\frac{1}{C}\sum x_i^2 + \epsilon}$.
  - `EqualizedLinear` / `EqualizedConv2d`: Runtime He-scaling ($w = \hat{w} \times \frac{\text{gain}}{\sqrt{\text{fan\_in}}} \times \text{lr\_mul}$) preserving forward variance.
  - `NoiseInjection`: Per-channel learned scaling multiplying spatial Gaussian noise maps.
  - `AdaIN`: Instance normalization with affine modulation: $y = y_s \cdot \left(\frac{x - \mu}{\sigma}\right) + y_b$.
  - `MinibatchStdDev`: Robust group-based batch standard deviation injected at $4\times 4$ resolution.
  - `SelfAttention2d`: SAGAN self-attention mechanism with spectral normalization support.
- **Dependencies:** `torch`, `torch.nn`, `math`.
- **Callers & Dependents:** `models/style_generator.py`, `models/critic.py`, `models/wgan_baseline.py`.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by unit tests in `tests/test_models.py`.
- **Production Use:** **YES.** Foundation of the entire neural architecture.

#### 4. `models/wgan_baseline.py`
- **Purpose:** Standard convolutional DCGAN / WGAN baseline for empirical benchmarking.
- **Implemented Components:** `BaselineGenerator` (transposed convolutions) and `BaselineCritic` (strided convolutions).
- **Status:** **IMPLEMENTED & VERIFIED.** Used in Phase 1 & Phase 2 baseline comparisons.

---

### B. Training & Optimization Pipeline (`training/`)

#### 1. `training/trainer.py`
- **Purpose:** Core training orchestration loop.
- **Implemented Components:**
  - `ExponentialMovingAverage (EMA)`: Shadow weight tracking with decay $\beta=0.999$, warmup schedule $\beta(t) = \min(\beta, \frac{1+t}{10+t})$, and mathematical bias correction.
  - `GANTrainer`: Two-stage step execution (critic updates $\times n_{\text{critic}}$, generator update $\times 1$), PyTorch AMP dynamic gradient scaling (`GradScaler`), gradient clipping, TensorBoard logging, metrics history serialization, and periodic sample generation.
- **Inputs & Outputs:** Consumes real images from DataLoader; trains Generator and Critic; outputs serialized `.pt` checkpoints, sample PNG grids, and `metrics.json`.
- **Dependencies:** `torch`, `torch.amp`, `torchvision`, `tqdm`, `training.losses`, `models`.
- **Callers & Dependents:** Called by `scripts/train.py`, `scripts/run_phase4_continuation.py`, `tests/test_integration.py`.
- **Status:** **IMPLEMENTED & VERIFIED.**

#### 2. `training/losses.py`
- **Purpose:** Adversarial loss formulations and regularizers.
- **Implemented Components:**
  - `compute_gradient_penalty`: WGAN-GP 2-sided gradient penalty on random straight-line interpolations.
  - `compute_r1_penalty`: Non-saturating R1 gradient penalty evaluated on real data in FP32.
  - `GANLoss`: Unified loss module supporting WGAN-GP, non-saturating Logistic loss, and Lazy R1 regularization (scaled by interval $k$).
- **Dependencies:** `torch`, `torch.nn`, `torch.amp`.
- **Callers & Dependents:** Called by `training/trainer.py`, `tests/test_losses.py`.
- **Status:** **IMPLEMENTED & VERIFIED.**

#### 3. `training/augmentation.py`
- **Purpose:** Data augmentation pipeline and Adaptive Discriminator Augmentation (ADA) controller.
- **Implemented Components:**
  - `build_face_transforms`: Deterministic torchvision pipeline for validation vs augmented pipeline for training (Horizontal Flip $p=0.5$, subtle rotation $\pm 5^\circ$, color jitter $10\%$).
  - `AdaptiveAugmentationController`: Monitors discriminator heuristic $r_t = \mathbb{E}[\text{sign}(D(x_{\text{real}}))]$ and dynamically modulates augmentation probability $p$.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by `tests/test_phase4_training.py`.

---

### C. Dataset Preparation & Pipeline (`data/`)

#### 1. `data/dataset.py`
- **Purpose:** Dataset loading, face alignment/cropping, normalization, and partition enforcement.
- **Implemented Components:**
  - `FaceTransform`: Standard CenterCrop(140) $\to$ Resize(64, 64) $\to$ Normalize to $[-1, 1]$.
  - `CelebADataset`: Loads images from `data/celeba/img_align_celeba`, strictly respecting `list_eval_partition.txt`.
  - `SyntheticFaceDataset`: Procedural geometric face generator for offline smoke tests.
- **Status:** **IMPLEMENTED & VERIFIED.** Disjoint partitions confirmed by `tests/test_dataset_split.py`.

---

### D. Two-Stage Inference & Enhancer Engine (`core/engine/` & `inference/`)

#### 1. `core/engine/enhancer.py`
- **Purpose:** Stage 2 Neural Face Restoration via GFPGAN 1.4 ONNX.
- **Implemented Components:**
  - `FaceEnhancer`: Thread-safe singleton wrapping ONNX Runtime session (`checkpoints/enhancer/gfpgan_1.4.onnx`).
  - Preprocesses $64\times 64$ PIL image to $512\times 512$ BGR tensor in $[-1, 1]$, executes ONNX session, and reconstructs $512\times 512$ RGB PIL image.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by `tests/test_enhancer.py`.

#### 2. `core/engine/inference_engine.py`
- **Purpose:** Thread-safe production inference engine.
- **Implemented Components:**
  - `InferenceEngine`: Loads official checkpoint, manages Raw vs EMA generators, supports noise scale scaling (e.g. 0.25 Smooth Mode), SLERP/LERP latent interpolation, and multi-scale style mixing.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by `tests/test_core_engine.py`.

#### 3. `inference/infer.py`
- **Purpose:** High-level Python API wrapper (`StyleForgeInference`) providing backward compatibility.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by `tests/test_inference.py`.

#### 4. `inference/app.py`
- **Purpose:** Gradio web application for interactive model exploration.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by `tests/test_inference.py`.

---

### E. Production Backend & Serving (`backend/`)

#### 1. `backend/main.py` & `backend/config.py`
- **Purpose:** FastAPI v2.0 application entry point, CORS middleware, lifespan startup/shutdown, and pydantic settings.
- **Status:** **IMPLEMENTED & VERIFIED.**

#### 2. `backend/api/v1/generate.py`
- **Purpose:** Generative REST endpoints (`/generate`, `/batch-generate`, `/interpolate`, `/mix-styles`).
- **Features:** Bounded GPU concurrency via `AsyncThreadSafeSemaphore`, authentication enforcement, and async background storage.
- **Status:** **IMPLEMENTED & VERIFIED.**

#### 3. `backend/security/auth.py` & `rate_limiter.py`
- **Purpose:** Constant-time API key verification (`secrets.compare_digest`) and in-memory rate limiting with stale IP cleanup.
- **Status:** **IMPLEMENTED & VERIFIED.**

---

### F. Quantitative Evaluation & Metrics (`evaluation/`)

#### 1. `evaluation/fid_kid.py`
- **Purpose:** Quantitative distribution distance evaluation.
- **Implemented Components:**
  - `PerceptualFeatureBank`: 4-layer convolutional feature extractor.
  - `calculate_frechet_distance`: SciPy matrix square root based Fréchet distance calculator.
  - `MetricsEvaluator`: Computes Inception-v3 FID/KID via TorchMetrics when weights are present, or falls back to `PerceptualFeatureBank`.
- **Status:** **IMPLEMENTED & VERIFIED.**

#### 2. `evaluation/diversity.py`
- **Purpose:** Mode collapse diagnostics, pairwise $L_1/L_2$ diversity, and $k$-NN Precision & Recall.
- **Implemented Components:** `compute_precision_recall` and `DiversityEvaluator`.
- **Status:** **IMPLEMENTED & VERIFIED.** Verified by `tests/test_metrics_precision_recall.py`.
