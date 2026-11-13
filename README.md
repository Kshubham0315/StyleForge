# StyleForge

A StyleGAN-inspired face generation and latent exploration system built with PyTorch, focused on GAN training stability, style-based generation, latent space disentanglement, and quantitative evaluation.

---

## Highlights

- **Dual Generator Architectures**: Classical convolutional WGAN-GP baseline and full StyleGAN-style generator (8-layer Mapping Network $f: \mathcal{Z} \to \mathcal{W}$ + learned constant input + AdaIN blocks + per-channel stochastic noise injection).
- **Stabilized Training Pipeline**: 2-sided Gradient Penalty (WGAN-GP), R1 gradient regularization, Minibatch Standard Deviation to prevent mode collapse, Exponential Moving Average (EMA) of generator weights, and mixed-precision (FP16 AMP) acceleration.
- **Architectural Ablations**: Toggleable SAGAN Self-Attention blocks and Spectral Normalization in Critic for controlled experiments.
- **Latent Space Exploration**: Spherical Linear Interpolation (SLERP) in $\mathcal{Z}$, Linear Interpolation (LERP) in $\mathcal{W}$, style mixing across coarse/middle/fine layers, and Truncation Trick ($\psi$).
- **Comprehensive Evaluation**: Automated FID, KID, pairwise pixel diversity (L1/L2), and batch variance metrics with offline fallback support.
- **Zero-Setup Quickstart**: Built-in synthetic procedural face generator for immediate training and pipeline verification without waiting for dataset downloads.

---

## Project Structure

```text
StyleForge/
├── configs/
│   ├── baseline_wgan_gp.yaml        # Reference WGAN-GP baseline
│   ├── stylegan_adain.yaml          # Style-based generator with AdaIN & mapping network
│   ├── ablation_spectral_norm.yaml  # Spectral normalization critic ablation
│   ├── ablation_self_attention.yaml # Self-attention SAGAN ablation
│   └── quick_run.yaml               # Rapid smoke-test configuration
├── data/
│   ├── __init__.py
│   └── dataset.py                   # CelebA loader, center crop transform & synthetic face dataset
├── models/
│   ├── __init__.py
│   ├── common.py                    # AdaIN, PixelNorm, MinibatchStdDev, SelfAttention, Equalized layers
│   ├── wgan_baseline.py             # Classical DCGAN/WGAN-GP generator & critic
│   ├── style_generator.py           # MappingNetwork, SynthesisNetwork, and StyleGenerator
│   └── critic.py                    # Multi-scale residual critic with MinibatchStdDev & SN
├── training/
│   ├── __init__.py
│   ├── losses.py                    # WGAN loss, 2-sided gradient penalty, R1 regularization
│   └── trainer.py                   # GANTrainer with EMA, AMP, metrics logging & checkpointing
├── evaluation/
│   ├── __init__.py
│   ├── fid_kid.py                   # Fréchet Inception Distance & KID evaluator
│   ├── latent_explorer.py           # SLERP, LERP, truncation sweep & style mixing matrix
│   └── diversity.py                 # Mode collapse & pairwise diversity diagnostics
├── scripts/
│   ├── train.py                     # Training CLI
│   ├── generate.py                  # Image generation & latent exploration CLI
│   ├── evaluate.py                  # Quantitative metrics CLI
│   └── download_celeba.py           # CelebA setup helper & mock dataset generator
├── tests/
│   ├── test_models.py               # Unit tests for models & layers
│   ├── test_losses.py               # Unit tests for losses & GP/R1 penalties
│   └── test_latent_explorer.py      # Unit tests for SLERP, LERP & style mixing
├── notebooks/
│   └── styleforge_walkthrough.ipynb # Interactive exploration walkthrough
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Verify your setup with the unit test suite:

```bash
python -m pytest tests/
```

---

## Training

### Instant Smoke Test (Zero Download)

Verify training on your GPU immediately using the built-in synthetic dataset:

```bash
python scripts/train.py --config configs/quick_run.yaml --steps 50
```

### CelebA Face Training

1. Download `img_align_celeba.zip` from [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or Kaggle and extract into:
   ```text
   StyleForge/data/celeba/img_align_celeba/
   ```
   *(Or generate a local mock dataset with `python scripts/download_celeba.py --mock --count 1000`)*

2. Train the **StyleGAN AdaIN** architecture:
   ```bash
   python scripts/train.py --config configs/stylegan_adain.yaml
   ```

3. Train the **WGAN-GP Baseline**:
   ```bash
   python scripts/train.py --config configs/baseline_wgan_gp.yaml
   ```

4. Resume from a saved checkpoint:
   ```bash
   python scripts/train.py --config configs/stylegan_adain.yaml --resume experiments/stylegan_adain/checkpoints/checkpoint_latest.pt
   ```

---

## Generation & Latent Space Manipulation

Use `scripts/generate.py` to create images and explore the latent space:

### 1. Sample Grid Generation
```bash
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode samples --n 16 --psi 0.7
```

### 2. Spherical Latent Interpolation (SLERP)
```bash
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode interpolate --steps 10
```

### 3. Truncation Trick Sweep ($\psi \in [0.0, 1.0]$)
```bash
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode truncation
```

### 4. Multilevel Style Mixing Matrix
```bash
python scripts/generate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --mode style_mix
```

---

## Quantitative Evaluation

Evaluate image quality, distribution distance, and diversity metrics:

```bash
python scripts/evaluate.py --checkpoint experiments/stylegan_adain/checkpoints/checkpoint_latest.pt --num_samples 200 --out_json experiments/eval_report.json
```

Outputs:
- **FID / FFD**: Distribution distance between real face data and generated images
- **Pairwise L1 / L2 Distance**: Sample variability across random latent codes
- **Batch Pixel Variance**: Variance across spatial dimensions to detect mode collapse
- **Diversity Score**: Standardized diversity metric

---

## Interactive Notebook

Open the interactive Jupyter walkthrough:

```bash
jupyter notebook notebooks/styleforge_walkthrough.ipynb
```

---

## Architectural Details

```text
       Latent Code z ~ N(0, I) [512-dim]
                   │
                   ▼ [PixelNorm]
         ┌───────────────────┐
         │  Mapping Network  │ 8x EqualizedLinear + LeakyReLU
         │   f: Z -> W       │
         └─────────┬─────────┘
                   │
                   ▼ Intermediate Style Vector w [512-dim]
                   │
        Learned 4x4 Constant Input
                   │
                   ▼
         ┌───────────────────┐
         │    AdaIN Block    │ <--- Modulate with w (scale & bias)
         │  + Noise Inject   │ <--- Learned stochastic Gaussian noise
         └─────────┬─────────┘
                   │
             Upsample (2x)
                   │
         ┌───────────────────┐
         │    AdaIN Block    │ <--- Modulate with w
         └─────────┬─────────┘
                   │
                 ... (Up to 64x64 / 128x128)
                   │
                   ▼
              ToRGB (Tanh) -> Generated Face Image [-1, 1]
```

---

## License

MIT License.
