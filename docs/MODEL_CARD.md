# StyleForge Model Card — v2.0 Production Release

## Model Details

- **Model Name:** StyleForge Production Face Generator
- **Version:** v1.0.0 (Production Weights) / v2.0.0 (Platform Engine)
- **Model Type:** Generative Adversarial Network (StyleGAN-inspired)
- **Architecture:** 8-layer MLP Mapping Network ($f: \mathcal{Z} \to \mathcal{W}$) + Progressive Synthesis Network ($4\times 4 \to 64\times 64$) with AdaIN and per-channel Noise Injection.
- **Framework:** PyTorch 2.5.1 + CUDA 12.1
- **Official Checkpoint:** `checkpoints/styleforge_final_model.pt`
- **SHA-256 Checksum:** `ec8e4bf1e4f21f240f7793a1b3922a4510394831d0c39de521c4cd456c9eed64`
- **License:** MIT License

---

## Intended Use

- **Primary Application:** Research, real-time photographic face synthesis, latent manifold exploration, and style mixing experiments.
- **Target Distribution:** Human facial synthesis at native $64\times 64$ resolution.
- **Out-of-Scope:** Full-body synthesis, non-human objects, high-resolution photographic printing, deepfake manipulation, biometric identity spoofing.

---

## Training Dataset

- **Dataset:** CelebA (Large-scale CelebFaces Attributes Dataset)
- **Resolution:** $64 \times 64$ RGB pixels (Center-cropped and normalized to $[-1, 1]$)
- **Partition:** 4,912 training / 614 validation / 614 test images
- **Preprocessing:** Square center crop, bicubic interpolation downsample to $64\times 64$, normalized via $\mu=0.5, \sigma=0.5$.

---

## Master Empirical Benchmarks

Evaluated on $N=614$ authentic photographic CelebA validation partition using official TorchMetrics Inception-v3 pool3 ($2048$-dim) embeddings:

| Model / Condition | Weight State | Noise Scale | Inception-v3 FID ↓ | KID $\times 10^3$ ↓ | Laplacian Var (Sharpness) | HF PSD Ratio (Rel to Real) | Batch Pixel Var | Pairwise $L_1$ Div |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Real CelebA Validation** | **Photographic** | **N/A** | **0.00** | **0.0000** | $\mathbf{874.23 \pm 277.59}$ | $\mathbf{0.000046\text{ (1.00}\times)}$ | $\mathbf{0.2475}$ | **N/A** |
| Baseline C1 (v1.0.0 Default) | Raw | $1.00$ | $298.86$ | $0.3100$ | $4057.01 \pm 1012.35$ | $0.000090\text{ (1.95}\times)$ | $0.2909$ | $0.5862$ |
| **Ablation C2 (Best FID)** | **Raw** | **0.50** | $\mathbf{294.75}$ | $\mathbf{0.3104}$ | $1561.56 \pm 449.06$ | $0.000054\text{ (1.17}\times)$ | $0.2868$ | $0.5813$ |
| **Ablation C3 (Clean Parity Mode)** | **Raw** | **0.25** | $304.76$ | $0.3378$ | $\mathbf{922.43 \pm 341.26}$ | $\mathbf{0.000045\text{ (0.97}\times)}$ | $0.2858$ | $0.5801$ |
| Ablation C4 (Denoised) | Raw | $0.00$ | $311.89$ | $0.3621$ | $707.67 \pm 314.85$ | $0.000042\text{ (0.91}\times)$ | $0.2854$ | $0.5797$ |
| Baseline EMA | EMA | $1.00$ | $339.50$ | $0.4046$ | $2636.37 \pm 714.58$ | $0.000068\text{ (1.48}\times)$ | $0.3348$ | $0.6277$ |

---

## Known Scientific Limitations

1. **Resolution Scale:** The model outputs authentic $64\times 64$ pixel faces. The platform does not hallucinate higher-resolution detail.
2. **Inception Metric Properties:** Completely removing noise (`scale=0.0`) smooths skin visually, but Inception pool3 feature maps penalize the total absence of natural spatial micro-texture, causing FID to increase from $294.75$ to $311.89$.
3. **AdaIN Clamping Dynamics:** Forcing AdaIN scale parameters to $\ge 0.0$ at test time mutes $19.36\%$ of learned channels, decreasing pixel variance by $-19.3\%$ and pairwise $L_1$ diversity by $-10.2\%$. Clamping is therefore disabled.
