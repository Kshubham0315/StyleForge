# StyleForge Research Limitations & Next Experiments

## 1. Experimental Limitations

### 1.1 Dataset Scale
- **Current Dataset**: 500 images ($128\times 128$) from the CelebA dataset partitioned into 450 training images and 50 validation images.
- **Limitation**: While zero-leakage deterministic partitioning was mathematically verified, 450 training images cannot capture the full combinatorial diversity of the human face population. Genuine CelebA scale comprises 202,599 images.
- **Impact on FID**: Calculating Inception-v3 FID against only 50 validation images introduces high finite-sample bias. Standard evaluation requires at least 2,048 (ideally 10,000 to 50,000) validation images.

### 1.2 Training Budget & Convergence
- **Current Step Count**: 600 generator updates ($9,600$ image presentations).
- **Limitation**: GAN training on face benchmarks typically requires $50,000$ to $150,000$ steps (several million image presentations). At 600 steps, models exhibit formative facial contours, color blocks, and structural symmetry, but high-frequency textures (hair strands, photorealistic iris patterns) are in early evolution.

### 1.3 Hardware Boundaries
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU with 6.00 GB VRAM.
- **Memory Ceiling**: StyleForge AdaIN consumes ~3.5 GB VRAM at batch size 16 at $64\times 64$ resolution. Scaling resolution to $128\times 128$ or $256\times 256$ in FP32 would exceed the 6GB VRAM budget without gradient accumulation or activation checkpointing.

---

## 2. Recommended Next Research Experiments

1. **Dataset Scaling**: Expand the training set to 10,000+ CelebA images using the automated downloader (`scripts/download_celeba.py`).
2. **Extended Training**: Scale training budget to 10,000 - 25,000 steps with checkpoint resume.
3. **Progressive Growing / StyleGAN2 Skip Generator**: Implement skip-connection synthesis architectures to scale resolution to $128\times 128$ and $256\times 256$ within the 6GB VRAM limit.
4. **Path Length Regularization**: Incorporate StyleGAN2 path length regularization ($L_{\text{PPL}}$) to encourage smooth latent space geometry and preserve isometric mappings.
