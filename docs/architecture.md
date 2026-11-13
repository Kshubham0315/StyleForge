# StyleForge Architectural Specification

## 1. Overview & Theoretical Framework

StyleForge is a research-grade generative adversarial network implementing modern style-based synthesis principles inspired by StyleGAN (Karras et al., 2019) alongside a Wasserstein GAN with Gradient Penalty (WGAN-GP) baseline.

The architecture decouples image synthesis into two distinct spaces:
1. **Latent Space $\mathcal{Z} \subset \mathbb{R}^{d_z}$**: An unconstrained standard isotropic Gaussian distribution $\mathcal{N}(0, I)$.
2. **Intermediate Latent Space $\mathcal{W} \subset \mathbb{R}^{d_w}$**: A learned manifold where factors of variation are linearly disentangled.

```
       z ~ N(0, I)
            │
            ▼
   ┌─────────────────┐
   │ Mapping Network │  (8-layer MLP, PixelNorm, lr_mul=0.01)
   └─────────────────┘
            │
            ▼  w ∈ W
  Learned   │
  Constant  │ (Style Modulation via AdaIN)
  (4x4x512) ├──────────────► ResBlock 4x4  ──► Noise Injection
            ├──────────────► ResBlock 8x8  ──► Noise Injection
            ├──────────────► ResBlock 16x16 ──► [Self-Attention]
            ├──────────────► ResBlock 32x32 ──► Noise Injection
            └──────────────► ResBlock 64x64 ──► ToRGB (Tanh) ──► Image [-1, 1]
```

---

## 2. Mathematical Components

### 2.1 Mapping Network ($f: \mathcal{Z} \to \mathcal{W}$)
Given an input vector $z \sim \mathcal{N}(0, I) \in \mathbb{R}^{512}$, we first apply **PixelNorm** across the channel dimension to project it onto a unit hypersphere:
$$\text{PixelNorm}(z) = \frac{z}{\sqrt{\frac{1}{d_z} \sum_{j=1}^{d_z} z_j^2 + \epsilon}}$$

The normalized vector passes through an 8-layer fully-connected multi-layer perceptron (MLP). Each linear layer uses **Equalized Learning Rate** with LeakyReLU activation ($\alpha = 0.2$):
$$w_0 = \text{PixelNorm}(z)$$
$$w_{l} = \text{LeakyReLU}\left(\hat{W}_l w_{l-1} + \hat{b}_l\right), \quad l \in \{1, \dots, 8\}$$

#### Equalized Learning Rate & `lr_mul` Scaling
Rather than initializing weights according to He/Kaiming initialization and optimizing them directly, weights are initialized as:
$$W_l \sim \mathcal{N}\left(0, \left(\frac{1}{\text{lr\_mul}}\right)^2\right)$$
During runtime forward propagation, the effective weight is scaled dynamically:
$$\hat{W}_l = W_l \cdot \left(\frac{\sqrt{2}}{\sqrt{\text{fan\_in}_l}} \cdot \text{lr\_mul}\right)$$
This ensures:
1. The forward signal variance is strictly unit-preserved ($\text{Var}(\hat{W}_l x) \approx \text{Var}(x)$), preventing forward signal collapse across deep mapping stacks.
2. In Adam optimization, where step size is normalized by gradient magnitude ($\Delta W \approx \alpha$), the effective update on $\hat{W}_l$ scales by $\alpha \cdot \text{lr\_mul}$. For the mapping network, $\text{lr\_mul} = 0.01$, stabilizing early latent space divergence.

---

### 2.2 Learned Constant Spatial Input
Traditional generators (DCGAN, WGAN) feed latent vectors directly into spatial transpose convolutions, forcing the spatial geometry to entangle with global style. StyleForge discards spatial latent inputs in favor of a learned tensor parameter:
$$C \in \mathbb{R}^{1 \times 512 \times 4 \times 4}, \quad C \sim \mathcal{N}(0, I)$$
This tensor is broadcast across batch size $N$ and represents the canonical face template, while all identity, lighting, color, and geometric features are modulated externally via AdaIN.

---

### 2.3 Adaptive Instance Normalization (AdaIN)
Each convolutional feature map $x_i \in \mathbb{R}^{B \times C \times H \times W}$ is normalized independently per channel and modulated by a style vector $w \in \mathbb{R}^{d_w}$:
$$\mu(x)_c = \frac{1}{HW} \sum_{h=1}^H \sum_{w'=1}^W x_{c, h, w'}$$
$$\sigma(x)_c = \sqrt{\frac{1}{HW} \sum_{h=1}^H \sum_{w'=1}^W (x_{c, h, w'} - \mu(x)_c)^2 + \epsilon}$$
$$\text{AdaIN}(x, y) = y_{s, c} \left(\frac{x_c - \mu(x)_c}{\sigma(x)_c}\right) + y_{b, c}$$
where affine style transformation parameters $[y_s, y_b] = \text{EqualizedLinear}(w)$ produce channel-wise scale $y_s$ and bias $y_b$.

---

### 2.4 Stochastic Noise Injection
To model localized stochastic variation (hair texture, freckles, skin pores) without entangling the global style vector, dedicated spatial noise maps $B \in \mathbb{R}^{N \times 1 \times H \times W} \sim \mathcal{N}(0, I)$ are scaled by learnable per-channel scaling factors $P \in \mathbb{R}^{1 \times C \times 1 \times 1}$ and added prior to AdaIN:
$$x' = x + P \odot B$$

---

### 2.5 Minibatch Standard Deviation
Mode collapse occurs when the critic cannot evaluate diversity across samples within a batch. Minibatch standard deviation computes dispersion statistics across mini-batches:
1. Divide batch into sub-groups of size $G \le N$.
2. Compute variance across group samples for every channel and spatial location:
   $$\sigma_{g, c, h, w} = \sqrt{\frac{1}{G}\sum_{i=1}^G (x_{i, c, h, w} - \mu_{g, c, h, w})^2 + \epsilon}$$
3. Average across all channels and spatial dimensions to produce a scalar per group:
   $$S_g = \frac{1}{C \cdot H \cdot W} \sum_{c, h, w} \sigma_{g, c, h, w}$$
4. Replicate $S_g$ across spatial dimensions $H \times W$ and concatenate as an additional feature channel before the final critic classification block.

---

### 2.6 Multi-Scale Residual Critic
The critic evaluates whether images $x \in [-1, 1]$ are authentic CelebA faces or generated artifacts:
- Downsampling via strided $3 \times 3$ convolutions and average pooling.
- Residual skip connections with $1 \times 1$ downsampling projections.
- Minibatch standard deviation at the penultimate resolution.
- Final scalar output without sigmoid (unbounded Wasserstein score $\mathbb{R}$).
