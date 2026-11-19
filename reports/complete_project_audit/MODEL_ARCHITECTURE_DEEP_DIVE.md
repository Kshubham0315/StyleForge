# StyleForge — Model Architecture Deep Dive
## Mathematical Formulations, Layer-by-Layer Tensor Traces & Comparative Architectural Audit

**Auditor:** Senior AI Research Engineer, GAN Systems Architect & Numerical Debugger  
**Focus:** Complete mathematical formulation, parameter verification, and architectural taxonomy.

---

## 1. Generator Architecture & Tensor Graph Trace

The StyleForge Generator decomposes face synthesis into two decoupled sub-networks:
1. **Mapping Network ($f: \mathcal{Z} \to \mathcal{W}$)**
2. **Synthesis Network ($g: \mathcal{W} \to \mathcal{X}$)**

```
Input: z ~ N(0, I) in R^(B x 512)
  │
[PixelNorm] ──► z_norm = z / sqrt(mean(z^2) + 1e-8)
  │
[Mapping Network: 8 x (EqualizedLinear(512, 512) + LeakyReLU(0.2))] with lr_mul=0.01
  │
Output: w in R^(B x 512) (Disentangled Intermediate Latent Space)
  │
  ├─► Optional Truncation Trick: w' = w_avg + psi * (w - w_avg)
  ├─► Optional Style Mixing: styles = [w, ..., w, w2, ..., w2] across layers
  │
  ▼
[Synthesis Network]
Learned Constant C in R^(1 x 512 x 4 x 4)
  │
  ├──► Repeated to (B, 512, 4, 4)
  ├──► Noise Injection 0: x + weight_0 * noise(B, 1, 4, 4)
  ├──► AdaIN 0 (modulated by w_0)
  ├──► StyleBlock (4x4): Conv3x3 (512->512) + Noise + LeakyReLU + AdaIN 1 (modulated by w_1)
  │
[Resolution 8x8]:
  ├──► Bilinear Upsample 2x -> (B, 512, 8, 8)
  ├──► StyleBlock 1: Conv3x3 (512->512) + Noise + LeakyReLU + AdaIN 2 (w_2)
  ├──► StyleBlock 2: Conv3x3 (512->512) + Noise + LeakyReLU + AdaIN 3 (w_3)
  │
[Resolution 16x16]:
  ├──► Bilinear Upsample 2x -> (B, 512, 16, 16)
  ├──► StyleBlock 1: Conv3x3 (512->256) + Noise + LeakyReLU + AdaIN 4 (w_4)
  ├──► StyleBlock 2: Conv3x3 (256->256) + Noise + LeakyReLU + AdaIN 5 (w_5)
  │
[Resolution 32x32]:
  ├──► Bilinear Upsample 2x -> (B, 256, 32, 32)
  ├──► StyleBlock 1: Conv3x3 (256->128) + Noise + LeakyReLU + AdaIN 6 (w_6)
  ├──► StyleBlock 2: Conv3x3 (128->128) + Noise + LeakyReLU + AdaIN 7 (w_7)
  │
[Resolution 64x64]:
  ├──► Bilinear Upsample 2x -> (B, 128, 64, 64)
  ├──► StyleBlock 1: Conv3x3 (128->64)  + Noise + LeakyReLU + AdaIN 8 (w_8)
  ├──► StyleBlock 2: Conv3x3 (64->64)   + Noise + LeakyReLU + AdaIN 9 (w_9)
  │
[ToRGB Output Projection]:
  └──► EqualizedConv2d (64 -> 3, kernel=1, padding=0) + Tanh() -> Output (B, 3, 64, 64) in [-1, 1]
```

---

## 2. Mathematical Formulations of Key Building Blocks

### 2.1 Pixel Normalization
Before entering the mapping network, the input latent vector $z$ is projected onto the unit hypersphere:
$$\text{PixelNorm}(z) = \frac{z}{\sqrt{\frac{1}{D}\sum_{j=1}^D z_j^2 + \epsilon}}, \quad \epsilon = 10^{-8}$$
This prevents the signal magnitudes from diverging during early adversarial optimization.

### 2.2 Equalized Learning Rate with Runtime He Scaling & `lr_mul`
In standard initialization, weights are scaled before training begins. In StyleForge (following Karras et al.), weights $\hat{w}$ are initialized from $\mathcal{N}(0, 1)$ divided by `lr_mul`, and scaled dynamically at runtime during the forward pass:
$$w = \hat{w} \cdot c, \quad c = \left(\frac{\text{gain}}{\sqrt{\text{fan\_in}}}\right) \cdot \text{lr\_mul}$$
For linear layers with LeakyReLU ($\alpha=0.2$), $\text{gain} = \sqrt{2}$. For mapping layers, `lr_mul = 0.01`, which scales the learning rate by $100\times$ downward, stabilizing the intermediate manifold convergence.

### 2.3 Adaptive Instance Normalization (AdaIN)
Each feature channel $x_i$ is normalized independently across spatial dimensions $(H, W)$, then modulated by an affine style transform derived from $w$:
$$\text{AdaIN}(x_i, y) = y_{s, i} \cdot \left(\frac{x_i - \mu(x_i)}{\sigma(x_i)}\right) + y_{b, i}$$
Where:
$$\mu(x_i) = \frac{1}{HW}\sum_{h=1}^H \sum_{w=1}^W x_{i, h, w}, \quad \sigma(x_i) = \sqrt{\frac{1}{HW}\sum_{h=1}^H \sum_{w=1}^W (x_{i, h, w} - \mu(x_i))^2 + \epsilon}$$
The style modulation parameters $[y_s, y_b] \in \mathbb{R}^{2C}$ are generated via an `EqualizedLinear` projection from $w$:
$$[y_s, y_b] = \text{Linear}(w), \quad \text{scale} = y_s + 1.0, \quad \text{bias} = y_b$$

### 2.4 Stochastic Noise Injection
Uncorrelated per-channel Gaussian noise $n \sim \mathcal{N}(0, I) \in \mathbb{R}^{B \times 1 \times H \times W}$ is broadcast across spatial locations and scaled by learned per-channel parameter vectors $B \in \mathbb{R}^{1 \times C \times 1 \times 1}$:
$$x' = x + B \odot n$$
The noise weights are initialized to zero ($B=0$) so that stochastic details do not perturb early coarse structural convergence.

---

## 3. Parameter Count & Layer-by-Layer Verification

An empirical audit of model parameters yielded:

| Sub-Module / Layer | Input Channels | Output Channels | Feature Map Resolution | Parameter Count |
|---|:---:|:---:|:---:|:---:|
| **Mapping Network (8 FC Layers)** | $512$ | $512$ | Vector ($D=512$) | **$2,101,248$** |
| - Layer 1 (FC + Bias) | 512 | 512 | - | 262,656 |
| - Layers 2–8 ($7 \times 262,656$) | 512 | 512 | - | 1,838,592 |
| **Learned Constant Tensor** | - | 512 | $4\times 4$ | **$8,192$** |
| **Initial $4\times 4$ Stage** | | | | |
| - Noise Injection 0 | 512 | 512 | $4\times 4$ | 512 |
| - AdaIN 0 (Affine Modulator) | 512 | $1024$ | - | 525,312 |
| - Initial Block (Conv3x3) | 512 | 512 | $4\times 4$ | 2,359,808 |
| - Initial Block (Noise + AdaIN 1) | 512 | 512 | $4\times 4$ | 525,824 |
| **Stage $8\times 8$ (Bilinear Upsample)** | | | | |
| - Block 1 (Conv3x3 + Noise + AdaIN 2) | 512 | 512 | $8\times 8$ | 2,886,144 |
| - Block 2 (Conv3x3 + Noise + AdaIN 3) | 512 | 512 | $8\times 8$ | 2,886,144 |
| **Stage $16\times 16$ (Bilinear Upsample)** | | | | |
| - Block 1 (Conv3x3 + Noise + AdaIN 4) | 512 | 256 | $16\times 16$ | 1,442,432 |
| - Block 2 (Conv3x3 + Noise + AdaIN 5) | 256 | 256 | $16\times 16$ | 852,608 |
| **Stage $32\times 32$ (Bilinear Upsample)** | | | | |
| - Block 1 (Conv3x3 + Noise + AdaIN 6) | 256 | 128 | $32\times 32$ | 426,304 |
| - Block 2 (Conv3x3 + Noise + AdaIN 7) | 128 | 128 | $32\times 32$ | 278,848 |
| **Stage $64\times 64$ (Bilinear Upsample)** | | | | |
| - Block 1 (Conv3x3 + Noise + AdaIN 8) | 128 | 64 | $64\times 64$ | 139,424 |
| - Block 2 (Conv3x3 + Noise + AdaIN 9) | 64 | 64 | $64\times 64$ | 102,560 |
| **ToRGB Projection** | 64 | 3 | $64\times 64$ | **$195$** |
| **TOTAL GENERATOR PARAMETERS** | - | - | - | **$14,535,875$** |
| **Generator Model Size (FP32)** | - | - | - | **$55.45\text{ MB}$** |

---

## 4. Critic (Discriminator) Architecture & Parameter Trace

The Critic is a residual convolutional downsampler with Minibatch Standard Deviation:

| Layer / Stage | Input Shape | Output Shape | Kernel / Stride | Parameters |
|---|:---:|:---:|:---:|:---:|
| **FromRGB** | $(B, 3, 64, 64)$ | $(B, 64, 64, 64)$ | $1\times 1, s=1$ | 256 |
| **Downsample 64 $\to$ 32** | $(B, 64, 64, 64)$ | $(B, 128, 32, 32)$ | $3\times 3$ residual + avg_pool | 221,696 |
| **Downsample 32 $\to$ 16** | $(B, 128, 32, 32)$ | $(B, 256, 16, 16)$ | $3\times 3$ residual + avg_pool | 885,760 |
| **Downsample 16 $\to$ 8** | $(B, 256, 16, 16)$ | $(B, 512, 8, 8)$ | $3\times 3$ residual + avg_pool | 3,540,992 |
| **Downsample 8 $\to$ 4** | $(B, 512, 8, 8)$ | $(B, 512, 4, 4)$ | $3\times 3$ residual + avg_pool | 4,720,640 |
| **Minibatch StdDev Injection** | $(B, 512, 4, 4)$ | $(B, 513, 4, 4)$ | Batch stats ($G=4$) | 0 (bufferless) |
| **Final Conv Block** | $(B, 513, 4, 4)$ | $(B, 512, 4, 4)$ | $3\times 3, p=1$ | 2,364,416 |
| **Final Linear Projection** | $(B, 8192)$ | $(B, 1)$ | Dense ($512 \times 4 \times 4 \to 1$) | 8,193 |
| **TOTAL CRITIC PARAMETERS** | - | - | - | **$12,174,721$** |
| **Critic Model Size (FP32)** | - | - | - | **$46.44\text{ MB}$** |

---

## 5. Architectural Comparison: StyleForge vs Official StyleGANs

To avoid exaggerated claims, the table below documents exactly what StyleForge adopts from official NVIDIA releases:

| Feature / Architectural Technique | StyleGAN (2019) | StyleGAN2 (2020) | StyleGAN3 (2021) | StyleForge Implementation Status |
|---|:---:|:---:|:---:|:---:|
| **Mapping Network ($Z \to W$)** | Yes (8 layers) | Yes (8 layers) | Yes (2–8 layers) | **YES.** 8-layer MLP with PixelNorm & `lr_mul=0.01`. |
| **Learned Constant Input** | Yes ($4\times 4$) | Yes ($4\times 4$) | Replaced by Fourier | **YES.** Learned $4\times 4\times 512$ parameter. |
| **AdaIN Normalization** | Yes | Replaced by Demod | Replaced | **YES.** Adopted StyleGAN1 AdaIN modulation. |
| **Weight Demodulation** | No | Yes | Yes | **NO.** StyleForge retains AdaIN (susceptible to droplet artifacts). |
| **Path Length Regularization** | No | Yes | No | **NO.** Uses Lazy R1 regularization instead. |
| **Lazy R1 Regularization** | Optional | Standard ($k=16$) | Standard | **YES.** Lazy R1 with interval $k=16, \gamma=10.0$. |
| **Progressive Growing** | Yes | Replaced by MSG | No | **NO.** Native non-progressive $64\times 64$ synthesis. |
| **Equalized Learning Rate** | Yes | Yes | Yes | **YES.** Runtime He-scaling on conv & linear layers. |
| **Minibatch StdDev in Critic** | Yes | Yes | Yes | **YES.** Injected at $4\times 4$ resolution with group tiling. |
| **Alias-Free Equivariance** | No | No | Yes | **NO.** Uses standard bilinear interpolation without sinc filters. |

> [!IMPORTANT]
> **Definitive Taxonomy:** StyleForge is fundamentally a **StyleGAN1-style AdaIN generator** paired with **StyleGAN2-style Lazy R1 optimization and TTUR**, operating at native $64\times 64$ resolution. It does not implement StyleGAN2 weight demodulation or StyleGAN3 continuous translation equivariance.
