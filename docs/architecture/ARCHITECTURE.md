# StyleForge System Architecture

This document specifies the complete system architecture, training loops, regularization pipelines, and inference interfaces of the **StyleForge** generative engine.

---

## 1. End-to-End System Pipeline

```mermaid
flowchart TD
    subgraph DataPipeline ["Dataset & Preprocessing"]
        CelebA["CelebA Photographic Portraits (178x218)"] --> Crop["Center Crop (178x178)"]
        Crop --> Resize["Bicubic Anti-Aliased Resize (64x64)"]
        Resize --> Norm["Normalize to [-1, 1]"]
        Norm --> Loader["Deterministic DataLoader (Train/Val/Test)"]
    end

    subgraph GeneratorArchitecture ["StyleForge Generator Architecture"]
        Z["Latent Space Z ~ N(0, I)\n(512-dim)"] --> PN["PixelNorm"]
        PN --> MapNet["Mapping Network f : Z -> W\n(8-Layer MLP, EqualizedLinear, lr_mul=0.01)"]
        MapNet --> W["Intermediate Style Space W\n(512-dim)"]
        
        Const["Learned Constant Tensor\n(4x4x512)"] --> Synth["Synthesis Network g\nProgressive AdaIN Modulations\n(4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64)"]
        W --> Synth
        Noise["Per-Channel Stochastic Noise"] --> Synth
        Synth --> ToRGB["EqualizedConv2d (1x1) + Tanh"]
        ToRGB --> FakeImg["Synthesized Face Image\n(64x64x3)"]
    end

    subgraph CriticArchitecture ["Multi-Scale Residual Critic"]
        RealOrFake["Input Image (Real or Fake)\n(64x64x3)"] --> FromRGB["FromRGB Conv"]
        FromRGB --> DownBlocks["Residual Downsampling Blocks\n(64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4)"]
        DownBlocks --> MiniBatch["Minibatch Standard Deviation\n(Mode Collapse Prevention)"]
        MiniBatch --> Score["Final Linear Projection -> Unbounded Critic Score D(x)"]
    end

    subgraph AdversarialLoop ["Adversarial Optimization & Regularization"]
        FakeImg --> RealOrFake
        Loader --> RealOrFake
        Score --> Loss["Adversarial Loss: WGAN-GP / Logistic + R1"]
        Loss --> GP["Gradient Penalty: ||grad D(x)||_2 - 1 or ||grad D(x)||^2"]
        GP --> Backprop["Pure FP32 Backpropagation & Adam Updates"]
        Backprop --> EMA["Dynamic Warmup EMA Shadow Model\nbeta(t) = min(beta, (1+t)/(10+t))"]
    end

    subgraph InferenceSystem ["Inference & Production Demo"]
        EMA --> ProdWeights["checkpoints/styleforge_final_model.pt\n(110.95 MB stripped production weights)"]
        ProdWeights --> API["inference.infer.StyleForgeInference"]
        API --> GradioApp["Gradio Interactive Web Studio\n(Face Synthesis, Latent Walk, Style Mixing)"]
    end
```

---

## 2. Style Modulation: Adaptive Instance Normalization (AdaIN)

At each resolution layer $l$, feature maps $x \in \mathbb{R}^{B \times C \times H \times W}$ are normalized across spatial dimensions $(H, W)$ and modulated by learned affine transformations of the intermediate style vector $w \in \mathcal{W}$:

$$\mu(x_c) = \frac{1}{HW} \sum_{h=1}^H \sum_{w=1}^W x_{c, h, w}$$

$$\sigma(x_c) = \sqrt{\frac{1}{HW} \sum_{h=1}^H \sum_{w=1}^W (x_{c, h, w} - \mu(x_c))^2 + \epsilon}$$

$$\text{AdaIN}(x, w) = y_{s, c}(w) \cdot \left(\frac{x_c - \mu(x_c)}{\sigma(x_c)}\right) + y_{b, c}(w)$$

where $(y_{s, c}, y_{b, c}) = \text{EqualizedLinear}(w)$.

---

## 3. Dynamic Warm-Up EMA Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant G as Active Generator θ
    participant EMA as EMA Shadow Weights
    participant Disc as Critic D

    Note over EMA: Initialize EMA = θ_0, updates = 0
    loop Every Training Step t
        G->>Disc: Synthesize fake images G(z)
        Disc->>G: Compute adversarial loss & backprop
        Note over G: Update θ_t via Adam
        Note over EMA: updates += 1
        Note over EMA: Compute beta(t) = min(beta, (1 + t)/(10 + t))
        EMA->>EMA: EMA_t = beta(t) * EMA_{t-1} + (1 - beta(t)) * θ_t
    end
    Note over EMA: Export final production model (Zero parameter lag)
```

---

## 4. Component Parameter Breakdown

| Component | Layer Count | Input Dimension | Output Dimension | Trainable Parameters |
|---|:---:|:---:|:---:|---:|
| **Mapping Network ($f$)** | 8 | $512$ (Gaussian $z$) | $512$ (Disentangled $w$) | $2,098,176$ |
| **Synthesis Network ($g$)** | 10 blocks | $4\times 4\times 512$ | $64\times 64\times 3$ | $12,437,699$ |
| **Total Generator ($G$)** | — | **512** | **$64\times 64\times 3$** | **$14,535,875$** |
| **StyleCritic ($D$)** | 5 residual stages | $64\times 64\times 3$ | Scalar $1$ | **$12,174,721$** |
| **Total System** | — | — | — | **$26,710,596$** |
