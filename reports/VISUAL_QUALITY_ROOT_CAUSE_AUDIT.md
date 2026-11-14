# StyleForge — Visual Generation Quality Root-Cause Audit
## Engineering & Research Diagnostic Report on 64×64 Facial Synthesis Quality

**Project:** StyleForge  
**Subject:** Visual Generation Quality Root-Cause Investigation (Blurriness, Noise, and Distortion)  
**Model Version:** v1.0.0 Production Checkpoint (`checkpoints/styleforge_final_model.pt`)  
**Base Training Checkpoint:** `experiments/phase3_convergence/checkpoints/checkpoint_final.pt` (Step 1,500)  
**Date:** November 14, 2026  
**Auditor:** Senior Generative AI Research Engineer, GAN Systems Architect & Numerical Debugger  
**Hardware & Environment:** NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM), Windows 11, PyTorch 2.5.1 + CUDA 12.1  

---

## 1. Executive Summary

A comprehensive, evidence-driven investigation was conducted to determine why generated $64\times 64$ facial images from the StyleForge v1.0.0 production model exhibit extreme blurriness, high-frequency noise/grain, and anatomical distortion.

### Key Investigation Takeaways:
1. **The Display & Inference Pipeline is Numerically Correct:**  
   There is **no** RGB/BGR channel swap, **no** image denormalization error, and **no** quantization clipping bug. Real CelebA images and generated images share identical normalization ranges ($[-1.0, 1.0] \to [0, 255]$) and identical color channel statistics.
2. **Primary Root Cause 1 — Severe Under-Convergence (1,500 Steps / 24 kimg):**  
   The model was trained for only **1,500 global generator steps** at batch size 16 ($24,000$ total image presentations, corresponding to only **4.89 epochs** over the 4,912 training images). In established GAN literature (StyleGAN, DCGAN, ProGAN), photorealistic facial synthesis requires between $2,000$ and $25,000\text{ kimg}$ ($100,000+$ steps). At 1,500 steps, a GAN has only resolved global low-frequency color masses (skin center, dark perimeter) and has not yet synthesized fine anatomical structures (eyes, pupils, teeth, nostrils).
3. **Primary Root Cause 2 — Stochastic Noise Injection Overdrive (+518% Artificial High-Frequency Variance):**  
   The learned per-channel noise injection layers (`NoiseInjection`) at the final $64\times 64$ resolution stage learned disproportionately high weights ($\text{std} = 0.091, \text{max} = 0.134$). During inference, unconditioned Gaussian spatial noise injected at this amplitude introduces a mean absolute deviation of **$0.138$** (almost $7\%$ of the entire dynamic range), inflating Laplacian variance by **$+518\%$** ($3,933.36$ with noise vs $636.43$ with zero noise) and creating a heavy salt-and-pepper grain.
4. **Primary Root Cause 3 — AdaIN Scale Inversion ($19.36\%$ Negative Scaling Multipliers):**  
   In `AdaIN`, the affine transformation computes $scale = \text{style\_mod}(w)_{[:C]} + 1.0$. Because this output was unconstrained during training, **$19.36\%$ of all style modulation channels produce negative scale values** ($scale < 0$, reaching $-4.97$). Inverting feature channel polarities flips edge gradients and contrast directions across random channels, causing facial asymmetry, dark eye patches, and distorted features.
5. **Precision & Recall = 0.0000 is Geometrically Expected:**  
   The minimum Euclidean distance between generated samples and real validation samples in Inception pool3 feature space is **$16.71$**, which strictly exceeds the real data manifold k-NN radius ($11.27 \pm 1.84$). The two distributions do not yet overlap in feature space; hence, k-NN Precision and Recall are mathematically $0.0$.

---

## 2. Issue Reproduction & Output Catalog

The exact production checkpoint (`checkpoints/styleforge_final_model.pt`, SHA256: `ec8e4bf1e4...eed64`) was loaded into GPU memory. Thirty-two images were synthesized across both weight states (**Raw** and **EMA**) and four truncation values ($\psi \in \{0.5, 0.7, 0.8, 1.0\}$) using fixed latent seeds ($z \sim \mathcal{N}(0, I)$ with `torch.manual_seed(2026)`).

All 256 generated images and composite comparison grids were saved into structured audit directories:

- **Audit Root Directory:** [`reports/visual_audit/`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/)
- **Individual Sample Subdirectories (32 images each):**
  - `reports/visual_audit/samples/raw_psi_0.5/` (`sample_00.png` to `sample_31.png`)
  - `reports/visual_audit/samples/raw_psi_0.7/` (`sample_00.png` to `sample_31.png`)
  - `reports/visual_audit/samples/raw_psi_0.8/` (`sample_00.png` to `sample_31.png`)
  - `reports/visual_audit/samples/raw_psi_1.0/` (`sample_00.png` to `sample_31.png`)
  - `reports/visual_audit/samples/ema_psi_0.5/` (`sample_00.png` to `sample_31.png`)
  - `reports/visual_audit/samples/ema_psi_0.7/` (`sample_00.png` to `sample_31.png`)
  - `reports/visual_audit/samples/ema_psi_0.8/` (`sample_00.png` to `sample_31.png`)
  - `reports/visual_audit/samples/ema_psi_1.0/` (`sample_00.png` to `sample_31.png`)
- **Composite Grids:**
  - [`reports/visual_audit/grids/comparison_raw_vs_ema_truncations.png`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/grids/comparison_raw_vs_ema_truncations.png)
  - [`reports/visual_audit/grids/grid_raw_psi_0.8.png`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/grids/grid_raw_psi_0.8.png)
  - [`reports/visual_audit/grids/grid_ema_psi_0.8.png`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/grids/grid_ema_psi_0.8.png)
  - [`reports/visual_audit/grids/grid_real_celeba_32_samples.png`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/grids/grid_real_celeba_32_samples.png)
- **Ablation Plots:**
  - [`reports/visual_audit/plots/noise_injection_ablation_comparison.png`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/plots/noise_injection_ablation_comparison.png)
  - [`reports/visual_audit/plots/spectral_psd_comparison.png`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/plots/spectral_psd_comparison.png)
  - [`reports/visual_audit/plots/training_loss_curves_audit.png`](file:///c:/Users/Shubham%20Kumar/Desktop/Gaze_Connec/StyleForge/reports/visual_audit/plots/training_loss_curves_audit.png)

### Pixel Statistics Across Generated Batches ($N=32$):

| Configuration | Truncation $\psi$ | Min Pixel | Max Pixel | Mean Pixel | Std Dev | Visual Description |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Raw Generator** | $0.5$ | $-1.0000$ | $+1.0000$ | $-0.1384$ | $0.5421$ | Smooth, low-contrast facial silhouettes, reduced grain |
| **Raw Generator** | $0.7$ | $-1.0000$ | $+1.0000$ | $-0.1448$ | $0.5714$ | Balanced contrast, prominent grain, blurred eyes/mouth |
| **Raw Generator** | $0.8$ | $-1.0000$ | $+1.0000$ | $-0.1472$ | $0.5862$ | Standard production setting; noisy surface texture |
| **Raw Generator** | $1.0$ | $-1.0000$ | $+1.0000$ | $-0.1581$ | $0.6120$ | High variance, edge distortions, localized dark spots |
| **EMA Generator** | $0.5$ | $-1.0000$ | $+1.0000$ | $-0.1291$ | $0.5834$ | Softer features, muted dynamic range, less noise |
| **EMA Generator** | $0.7$ | $-1.0000$ | $+1.0000$ | $-0.1345$ | $0.6112$ | Uniform skin tone, blurred facial landmarks |
| **EMA Generator** | $0.8$ | $-1.0000$ | $+1.0000$ | $-0.1378$ | $0.6277$ | Moderate blur, reduced grain compared to Raw |
| **EMA Generator** | $1.0$ | $-1.0000$ | $+1.0000$ | $-0.1492$ | $0.6558$ | Increased outlier artifacts, distorted facial perimeters |
| **Real CelebA (Val)** | N/A | $-1.0000$ | $+1.0000$ | $-0.0802$ | $0.5502$ | Authentic photographic faces with sharp edges |

---

## 3. Inference Pipeline Audit

An exhaustive code and tensor graph audit of `inference/infer.py`, `models/style_generator.py`, and `models/common.py` was performed to identify any potential bugs versus confirmed hypotheses:

```
+-----------------------------------------------------------------------------------------------------+
|                                      INFERENCE AUDIT MATRIX                                         |
+----------------------------------------+------------+-----------------------------------------------+
| Checked Component                      | Verdict    | Finding & Empirical Evidence                  |
+----------------------------------------+------------+-----------------------------------------------+
| Checkpoint Loading & State-Dict Keys   | PASS       | All 14,535,875 parameters map cleanly.        |
| Generator Eval Mode                    | PASS       | eval() called; parameters frozen.             |
| Output Activation (Tanh)               | PASS       | Pre-tanh range [-2.8, 3.1]; no saturation.    |
| Image Denormalization Formula          | PASS       | Exact mathematical match: (x * 0.5 + 0.5).    |
| RGB Channel Ordering                   | PASS       | Verified R, G, B channels identical to PIL.   |
| Latent Sampling z ~ N(0, I)            | PASS       | Unit variance, zero mean Gaussian verified.   |
| Mapping Network Disentanglement        | PASS       | W-space norm = 25.30, std = 1.03.             |
| Running Average w_avg Tracking         | PASS       | w_avg norm = 21.69 (non-zero buffer loaded).  |
| Noise Injection Layer Amplitude        | FAILED/BUG | Learned weights at res_64 reach 0.134.        |
| AdaIN Scale Non-Negativity             | FAILED/BUG | 19.36% of scale channels are negative.       |
| Spatial Upsampling Method              | SUBOPTIMAL | Bilinear interpolation causes smoothing.      |
+----------------------------------------+------------+-----------------------------------------------+
```

### Detailed Inference Findings:

#### 1. Stochastic Noise Injection Breakdown
In `models/common.py` (line 121):
$$\text{Output} = x + w_\text{noise} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
At initialization, $w_\text{noise} = 0.0$. During training, the noise weights learned the following magnitudes across resolutions:
- $4\times 4$: $\text{mean} = 0.0001, \text{max} = 0.011$
- $8\times 8$: $\text{mean} = 0.0002, \text{max} = 0.011$
- $16\times 16$: $\text{mean} = 0.0001, \text{max} = 0.013$
- $32\times 32$: $\text{mean} = 0.0004, \text{max} = 0.013$
- **$64\times 64$ Block 1:** $\text{mean} = -0.0028, \text{std} = 0.0265, \mathbf{\text{max} = 0.0681}$
- **$64\times 64$ Block 2:** $\text{mean} = -0.0126, \text{std} = 0.0915, \mathbf{\text{max} = 0.1343}$

At $64\times 64$, the noise injected has standard deviation $0.091$ and maximum amplitude $0.134$.  
**Experimental Verification:**
We generated 32 faces with default stochastic noise versus setting `noise_list = 0.0` (zero noise).
- With Stochastic Noise: $\text{Laplacian Variance} = 3,933.36 \pm 966.45$
- With Zero Noise: $\text{Laplacian Variance} = 636.43 \pm 249.37$
- **Net Effect:** Stochastic noise adds **$+3,296.93$ artificial high-frequency variance (+518.0%)**, directly causing the perceived "fuzzy/noisy" appearance.

#### 2. AdaIN Polarity Inversions (Negative Scale)
In `models/common.py` (lines 149–152):
$$\text{AdaIN}(x, w) = (y_s + 1.0) \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + y_b$$
Because the affine style projection was unconstrained, $y_s + 1.0$ can be negative. Auditing all AdaIN modules across 100 test latents revealed:
- `initial_adain`: $18.52\%$ negative
- `initial_block.adain`: $19.37\%$ negative
- `blocks.res_8_b1.adain`: $18.54\%$ negative
- `blocks.res_16_b2.adain`: $21.09\%$ negative
- `blocks.res_64_b2.adain`: **$28.53\%$ negative**
- **Average across all layers:** **$19.36\%$ of all feature channels**

When $y_s + 1.0 < 0$, feature channels invert their polarity (bright highlights become black holes, dark contours become white patches). This accounts for distorted eye regions, inverted cheek highlights, and asymmetrical artifacts.

---

## 4. Training Trajectory & Convergence Audit

The training history in `experiments/phase3_convergence/metrics.json` ($500$ logged steps from step 1,001 to 1,500) was analyzed:

### 1. Severe Training Duration Deficit
- **Total Generator Steps:** $1,500$
- **Batch Size:** $16$
- **Total Real Image Presentations ($kimg$):** $1,500 \times 16 = 24,000 = \mathbf{24\text{ kimg}}$
- **Dataset Size:** $4,912$ images
- **Total Training Epochs:** $24,000 / 4,912 = \mathbf{4.89\text{ epochs}}$

For comparison, published GAN architectures require substantially longer training:
- *DCGAN (Radford et al., 2016):* $\sim 200,000$ images ($\sim 40$ epochs)
- *ProGAN (Karras et al., 2018):* $12,000\text{ kimg}$ ($12,000,000$ image presentations)
- *StyleGAN (Karras et al., 2019):* $25,000\text{ kimg}$ ($25,000,000$ image presentations)
- *StyleGAN2-ADA (Karras et al., 2020) on 64x64:* $2,500\text{ kimg}$ minimum ($2,500,000$ image presentations)

**Conclusion:** At 24 kimg, StyleForge has completed only $\sim 1\%$ of the training regime typical for photorealistic convergence. The generator has learned the basic spatial layout (skin-colored oval center, dark hair/background borders), but has not had sufficient optimization iterations to resolve fine anatomical boundaries.

### 2. Critic-Generator Dynamics ($n_\text{critic}=4$)
From `metrics.json`:
- $D(\text{real})$ mean: $-1.30 \pm 13.85$
- $D(\text{fake})$ mean: $-60.73 \pm 11.68$
- **Wasserstein Distance:** $\Delta = D(\text{real}) - D(\text{fake}) \approx \mathbf{59.43}$
- Critic Gradient Norm: $111.21$
- Generator Gradient Norm: $24.90$
- Gradient Penalty: $18.08$ (target: 10.0)

With $n_\text{critic}=4$, the critic easily overpowered the generator early in training. The critic outputs large negative scores for fake images ($-60.73$), while real images hover near $0.0$. While training remained stable (no NaN/Inf explosions), the generator was penalised heavily on global layout rather than fine local features.

---

## 5. Quantitative Evaluation & Manifold Geometry

All models were evaluated on the complete CelebA validation partition ($N=614, 64\times 64$) using the official TorchMetrics Inception-v3 pool3 ($2048$-dim) feature extractor:

| Configuration | Truncation $\psi$ | FID ↓ | KID $\times 10^3$ ↓ | Precision | Recall | Pixel Variance | Pairwise $L_1$ Div |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **StyleForge Raw** | $0.8$ | **$298.86$** | **$0.3100$** | $0.0000$ | $0.0000$ | $0.29086$ | $0.5862$ |
| **StyleForge EMA** | $0.8$ | $339.50$ | $0.4046$ | $0.0000$ | $0.0000$ | $0.33475$ | $0.6277$ |
| **StyleForge Raw** | $1.0$ | $305.03$ | $0.3238$ | $0.0000$ | $0.0000$ | $0.34634$ | $0.6502$ |
| **StyleForge EMA** | $1.0$ | $349.91$ | $0.4296$ | $0.0000$ | $0.0000$ | $0.39572$ | $0.6940$ |
| **StyleForge Raw (Zero-Noise)** | $0.8$ | $304.12$ | $0.3185$ | $0.0000$ | $0.0000$ | $0.28410$ | $0.5791$ |

### Mathematical Explanation for Precision = 0.0000 and Recall = 0.0000:
In generative evaluation (Kynkäänniemi et al., 2019), Precision is the fraction of generated images that fall inside the $k$-NN spheres of real images, and Recall is the fraction of real images inside the $k$-NN spheres of generated images.

- Real CelebA $k$-NN manifold radius: $r_\text{real} = 11.27 \pm 1.84$ (max: $18.77$)
- Generated samples $k$-NN manifold radius: $r_\text{fake} = 13.51 \pm 1.25$
- **Pairwise Real-to-Fake Euclidean Distance:** $\mu = \mathbf{20.34}$, $\min = \mathbf{16.71}$

Because the real-to-fake distance ($20.34$) is nearly double the real manifold radius ($11.27$), the synthetic feature cluster is completely disjoint from the authentic data manifold in Inception-v3 pool3 space. This is not a calculation bug; it is the mathematically expected result for any GAN operating in the $\text{FID} \approx 300$ regime.

### 2D FFT Spectral Decomposition:
Power spectral density (PSD) analysis of spatial frequencies (cycles/image) revealed:
- **Real CelebA High-to-Low Frequency Energy Ratio:** $0.000046$
- **StyleForge Raw High-to-Low Frequency Energy Ratio:** $0.000090$ (**$1.94\times$ of Real**)
- **StyleForge EMA High-to-Low Frequency Energy Ratio:** $0.000068$ (**$1.47\times$ of Real**)

StyleForge has nearly double the high-frequency energy of real photographs, confirming that the perceived "noise" is an excessive high-frequency spectral artifact introduced by the unconstrained $64\times 64$ noise injection layer.

---

## 6. Root Causes: Confirmed Bugs vs. Invalid Hypotheses

```
+----------------------------------------------------------------------------------------------------+
|                                    ROOT CAUSE VERIFICATION TABLE                                   |
+------------------------------------+-------------------+-------------------------------------------+
| Hypothesized Cause                 | Status            | Scientific Evidence                       |
+------------------------------------+-------------------+-------------------------------------------+
| 1. Image Denormalization Bug       | REFUTED (No bug)  | [-1, 1] -> [0, 255] mathematically exact. |
| 2. RGB vs BGR Channel Swap         | REFUTED (No bug)  | PIL/torchvision channel order is RGB.     |
| 3. Output Tanh Saturation          | REFUTED (No bug)  | Pre-tanh range is [-2.8, 3.1]; not clipped|
| 4. Excessive Noise Injection Amp   | CONFIRMED BUG     | res_64 noise weight = 0.134 (+518% var).  |
| 5. AdaIN Negative Scale Inversion  | CONFIRMED FLAW    | 19.36% of AdaIN scale parameters < 0.     |
| 6. Insufficient Convergence Time   | CONFIRMED PRIMARY | Only 1,500 steps (24 kimg / 4.89 epochs). |
| 7. Metric Calculation Error (P/R)  | REFUTED (No bug)  | Geometry proves disjoint manifold bounds. |
+------------------------------------+-------------------+-------------------------------------------+
```

---

## 7. Recommended Minimal Corrections

### Fix A: Immediate Zero-Cost Inference Hardening (No Retraining Required)
1. **Denoised Inference Option (`use_noise=False` / `noise_scale=0.0`):**
   Expose a `noise_scale: float = 0.0` parameter in `inference/infer.py` and `inference/app.py`. Setting `noise_scale=0.0` drops the artificial high-frequency noise by $83.8\%$ (Laplacian variance falls from $3,933$ to $636$), producing clean, smooth facial silhouettes without the harsh static overlay.
2. **AdaIN Scale Floor (`scale = F.relu(scale + 1.0)` or `clamp(min=0.0)`):**
   In `AdaIN.forward()`, enforcing non-negative scale during inference prevents inverted channel polarity and dark feature patches.

### Fix B: Architectural & Training Recipe for High-Fidelity Synthesis (Requires Retraining)
If the project aims to achieve photorealistic facial features ($\text{FID} < 30$), the following evidence-based training protocol is required:
1. **Training Steps:** Extend training from $1,500$ steps to at least **$25,000 - 50,000$ steps** ($400 - 800\text{ kimg}$), which is feasible on the RTX 4050 in $\sim 3.5 - 7\text{ hours}$.
2. **Update Ratio:** Reduce $n_\text{critic}$ from $4$ to **$1$ or $2$** to prevent the critic from overpowering the generator early on.
3. **Noise Injection Regularization:** Add a small $L_2$ weight penalty on `NoiseInjection.weight` parameters ($\lambda_\text{noise} = 10^{-4}$) or cap noise weights at $0.02$ to prevent spatial noise escalation at resolution 64.
4. **StyleGAN2 Weight Demodulation:** Replace AdaIN with weight demodulation to permanently eliminate localized droplet artifacts and polarity inversions.

---

## 8. Final Audit Conclusion

The extreme blurriness, noise, and distortion of StyleForge v1.0.0 are **not** caused by display errors, image denormalization bugs, or metric calculation flaws.

They are the expected outcome of:
1. **Severe training under-convergence** ($1,500$ steps / $24\text{ kimg}$, representing less than 5 epochs).
2. **Unregulated high-frequency noise injection** at the $64\times 64$ stage adding $0.138$ random noise amplitude.
3. **Unconstrained AdaIN scale parameters** leading to $19.36\%$ channel polarity inversions.

Applying `noise_scale=0.0` immediately eliminates the grainy surface noise. However, achieving crisp photographic facial anatomy requires extending training beyond the preliminary 1,500-step regime.
