"""StyleForge Interactive Generative AI Studio (Gradio UI).

A professional web interface for real-time face synthesis, latent space
exploration, truncation sweeps, and multi-scale style mixing.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import numpy as np
from PIL import Image

from inference.infer import StyleForgeInference, make_grid

# Initialize inference engine once
try:
    engine = StyleForgeInference()
    init_error = None
except Exception as e:
    engine = None
    init_error = str(e)

def generate_faces(seed, num_samples, truncation_psi, use_ema, randomize_seed):
    if engine is None:
        return None, None, f"Error: {init_error}"
    if randomize_seed:
        seed = int(np.random.randint(0, 2**31 - 1))
    else:
        seed = int(seed)

    images = engine.generate(
        seed=seed,
        num_samples=int(num_samples),
        truncation_psi=float(truncation_psi),
        use_ema=bool(use_ema),
    )
    grid = make_grid(images, cols=min(4, len(images)))
    return images, grid, f"Generated {num_samples} face(s) using Seed={seed} (Truncation psi={truncation_psi:.2f}, EMA={use_ema})"

def run_interpolation(seed_a, seed_b, steps, space, truncation_psi, use_ema):
    if engine is None:
        return None, None, f"Error: {init_error}"

    frames = engine.interpolate(
        seed_a=int(seed_a),
        seed_b=int(seed_b),
        steps=int(steps),
        space=space,
        truncation_psi=float(truncation_psi),
        use_ema=bool(use_ema),
    )
    grid = make_grid(frames, cols=int(steps))
    return frames, grid, f"Interpolation between Seed {seed_a} and Seed {seed_b} in {space}-space ({steps} steps)"

def run_style_mix(seed_a, seed_b, coarse_source, truncation_psi, use_ema):
    if engine is None:
        return None, None, None, f"Error: {init_error}"

    coarse_from_b = (coarse_source == "Source B (Coarse Pose) + Source A (Fine Style)")
    img_a, img_b, img_mixed = engine.mix_styles(
        seed_a=int(seed_a),
        seed_b=int(seed_b),
        coarse_from_b=coarse_from_b,
        truncation_psi=float(truncation_psi),
        use_ema=bool(use_ema),
    )
    msg = f"Style mixing complete. Coarse geometry from {'Seed B' if coarse_from_b else 'Seed A'}."
    return img_a, img_b, img_mixed, msg

# Gradio Theme and Layout (Using system fonts for instant offline/air-gapped loading)
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
    font=["system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme, title="StyleForge — Generative AI Studio") as demo:
    gr.Markdown(
        """
        # StyleForge: Style-Based Facial Generation Studio
        ### High-Stability GAN Synthesis, Latent Space Traversal & Style Mixing
        *Trained on Authentic Photographic CelebA ($64\\times 64$) with PyTorch & CUDA*
        """
    )

    if init_error:
        gr.Warning(f"Checkpoint initialization warning: {init_error}")

    with gr.Tabs():
        # TAB 1: Face Synthesis
        with gr.TabItem("Face Synthesis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Generation Parameters")
                    seed_input = gr.Number(value=42, label="Random Seed", precision=0)
                    randomize_chk = gr.Checkbox(value=False, label="Randomize Seed Each Time")
                    num_slider = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size (Number of Faces)")
                    trunc_slider = gr.Slider(minimum=0.0, maximum=1.2, value=0.8, step=0.05, label="Truncation psi (Fidelity vs Variance)")
                    ema_chk = gr.Checkbox(value=False, label="Use EMA Weights (Smoothed)")
                    gen_btn = gr.Button("Generate Faces", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### Generated Visual Outputs")
                    gallery_output = gr.Gallery(label="Individual Samples", columns=4, height="auto")
                    grid_output = gr.Image(label="Composite Grid (Downloadable)", type="pil")
                    status_text = gr.Textbox(label="Status / Seed Metadata", interactive=False)

            gen_btn.click(
                fn=generate_faces,
                inputs=[seed_input, num_slider, trunc_slider, ema_chk, randomize_chk],
                outputs=[gallery_output, grid_output, status_text],
            )

        # TAB 2: Latent Interpolation
        with gr.TabItem("Latent Walk"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Interpolation Controls")
                    interp_seed_a = gr.Number(value=101, label="Source Identity A (Seed)", precision=0)
                    interp_seed_b = gr.Number(value=202, label="Target Identity B (Seed)", precision=0)
                    steps_slider = gr.Slider(minimum=4, maximum=16, value=8, step=1, label="Interpolation Frames")
                    space_radio = gr.Radio(choices=["W", "Z"], value="W", label="Interpolation Space (W=LERP, Z=SLERP)")
                    interp_trunc = gr.Slider(minimum=0.0, maximum=1.2, value=0.8, step=0.05, label="Truncation psi")
                    interp_ema = gr.Checkbox(value=False, label="Use EMA Weights")
                    interp_btn = gr.Button("Run Latent Walk", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Continuous Morphing Sequence")
                    interp_gallery = gr.Gallery(label="Sequence Frames", columns=8, height="auto")
                    interp_grid = gr.Image(label="Continuous Filmstrip", type="pil")
                    interp_status = gr.Textbox(label="Metadata", interactive=False)

            interp_btn.click(
                fn=run_interpolation,
                inputs=[interp_seed_a, interp_seed_b, steps_slider, space_radio, interp_trunc, interp_ema],
                outputs=[interp_gallery, interp_grid, interp_status],
            )

        # TAB 3: Style Mixing Studio
        with gr.TabItem("Style Mixing"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Style Sources")
                    mix_seed_a = gr.Number(value=777, label="Identity A (Seed)", precision=0)
                    mix_seed_b = gr.Number(value=888, label="Identity B (Seed)", precision=0)
                    mix_mode = gr.Dropdown(
                        choices=[
                            "Source B (Coarse Pose) + Source A (Fine Style)",
                            "Source A (Coarse Pose) + Source B (Fine Style)",
                        ],
                        value="Source B (Coarse Pose) + Source A (Fine Style)",
                        label="Mixing Split Paradigm",
                    )
                    mix_trunc = gr.Slider(minimum=0.0, maximum=1.2, value=0.8, step=0.05, label="Truncation psi")
                    mix_ema = gr.Checkbox(value=False, label="Use EMA Weights")
                    mix_btn = gr.Button("Mix Styles", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Disentanglement Output")
                    with gr.Row():
                        out_src_a = gr.Image(label="Source Identity A", type="pil")
                        out_src_b = gr.Image(label="Source Identity B", type="pil")
                        out_mixed = gr.Image(label="Synthesized Hybrid Face", type="pil")
                    mix_status = gr.Textbox(label="Status", interactive=False)

            mix_btn.click(
                fn=run_style_mix,
                inputs=[mix_seed_a, mix_seed_b, mix_mode, mix_trunc, mix_ema],
                outputs=[out_src_a, out_src_b, out_mixed, mix_status],
            )

        # TAB 4: Architecture & Benchmark
        with gr.TabItem("Model Card & Benchmarks"):
            gr.Markdown(
                """
                ### StyleForge Production Model Specifications
                - **Architecture:** Style-Based Generator with 8-layer MLP Mapping Network ($f: \\mathcal{Z} \\to \\mathcal{W}$), learned constant input ($4\\times 4\\times 512$), AdaIN style modulation, and per-channel noise injection.
                - **Resolution:** $64\\times 64\\times 3$ (RGB)
                - **Generator Parameters:** $14,535,875$
                - **Critic Parameters:** $12,174,721$
                - **Total System Parameters:** $26,710,596$
                - **Exported Checkpoint Size:** $110.95\\text{ MB}$ (Stripped optimizer momentum buffers)
                - **License:** MIT License

                ### Reconciled Quantitative Benchmarks (CelebA Validation Split, N=614)
                | Milestone / Configuration | Global Steps | Inception-v3 FID ↓ | KID Mean ↓ | Pairwise $L_1$ Diversity ↑ | Batch Variance ↑ |
                |---|:---:|:---:|:---:|:---:|:---:|
                | **StyleForge Raw (Production)** | **1,500** | **297.76** | **0.3285** | 0.5779 | 0.28987 |
                | StyleForge EMA (Stabilized) | 1,500 | 336.98 | 0.4167 | 0.6223 | 0.33502 |
                | StyleForge Intermediate | 1,000 | 305.99 | 0.3531 | 0.5953 | 0.31074 |
                | StyleForge Logistic + $R_1$ | 300 | 373.66 | 0.4975 | 0.6692 | 0.43134 |
                | Convolutional WGAN-GP Baseline | 1,000 | 344.90 | 0.4225 | 0.0410 (Collapsed) | 0.00153 (Collapsed) |

                *Evaluated with official TorchMetrics Inception-v3 pool3 ($2048$-dim) feature extractor.*
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
