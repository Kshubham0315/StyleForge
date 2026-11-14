import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort

def create_comparison():
    model_path = Path("checkpoints/enhancer/gfpgan_1.4.onnx")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    sys.path.insert(0, ".")
    from core.engine.inference_engine import InferenceEngine
    from core.engine.config import GenerationConfig, InferenceMode

    engine = InferenceEngine.get_instance()
    
    seeds = [42, 100, 2026, 777]
    raw_images = []
    enhanced_images = []

    for s in seeds:
        cfg = GenerationConfig(seed=s, num_samples=1, mode=InferenceMode.SMOOTH, noise_scale=0.25)
        raw_face = engine.generate(cfg)[0][0]
        raw_images.append(raw_face.resize((512, 512), Image.Resampling.NEAREST))

        # Preprocess
        face_512 = raw_face.resize((512, 512), Image.Resampling.BILINEAR)
        img_np = (np.array(face_512).astype(np.float32) / 255.0 - 0.5) / 0.5
        img_bgr = img_np[:, :, ::-1]
        inp = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

        out = session.run([output_name], {input_name: inp})[0][0]
        out = np.clip((out * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        out_rgb = np.transpose(out, (1, 2, 0))[:, :, ::-1]
        enhanced_images.append(Image.fromarray(out_rgb))

    # Build 2x4 grid
    # Row 1: Raw (upscaled)
    # Row 2: Enhanced
    grid = Image.new("RGB", (512 * 4, 512 * 2))
    for i in range(4):
        grid.paste(raw_images[i], (i * 512, 0))
        grid.paste(enhanced_images[i], (i * 512, 512))

    out_path = Path("artifacts/styleforge_enhanced_comparison.png")
    grid.save(out_path)
    print(f"Saved comparison grid to {out_path}")

if __name__ == "__main__":
    create_comparison()
