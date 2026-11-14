import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort

def test_enhancer():
    model_path = Path("checkpoints/enhancer/gfpgan_1.4.onnx")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    print(f"Loading ONNX model from {model_path} ({model_path.stat().st_size / (1024*1024):.1f} MB)...")
    
    # Check available execution providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    print(f"Available providers: {ort.get_available_providers()}, using: {providers}")
    
    session = ort.InferenceSession(str(model_path), providers=providers)
    
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"Input: {input_info.name}, shape={input_info.shape}, type={input_info.type}")
    print(f"Output: {output_info.name}, shape={output_info.shape}, type={output_info.type}")

    # Load one of our own StyleForge generated faces
    # e.g. from artifacts or generate one using our checkpoint
    sys.path.insert(0, ".")
    from core.engine.inference_engine import InferenceEngine
    from core.engine.config import GenerationConfig, InferenceMode

    engine = InferenceEngine.get_instance()
    cfg = GenerationConfig(seed=42, num_samples=1, mode=InferenceMode.SMOOTH, noise_scale=0.25)
    pil_images, _ = engine.generate(cfg)
    raw_face = pil_images[0] # 64x64 PIL Image
    
    out_dir = Path("artifacts/enhancer_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_face.save(out_dir / "01_raw_styleforge_64x64.png")
    print("Saved 01_raw_styleforge_64x64.png")

    # Resize raw 64x64 face to 512x512
    face_512 = raw_face.resize((512, 512), Image.Resampling.BILINEAR)
    face_512.save(out_dir / "02_resized_512x512.png")

    # Preprocess for GFPGAN:
    # Normalize RGB to [-1, 1], shape (1, 3, 512, 512)
    img_np = np.array(face_512).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5 # [-1, 1]
    # GFPGAN expects BGR or RGB? Usually BGR for OpenCV-based models, or RGB. Let's convert RGB to BGR:
    img_bgr = img_np[:, :, ::-1]
    input_tensor = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

    # Run enhancement
    t0 = time.perf_counter()
    outputs = session.run([output_info.name], {input_info.name: input_tensor})
    latency = (time.perf_counter() - t0) * 1000.0
    print(f"Enhancement completed in {latency:.2f} ms!")

    # Postprocess output
    out_img = outputs[0][0] # (3, 512, 512)
    out_img = np.clip((out_img * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    # Convert BGR back to RGB
    out_rgb = np.transpose(out_img, (1, 2, 0))[:, :, ::-1]
    enhanced_pil = Image.fromarray(out_rgb)
    enhanced_pil.save(out_dir / "03_enhanced_face_512x512.png")
    print(f"Saved enhanced face to {out_dir / '03_enhanced_face_512x512.png'}")

if __name__ == "__main__":
    test_enhancer()
