import pytest
from PIL import Image
import numpy as np

from core.engine.enhancer import FaceEnhancer
from core.engine.config import GenerationConfig, InferenceMode
from core.engine.inference_engine import InferenceEngine


def test_face_enhancer_initialization():
    enhancer = FaceEnhancer.get_instance()
    assert enhancer is not None
    assert enhancer.is_available is True


def test_face_enhancer_execution():
    enhancer = FaceEnhancer.get_instance()
    # Create synthetic test face image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    enhanced = enhancer.enhance_pil(dummy_img)
    assert enhanced is not None
    assert enhanced.size == (512, 512)


def test_inference_engine_with_enhancement():
    engine = InferenceEngine.get_instance()
    cfg = GenerationConfig(
        seed=42,
        num_samples=1,
        mode=InferenceMode.SMOOTH,
        noise_scale=0.25,
        enhance=True,
    )
    images, telemetry = engine.generate(cfg)
    assert len(images) == 1
    assert images[0].size == (512, 512)
    assert telemetry["enhanced"] is True
    assert telemetry["resolution"] == "512x512"
