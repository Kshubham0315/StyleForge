"""Tests for StyleForge v2.0 Modular Core Inference Engine."""

from pathlib import Path
import pytest
import torch
from PIL import Image

from core.engine.checkpoint_manager import CheckpointManager, PRODUCTION_CHECKPOINT_SHA256
from core.engine.config import (
    GenerationConfig,
    InferenceError,
    InferenceMode,
    InterpolationConfig,
    OutputFormat,
    StyleMixConfig,
)
from core.engine.device_manager import DeviceManager
from core.engine.inference_engine import InferenceEngine
from core.engine.model_loader import ModelLoader
from core.engine.postprocessor import ImagePostProcessor


def test_device_manager_resolution():
    cpu_dev = DeviceManager.resolve_device("cpu")
    assert cpu_dev.type == "cpu"

    auto_dev = DeviceManager.resolve_device(None)
    assert auto_dev.type in ("cuda", "cpu")

    vram = DeviceManager.get_vram_info(cpu_dev)
    assert "allocated_mb" in vram
    assert "total_mb" in vram


def test_checkpoint_manager_inspection():
    ckpt_path = Path("checkpoints/styleforge_final_model.pt")
    assert ckpt_path.exists()

    meta = CheckpointManager.inspect_checkpoint(ckpt_path)
    assert meta["sha256"] == PRODUCTION_CHECKPOINT_SHA256
    assert meta["is_official"] is True
    assert meta["img_size"] == 64
    assert meta["latent_dim"] == 512
    assert meta["has_raw_weights"] is True
    assert meta["has_ema_weights"] is True


def test_image_postprocessor():
    dummy_tensor = torch.zeros(3, 64, 64)  # in [-1, 1] range (zeros -> 0.5 -> 127 in uint8)
    pil_img = ImagePostProcessor.tensor_to_pil(dummy_tensor)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (64, 64)

    b64 = ImagePostProcessor.pil_to_base64(pil_img, format="PNG")
    assert b64.startswith("data:image/png;base64,")

    raw_bytes = ImagePostProcessor.pil_to_bytes(pil_img, format="JPEG")
    assert len(raw_bytes) > 0

    grid = ImagePostProcessor.make_grid([pil_img, pil_img], cols=2)
    assert grid.size == (128, 64)


def test_generation_config_modes():
    # Standard Mode defaults to noise_scale = 1.0
    cfg_standard = GenerationConfig(mode=InferenceMode.STANDARD)
    assert cfg_standard.noise_scale == 1.0

    # Smooth Mode defaults to noise_scale = 0.25
    cfg_smooth = GenerationConfig(mode=InferenceMode.SMOOTH)
    assert cfg_smooth.noise_scale == 0.25

    # Boundary validations
    with pytest.raises(Exception):
        GenerationConfig(num_samples=0)

    with pytest.raises(Exception):
        GenerationConfig(truncation_psi=1.5)


def test_inference_engine_modes_and_seeds():
    engine = InferenceEngine(checkpoint_path="checkpoints/styleforge_final_model.pt", device="cpu")

    # Deterministic generation with fixed seed
    cfg1 = GenerationConfig(seed=2026, num_samples=1, mode=InferenceMode.STANDARD)
    imgs1, meta1 = engine.generate(cfg1)
    assert len(imgs1) == 1
    assert imgs1[0].size == (64, 64)
    assert meta1["seed"] == 2026
    assert meta1["noise_scale"] == 1.0
    assert meta1["latency_ms"] > 0

    # Smooth mode generation with same seed
    cfg2 = GenerationConfig(seed=2026, num_samples=1, mode=InferenceMode.SMOOTH)
    imgs2, meta2 = engine.generate(cfg2)
    assert len(imgs2) == 1
    assert meta2["noise_scale"] == 0.25
    assert meta2["mode"] == "smooth"
