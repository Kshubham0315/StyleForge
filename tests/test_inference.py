"""Unit and integration tests for StyleForge inference engine."""

from pathlib import Path
import pytest
import torch
from PIL import Image

from inference.infer import StyleForgeInference, make_grid, tensor_to_pil

def test_inference_engine_initialization():
    engine = StyleForgeInference(checkpoint_path="checkpoints/styleforge_final_model.pt")
    assert engine is not None
    assert engine.raw_generator is not None
    assert engine.ema_generator is not None
    assert engine.metadata["img_size"] == 64

def test_inference_generate_single_and_batch():
    engine = StyleForgeInference(checkpoint_path="checkpoints/styleforge_final_model.pt")
    
    # Single generation
    imgs = engine.generate(seed=42, num_samples=1, truncation_psi=0.8)
    assert len(imgs) == 1
    assert isinstance(imgs[0], Image.Image)
    assert imgs[0].size == (64, 64)

    # Batch generation with EMA
    batch = engine.generate(seed=123, num_samples=4, truncation_psi=0.7, use_ema=True)
    assert len(batch) == 4
    for im in batch:
        assert isinstance(im, Image.Image)
        assert im.size == (64, 64)

def test_inference_interpolation_w_and_z():
    engine = StyleForgeInference(checkpoint_path="checkpoints/styleforge_final_model.pt")

    # W-space interpolation
    frames_w = engine.interpolate(seed_a=10, seed_b=20, steps=4, space="W")
    assert len(frames_w) == 4
    assert isinstance(frames_w[0], Image.Image)

    # Z-space interpolation
    frames_z = engine.interpolate(seed_a=10, seed_b=20, steps=4, space="Z")
    assert len(frames_z) == 4
    assert isinstance(frames_z[0], Image.Image)

def test_inference_style_mixing():
    engine = StyleForgeInference(checkpoint_path="checkpoints/styleforge_final_model.pt")
    img_a, img_b, img_mixed = engine.mix_styles(seed_a=1, seed_b=2, coarse_from_b=True)

    assert isinstance(img_a, Image.Image)
    assert isinstance(img_b, Image.Image)
    assert isinstance(img_mixed, Image.Image)
    assert img_mixed.size == (64, 64)

def test_make_grid():
    imgs = [Image.new("RGB", (64, 64), color="red") for _ in range(4)]
    grid = make_grid(imgs, cols=2)
    assert isinstance(grid, Image.Image)
    assert grid.size == (128, 128)
