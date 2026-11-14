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

def test_inference_parameter_validation():
    engine = StyleForgeInference(checkpoint_path="checkpoints/styleforge_final_model.pt")
    
    with pytest.raises(ValueError, match=r"num_samples must be >= 1"):
        engine.generate(seed=42, num_samples=0)
        
    with pytest.raises(ValueError, match=r"truncation_psi must be"):
        engine.generate(seed=42, truncation_psi=-0.5)

    with pytest.raises(ValueError, match=r"steps must be >= 2"):
        engine.interpolate(seed_a=1, seed_b=2, steps=1)

    with pytest.raises(ValueError, match=r"space must be 'W' or 'Z'"):
        engine.interpolate(seed_a=1, seed_b=2, space="INVALID")

def test_inference_missing_checkpoint():
    with pytest.raises(FileNotFoundError) as exc_info:
        StyleForgeInference(checkpoint_path="checkpoints/nonexistent_model.pt")
    msg = str(exc_info.value)
    assert "checkpoint not found" in msg.lower()
    assert "scripts/export_final_model.py" in msg

def test_inference_cpu_device_support():
    engine = StyleForgeInference(checkpoint_path="checkpoints/styleforge_final_model.pt", device="cpu")
    assert str(engine.device) == "cpu"
    imgs = engine.generate(seed=42, num_samples=1)
    assert len(imgs) == 1
    assert imgs[0].size == (64, 64)

def test_app_callback_arity_when_engine_none(monkeypatch):
    import inference.app as app
    monkeypatch.setattr(app, "engine", None)
    monkeypatch.setattr(app, "init_error", "Mock checkpoint missing error")

    # generate_faces callback arity (must be exactly 3 for gen_btn.click outputs)
    res_gen = app.generate_faces(seed=42, num_samples=1, truncation_psi=0.8, use_ema=False, randomize_seed=False)
    assert len(res_gen) == 3
    assert res_gen[0] is None and res_gen[1] is None
    assert "Mock checkpoint missing error" in res_gen[2]

    # run_interpolation callback arity (must be exactly 3 for interp_btn.click outputs)
    res_interp = app.run_interpolation(seed_a=1, seed_b=2, steps=4, space="W", truncation_psi=0.8, use_ema=False)
    assert len(res_interp) == 3
    assert res_interp[0] is None and res_interp[1] is None
    assert "Mock checkpoint missing error" in res_interp[2]

    # run_style_mix callback arity (must be exactly 4 for mix_btn.click outputs)
    res_mix = app.run_style_mix(seed_a=1, seed_b=2, coarse_source="Source A (Coarse Pose) + Source B (Fine Style)", truncation_psi=0.8, use_ema=False)
    assert len(res_mix) == 4
    assert res_mix[0] is None and res_mix[1] is None and res_mix[2] is None
    assert "Mock checkpoint missing error" in res_mix[3]

