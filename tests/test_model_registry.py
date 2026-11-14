"""Tests for StyleForge Model Registry & MLOps."""

from pathlib import Path
import pytest

from core.engine.checkpoint_manager import PRODUCTION_CHECKPOINT_SHA256
from core.registry.model_registry import ModelRegistry


def test_model_registry_initialization(tmp_path):
    reg_file = tmp_path / "registry.json"
    registry = ModelRegistry(registry_file=reg_file)

    active = registry.get_active_model()
    assert active["version_tag"] == "v1.0.0"
    assert active["sha256"] == PRODUCTION_CHECKPOINT_SHA256
    assert active["is_official"] is True
    assert active["resolution"] == 64


def test_model_registry_custom_registration_and_activation(tmp_path):
    reg_file = tmp_path / "registry.json"
    registry = ModelRegistry(registry_file=reg_file)

    ckpt_path = Path("checkpoints/styleforge_final_model.pt")
    entry = registry.register_checkpoint(
        checkpoint_path=ckpt_path,
        version_tag="v1.0.0-candidate",
        metrics={"fid": 298.86},
    )
    assert entry["version_tag"] == "v1.0.0-candidate"
    assert entry["status"] == "registered"

    # Activate new version
    activated = registry.set_active_version("v1.0.0-candidate")
    assert activated["status"] == "active"
    assert registry.active_version == "v1.0.0-candidate"

    # Rollback to v1.0.0
    rolled_back = registry.rollback("v1.0.0")
    assert rolled_back["status"] == "active"
    assert registry.active_version == "v1.0.0"
