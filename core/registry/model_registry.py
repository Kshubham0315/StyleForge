"""Model Registry catalog, lifecycle management, and SHA256 integrity governance."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.engine.checkpoint_manager import CheckpointManager, PRODUCTION_CHECKPOINT_SHA256
from core.engine.config import CheckpointError

logger = logging.getLogger(__name__)


class ModelMetadata(BaseModel):
    """Metadata describing a registered generative model checkpoint."""
    model_name: str = "StyleForge"
    version_tag: str = "v1.0.0"
    checkpoint_path: str
    sha256: str
    is_official: bool = False
    resolution: int = 64
    latent_dim: int = 512
    style_dim: int = 512
    global_step: int = 1500
    metrics: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(
        default_factory=lambda: [
            "Native output resolution is strictly 64x64 pixels.",
            "Trained exclusively on CelebA human facial distribution.",
            "Inference optimization (e.g. noise scale 0.25) improves edge spectral parity but does not alter resolution.",
        ]
    )


class ModelRegistry:
    """Manages model versions, verification, and active deployment state."""

    def __init__(self, registry_file: Path = Path("data/model_registry.json")):
        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.active_version: str = "v1.0.0"
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_file.exists():
            try:
                data = json.loads(self.registry_file.read_text(encoding="utf-8"))
                self.models = data.get("models", {})
                self.active_version = data.get("active_version", "v1.0.0")
            except Exception as e:
                logger.warning("Failed to load model registry from %s: %s", self.registry_file, e)

        # Ensure default official model is registered
        if "v1.0.0" not in self.models:
            official_meta = {
                "model_name": "StyleForge",
                "version_tag": "v1.0.0",
                "checkpoint_path": "checkpoints/styleforge_final_model.pt",
                "sha256": PRODUCTION_CHECKPOINT_SHA256,
                "is_official": True,
                "resolution": 64,
                "latent_dim": 512,
                "style_dim": 512,
                "global_step": 1500,
                "metrics": {
                    "fid_raw_baseline": 298.86,
                    "fid_raw_smooth_mode": 304.76,
                    "fid_raw_best": 294.75,
                    "laplacian_variance_baseline": 4057.01,
                    "laplacian_variance_smooth_mode": 922.43,
                    "psd_ratio_real_val": 0.000046,
                    "psd_ratio_smooth_mode": 0.000045,
                },
                "status": "active",
            }
            self.models["v1.0.0"] = official_meta
            self._save_registry()

    def _save_registry(self) -> None:
        payload = {
            "active_version": self.active_version,
            "models": self.models,
        }
        self.registry_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register_checkpoint(
        self,
        checkpoint_path: Path,
        version_tag: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Registers and validates a new checkpoint into the catalog."""
        if not checkpoint_path.exists():
            raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")

        meta = CheckpointManager.inspect_checkpoint(checkpoint_path)
        sha256 = meta["sha256"]

        entry = {
            "model_name": meta.get("model_name", "StyleForge"),
            "version_tag": version_tag,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "sha256": sha256,
            "is_official": sha256 == PRODUCTION_CHECKPOINT_SHA256,
            "resolution": meta.get("img_size", 64),
            "latent_dim": meta.get("latent_dim", 512),
            "style_dim": meta.get("style_dim", 512),
            "global_step": meta.get("global_step", "N/A"),
            "metrics": metrics or {},
            "status": "registered",
        }
        self.models[version_tag] = entry
        self._save_registry()
        logger.info("Successfully registered model checkpoint %s (version: %s)", sha256[:12], version_tag)
        return entry

    def set_active_version(self, version_tag: str) -> Dict[str, Any]:
        """Activates a verified registered model version."""
        if version_tag not in self.models:
            raise KeyError(f"Model version '{version_tag}' is not registered.")

        # Verify file still exists on disk
        ckpt_path = Path(self.models[version_tag]["checkpoint_path"])
        if not ckpt_path.exists():
            raise CheckpointError(f"Model checkpoint for '{version_tag}' is missing from disk: {ckpt_path}")

        for k in self.models:
            self.models[k]["status"] = "active" if k == version_tag else "inactive"

        self.active_version = version_tag
        self._save_registry()
        logger.info("Activated model version: %s", version_tag)
        return self.models[version_tag]

    def rollback(self, previous_version: str = "v1.0.0") -> Dict[str, Any]:
        """Rollback to a previous stable model version."""
        logger.warning("Rolling back active model version to %s", previous_version)
        return self.set_active_version(previous_version)

    def get_active_model(self) -> Dict[str, Any]:
        """Returns metadata of currently active registered model."""
        return self.models.get(self.active_version, self.models["v1.0.0"])

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns all registered model entries."""
        return list(self.models.values())
