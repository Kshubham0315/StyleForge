"""Checkpoint management, verification, and SHA256 security integrity checks."""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from core.engine.config import CheckpointError

logger = logging.getLogger(__name__)

PRODUCTION_CHECKPOINT_SHA256 = "ec8e4bf1e4f21f240f7793a1b3922a4510394831d0c39de521c4cd456c9eed64"
DEFAULT_CHECKPOINT_LOCATIONS = [
    Path("checkpoints/styleforge_final_model.pt"),
    Path("experiments/phase3_convergence/checkpoints/checkpoint_final.pt"),
    Path("experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt"),
]


class CheckpointManager:
    """Manages model checkpoint resolution, SHA256 hashing, and validation."""

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Computes the SHA-256 hexadecimal digest of a file in streaming chunks."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def resolve_checkpoint(cls, custom_path: Optional[str | Path] = None) -> Path:
        """Finds the checkpoint file from custom or default candidates.

        Raises:
            CheckpointError if no valid checkpoint is found.
        """
        if custom_path is not None:
            p = Path(custom_path)
            if p.exists() and p.is_file():
                return p
            raise CheckpointError(
                f"StyleForge checkpoint not found: {custom_path}.\n"
                f"Installation: run python scripts/export_final_model.py"
            )

        for candidate in DEFAULT_CHECKPOINT_LOCATIONS:
            if candidate.exists() and candidate.is_file():
                return candidate

        checked_paths = [str(c) for c in DEFAULT_CHECKPOINT_LOCATIONS]
        raise CheckpointError(
            f"StyleForge checkpoint not found.\n"
            f"Checked default locations:\n  - " + "\n  - ".join(checked_paths) + "\n\n"
            f"Expected: checkpoints/styleforge_final_model.pt (SHA256: {PRODUCTION_CHECKPOINT_SHA256})"
        )

    @classmethod
    def inspect_checkpoint(cls, checkpoint_path: Path, verify_hash: bool = False) -> Dict[str, Any]:
        """Inspects and validates checkpoint payload without loading onto GPU."""
        if not checkpoint_path.exists():
            raise CheckpointError(f"Checkpoint file not found: {checkpoint_path}")

        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        sha256_hash = cls.compute_sha256(checkpoint_path)

        if verify_hash and sha256_hash != PRODUCTION_CHECKPOINT_SHA256:
            logger.warning(
                "Checkpoint SHA256 (%s) does not match production official (%s)",
                sha256_hash,
                PRODUCTION_CHECKPOINT_SHA256,
            )

        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointError(f"Failed to parse checkpoint payload at {checkpoint_path}: {e}")

        if not isinstance(payload, dict):
            raise CheckpointError(f"Corrupt checkpoint format in {checkpoint_path}: root must be a dict.")

        has_raw = "generator" in payload or "model_state_dict" in payload
        if not has_raw:
            raise CheckpointError(f"Checkpoint missing 'generator' state dictionary: {checkpoint_path}")

        has_ema = "ema_generator" in payload

        metadata = {
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "file_size_mb": round(file_size_mb, 2),
            "sha256": sha256_hash,
            "is_official": sha256_hash == PRODUCTION_CHECKPOINT_SHA256,
            "global_step": payload.get("global_step", "N/A"),
            "img_size": payload.get("img_size", 64),
            "latent_dim": payload.get("latent_dim", 512),
            "style_dim": payload.get("style_dim", 512),
            "has_raw_weights": has_raw,
            "has_ema_weights": has_ema,
            "format_version": payload.get("format_version", "1.0"),
            "model_name": payload.get("model_name", "StyleForge"),
        }
        return metadata
