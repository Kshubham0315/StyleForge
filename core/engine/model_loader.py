"""Model instantiation, weight loading, and state dictionary validation."""

import logging
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn

from core.engine.config import ModelLoadError
from models.style_generator import StyleGenerator

logger = logging.getLogger(__name__)


class ModelLoader:
    """Instantiates and loads StyleForge Generator models with defensive integrity checks."""

    @classmethod
    def load_generators(
        cls,
        checkpoint_path: Path,
        device: torch.device,
    ) -> Tuple[StyleGenerator, StyleGenerator, Dict[str, any]]:
        """Loads Raw and EMA generator models into evaluation mode.

        Returns:
            (raw_generator, ema_generator, metadata_dict)
        """
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ModelLoadError(f"Cannot load checkpoint binary from {checkpoint_path}: {e}")

        if not isinstance(payload, dict):
            raise ModelLoadError(f"Invalid checkpoint at {checkpoint_path}: expected dictionary payload.")

        raw_state = payload.get("generator") or payload.get("model_state_dict")
        if raw_state is None:
            raise ModelLoadError(f"Invalid checkpoint at {checkpoint_path}: missing generator state dictionary.")

        latent_dim = payload.get("latent_dim", 512)
        style_dim = payload.get("style_dim", 512)
        img_size = payload.get("img_size", 64)
        base_channels = payload.get("base_channels", 512)
        max_channels = payload.get("max_channels", 512)

        # 1. Raw Generator
        raw_gen = StyleGenerator(
            latent_dim=latent_dim,
            style_dim=style_dim,
            img_size=img_size,
            base_channels=base_channels,
            max_channels=max_channels,
        )
        cleaned_raw = {k.replace("module.", ""): v for k, v in raw_state.items()}
        raw_gen.load_state_dict(cleaned_raw)
        raw_gen.eval()
        raw_gen.requires_grad_(False)
        raw_gen.to(device)

        # 2. EMA Generator (if present, else fallback to raw weights)
        ema_gen = StyleGenerator(
            latent_dim=latent_dim,
            style_dim=style_dim,
            img_size=img_size,
            base_channels=base_channels,
            max_channels=max_channels,
        )
        ema_state = payload.get("ema_generator")
        if ema_state is not None:
            cleaned_ema = {k.replace("module.", ""): v for k, v in ema_state.items()}
            ema_gen.load_state_dict(cleaned_ema)
        else:
            logger.info("EMA generator weights not found in checkpoint; falling back to raw state dict.")
            ema_gen.load_state_dict(cleaned_raw)

        ema_gen.eval()
        ema_gen.requires_grad_(False)
        ema_gen.to(device)

        metadata = {
            "checkpoint_name": checkpoint_path.name,
            "global_step": payload.get("global_step", "N/A"),
            "img_size": img_size,
            "latent_dim": latent_dim,
            "style_dim": style_dim,
            "device": str(device),
            "has_ema": ema_state is not None,
        }

        logger.info(
            "Successfully loaded StyleForge models from %s (step: %s, size: %dx%d) on %s",
            checkpoint_path.name,
            metadata["global_step"],
            img_size,
            img_size,
            device,
        )
        return raw_gen, ema_gen, metadata
