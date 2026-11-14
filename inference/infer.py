"""StyleForge High-Performance Production Inference Engine (Compatibility Layer).

Provides a modular, backward-compatible Python API for:
- Seed-controlled photographic face generation
- Truncation trick modulation (psi)
- Raw vs EMA parameter selection
- Latent space interpolation (SLERP in Z, LERP in W)
- Multi-scale style mixing
- Optional noise scale modulation (e.g. 0.25 Smooth Mode)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image
import torch

from core.engine.config import (
    GenerationConfig,
    InferenceMode,
    InterpolationConfig,
    StyleMixConfig,
)
from core.engine.inference_engine import InferenceEngine
from core.engine.postprocessor import ImagePostProcessor
from models.style_generator import StyleGenerator

# Export utility functions for backward compatibility
tensor_to_pil = ImagePostProcessor.tensor_to_pil
make_grid = ImagePostProcessor.make_grid


class StyleForgeInference:
    """Production inference wrapper for StyleForge Style-Based Generator."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        try:
            self._engine = InferenceEngine(checkpoint_path=checkpoint_path, device=device)
        except Exception as e:
            # Match the legacy FileNotFoundError format expected by tests if missing
            from core.engine.config import CheckpointError
            if isinstance(e, CheckpointError):
                raise FileNotFoundError(
                    f"StyleForge checkpoint not found: {checkpoint_path}.\n"
                    f"Expected path: checkpoints/styleforge_final_model.pt\n"
                    f"Installation: run python scripts/export_final_model.py"
                ) from e
            raise

        self.device = self._engine.device
        self.checkpoint_path = self._engine.checkpoint_path
        self.raw_generator = self._engine.raw_generator
        self.ema_generator = self._engine.ema_generator
        self.metadata = {
            "checkpoint_name": self.checkpoint_path.name,
            "global_step": self._engine.runtime_meta.get("global_step", "N/A"),
            "img_size": self._engine.runtime_meta.get("img_size", 64),
            "latent_dim": self._engine.runtime_meta.get("latent_dim", 512),
            "device": str(self.device),
        }

    def _get_active_model(self, use_ema: bool = False) -> StyleGenerator:
        return self._engine._get_active_generator(use_ema)

    @torch.inference_mode()
    def generate(
        self,
        seed: Optional[int] = None,
        num_samples: int = 1,
        truncation_psi: float = 0.8,
        use_ema: bool = False,
        noise_scale: float = 1.0,
    ) -> List[Image.Image]:
        """Generate one or more faces from random or fixed seeds."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        if truncation_psi < 0.0:
            raise ValueError(f"truncation_psi must be >= 0.0, got {truncation_psi}")

        config = GenerationConfig(
            seed=seed,
            num_samples=num_samples,
            truncation_psi=truncation_psi,
            use_ema=use_ema,
            noise_scale=noise_scale,
            mode=InferenceMode.CUSTOM if noise_scale != 1.0 else InferenceMode.STANDARD,
        )
        images, _ = self._engine.generate(config)
        return images

    @torch.inference_mode()
    def interpolate(
        self,
        seed_a: int,
        seed_b: int,
        steps: int = 8,
        space: str = "W",
        truncation_psi: float = 0.8,
        use_ema: bool = False,
        noise_scale: float = 1.0,
    ) -> List[Image.Image]:
        """Generate smooth continuous interpolation sequence between two seeds."""
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")
        if space.upper() not in ("W", "Z"):
            raise ValueError(f"space must be 'W' or 'Z', got '{space}'")
        if truncation_psi < 0.0:
            raise ValueError(f"truncation_psi must be >= 0.0, got {truncation_psi}")

        config = InterpolationConfig(
            seed_a=seed_a,
            seed_b=seed_b,
            steps=steps,
            space=space,
            truncation_psi=truncation_psi,
            use_ema=use_ema,
            noise_scale=noise_scale,
        )
        frames, _ = self._engine.interpolate(config)
        return frames

    @torch.inference_mode()
    def mix_styles(
        self,
        seed_a: int,
        seed_b: int,
        coarse_from_b: bool = True,
        truncation_psi: float = 0.8,
        use_ema: bool = False,
        noise_scale: float = 1.0,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Mix coarse structure from one seed with fine appearance from another."""
        if truncation_psi < 0.0:
            raise ValueError(f"truncation_psi must be >= 0.0, got {truncation_psi}")

        config = StyleMixConfig(
            seed_a=seed_a,
            seed_b=seed_b,
            coarse_from_b=coarse_from_b,
            truncation_psi=truncation_psi,
            use_ema=use_ema,
            noise_scale=noise_scale,
        )
        img_a, img_b, img_mixed, _ = self._engine.mix_styles(config)
        return img_a, img_b, img_mixed
