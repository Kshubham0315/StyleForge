"""Production-Grade Inference Engine for StyleForge Generative Models."""

import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from core.engine.checkpoint_manager import CheckpointManager
from core.engine.config import (
    GenerationConfig,
    InferenceError,
    InferenceMode,
    InterpolationConfig,
    StyleMixConfig,
)
from core.engine.device_manager import DeviceManager
from core.engine.model_loader import ModelLoader
from core.engine.postprocessor import ImagePostProcessor
from models.style_generator import StyleGenerator

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Thread-safe, high-stability production inference engine for StyleForge."""

    _instance: Optional["InferenceEngine"] = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        verify_hash: bool = False,
    ):
        self.device = DeviceManager.resolve_device(device if isinstance(device, str) else (str(device) if device else None))
        self.checkpoint_path = CheckpointManager.resolve_checkpoint(checkpoint_path)
        self.checkpoint_meta = CheckpointManager.inspect_checkpoint(self.checkpoint_path, verify_hash=verify_hash)
        
        # Load models
        self.raw_generator, self.ema_generator, self.runtime_meta = ModelLoader.load_generators(
            self.checkpoint_path,
            self.device,
        )
        self._inference_lock = threading.Lock()
        logger.info("StyleForge InferenceEngine successfully initialized on %s", self.device)

    @classmethod
    def get_instance(
        cls,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "InferenceEngine":
        """Singleton accessor for shared backend service."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls(checkpoint_path=checkpoint_path, device=device)
            return cls._instance

    def _get_active_generator(self, use_ema: bool = False) -> StyleGenerator:
        return self.ema_generator if use_ema else self.raw_generator

    def _build_noise_tensors(
        self,
        batch_size: int,
        noise_scale: float,
        synthesis: Any,
        generator_rng: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """Builds resolution-aligned spatial noise tensors scaled by noise_scale."""
        noise_list = []
        # Initial 4x4 block (2 noise injection slots)
        for _ in range(2):
            if noise_scale == 0.0:
                noise_list.append(torch.zeros(batch_size, 1, 4, 4, device=self.device))
            else:
                n = torch.randn(batch_size, 1, 4, 4, device=self.device, generator=generator_rng)
                noise_list.append(n * noise_scale)

        # Subsequent resolution stages (8, 16, 32, 64, ...)
        for res in synthesis.resolutions:
            for _ in range(2):
                if noise_scale == 0.0:
                    noise_list.append(torch.zeros(batch_size, 1, res, res, device=self.device))
                else:
                    n = torch.randn(batch_size, 1, res, res, device=self.device, generator=generator_rng)
                    noise_list.append(n * noise_scale)
        return noise_list

    def _apply_controlled_adain(self, adain_module: Any, feat: torch.Tensor, style_vec: torch.Tensor, clamp: bool) -> torch.Tensor:
        """Applies AdaIN with optional non-negative scale clamping."""
        mean = feat.mean(dim=[2, 3], keepdim=True)
        var = feat.var(dim=[2, 3], keepdim=True, unbiased=False)
        normalized = (feat - mean) / torch.sqrt(var + adain_module.eps)

        style = adain_module.style_mod(style_vec)
        scale, bias = style.chunk(2, dim=1)
        scale = (scale + 1.0).unsqueeze(-1).unsqueeze(-1)
        if clamp:
            scale = torch.clamp(scale, min=0.0)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return normalized * scale + bias

    @torch.inference_mode()
    def _forward_synthesis_controlled(
        self,
        generator: StyleGenerator,
        styles: torch.Tensor,
        noise_scale: float,
        clamp_adain: bool,
        generator_rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Executes synthesis network with explicit noise scaling and optional clamping."""
        synthesis = generator.synthesis
        batch_size = styles.size(0)

        # Fast path if standard defaults (noise_scale=1.0, clamp_adain=False)
        if noise_scale == 1.0 and not clamp_adain:
            return synthesis(styles)

        noise_list = self._build_noise_tensors(batch_size, noise_scale, synthesis, generator_rng)

        # 4x4 initial constant
        x = synthesis.constant_input.repeat(batch_size, 1, 1, 1)
        x = synthesis.initial_noise(x, noise=noise_list[0])
        x = self._apply_controlled_adain(synthesis.initial_adain, x, styles[:, 0, :], clamp_adain)

        # 4x4 initial block
        x = synthesis.initial_block.conv(x)
        x = synthesis.initial_block.noise(x, noise=noise_list[1])
        x = synthesis.initial_block.act(x)
        x = self._apply_controlled_adain(synthesis.initial_block.adain, x, styles[:, 1, :], clamp_adain)

        style_idx = 2
        noise_idx = 2
        for res in synthesis.resolutions:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            b1 = synthesis.blocks[f"res_{res}_b1"]
            b2 = synthesis.blocks[f"res_{res}_b2"]

            s1 = styles[:, style_idx, :]
            s2 = styles[:, style_idx + 1, :]
            style_idx += 2

            n_b1 = noise_list[noise_idx]
            n_b2 = noise_list[noise_idx + 1]
            noise_idx += 2

            # Block 1
            x = b1.conv(x)
            x = b1.noise(x, noise=n_b1)
            x = b1.act(x)
            x = self._apply_controlled_adain(b1.adain, x, s1, clamp_adain)

            # Block 2
            x = b2.conv(x)
            x = b2.noise(x, noise=n_b2)
            x = b2.act(x)
            x = self._apply_controlled_adain(b2.adain, x, s2, clamp_adain)

            if f"res_{res}" in synthesis.attn:
                x = synthesis.attn[f"res_{res}"](x)

        return synthesis.to_rgb(x)

    @torch.inference_mode()
    def generate(self, config: GenerationConfig) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Generates synthetic faces from config specification under thread-safe concurrency lock.

        Returns:
            Tuple of (list_of_pil_images, metadata_dict)
        """
        with self._inference_lock:
            start_time = time.perf_counter()
            generator = self._get_active_generator(config.use_ema)

            # Deterministic generator RNG
            cpu_rng = None
            if config.seed is not None:
                torch.manual_seed(config.seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(config.seed)
                cpu_rng = torch.Generator(device=self.device)
                cpu_rng.manual_seed(config.seed)

            # Sample latent z
            z = torch.randn(
                config.num_samples,
                generator.latent_dim,
                device=self.device,
                generator=cpu_rng,
            )

            # Map to W-space
            w = generator.mapping(z)
            styles = w.unsqueeze(1).repeat(1, generator.num_style_layers, 1)

            # Truncation trick in W-space
            if config.truncation_psi < 1.0 and not torch.all(generator.w_avg == 0):
                w_avg = generator.w_avg.view(1, 1, -1)
                styles = w_avg + config.truncation_psi * (styles - w_avg)

            # Resolve effective noise scale based on mode
            effective_noise = config.noise_scale
            if config.mode == InferenceMode.SMOOTH:
                effective_noise = 0.25
            elif config.mode == InferenceMode.STANDARD:
                effective_noise = 1.0

            # Forward pass
            out_tensor = self._forward_synthesis_controlled(
                generator=generator,
                styles=styles,
                noise_scale=effective_noise,
                clamp_adain=config.clamp_adain,
                generator_rng=cpu_rng,
            )

            DeviceManager.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            pil_images = ImagePostProcessor.batch_to_pil(out_tensor)

            telemetry = {
                "num_samples": config.num_samples,
                "seed": config.seed,
                "truncation_psi": config.truncation_psi,
                "mode": config.mode.value if isinstance(config.mode, InferenceMode) else str(config.mode),
                "noise_scale": effective_noise,
                "use_ema": config.use_ema,
                "clamp_adain": config.clamp_adain,
                "latency_ms": round(elapsed_ms, 2),
                "latency_per_face_ms": round(elapsed_ms / config.num_samples, 2),
                "device": str(self.device),
                "model_name": self.checkpoint_meta.get("model_name", "StyleForge"),
                "checkpoint_sha256": self.checkpoint_meta.get("sha256"),
                "resolution": f"{self.checkpoint_meta.get('img_size', 64)}x{self.checkpoint_meta.get('img_size', 64)}",
            }

            return pil_images, telemetry

    @torch.inference_mode()
    def interpolate(self, config: InterpolationConfig) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Generates continuous latent interpolation between two seed identities."""
        with self._inference_lock:
            start_time = time.perf_counter()
            generator = self._get_active_generator(config.use_ema)

            # Sample endpoints
            torch.manual_seed(config.seed_a)
            za = torch.randn(1, generator.latent_dim, device=self.device)
            torch.manual_seed(config.seed_b)
            zb = torch.randn(1, generator.latent_dim, device=self.device)

            alphas = np.linspace(0.0, 1.0, config.steps)
            frames = []

            if config.space == "W":
                wa = generator.mapping(za)
                wb = generator.mapping(zb)
                if config.truncation_psi < 1.0 and not torch.all(generator.w_avg == 0):
                    wa = generator.w_avg + config.truncation_psi * (wa - generator.w_avg)
                    wb = generator.w_avg + config.truncation_psi * (wb - generator.w_avg)

                for a in alphas:
                    w_interp = (1.0 - a) * wa + a * wb
                    styles = w_interp.unsqueeze(1).repeat(1, generator.num_style_layers, 1)
                    img_t = self._forward_synthesis_controlled(
                        generator=generator,
                        styles=styles,
                        noise_scale=config.noise_scale,
                        clamp_adain=False,
                    )[0]
                    frames.append(ImagePostProcessor.tensor_to_pil(img_t))
            else:
                za_norm = za / za.norm(dim=1, keepdim=True)
                zb_norm = zb / zb.norm(dim=1, keepdim=True)
                dot = (za_norm * zb_norm).sum().clamp(-0.9999, 0.9999)
                omega = torch.acos(dot)
                sin_omega = torch.sin(omega)

                for a in alphas:
                    z_interp = (torch.sin((1.0 - a) * omega) / sin_omega) * za + (torch.sin(a * omega) / sin_omega) * zb
                    w_interp = generator.mapping(z_interp)
                    styles = w_interp.unsqueeze(1).repeat(1, generator.num_style_layers, 1)
                    if config.truncation_psi < 1.0 and not torch.all(generator.w_avg == 0):
                        w_avg = generator.w_avg.view(1, 1, -1)
                        styles = w_avg + config.truncation_psi * (styles - w_avg)

                    img_t = self._forward_synthesis_controlled(
                        generator=generator,
                        styles=styles,
                        noise_scale=config.noise_scale,
                        clamp_adain=False,
                    )[0]
                    frames.append(ImagePostProcessor.tensor_to_pil(img_t))

            DeviceManager.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            meta = {
                "steps": config.steps,
                "space": config.space,
                "seed_a": config.seed_a,
                "seed_b": config.seed_b,
                "latency_ms": round(elapsed_ms, 2),
            }
            return frames, meta

    @torch.inference_mode()
    def mix_styles(self, config: StyleMixConfig) -> Tuple[Image.Image, Image.Image, Image.Image, Dict[str, Any]]:
        """Multi-scale style mixing between two seed identities."""
        with self._inference_lock:
            start_time = time.perf_counter()
            generator = self._get_active_generator(config.use_ema)

            torch.manual_seed(config.seed_a)
            za = torch.randn(1, generator.latent_dim, device=self.device)
            torch.manual_seed(config.seed_b)
            zb = torch.randn(1, generator.latent_dim, device=self.device)

            wa = generator.mapping(za)
            wb = generator.mapping(zb)

            if config.truncation_psi < 1.0 and not torch.all(generator.w_avg == 0):
                wa = generator.w_avg + config.truncation_psi * (wa - generator.w_avg)
                wb = generator.w_avg + config.truncation_psi * (wb - generator.w_avg)

            num_layers = generator.num_style_layers
            split = num_layers // 2

            if config.coarse_from_b:
                styles = [wb] * split + [wa] * (num_layers - split)
            else:
                styles = [wa] * split + [wb] * (num_layers - split)

            styles_mixed = torch.stack(styles, dim=1)  # (1, L, D)
            styles_a = wa.unsqueeze(1).repeat(1, num_layers, 1)
            styles_b = wb.unsqueeze(1).repeat(1, num_layers, 1)

            t_a = self._forward_synthesis_controlled(generator, styles_a, config.noise_scale, False)[0]
            t_b = self._forward_synthesis_controlled(generator, styles_b, config.noise_scale, False)[0]
            t_mix = self._forward_synthesis_controlled(generator, styles_mixed, config.noise_scale, False)[0]

            DeviceManager.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            img_a = ImagePostProcessor.tensor_to_pil(t_a)
            img_b = ImagePostProcessor.tensor_to_pil(t_b)
            img_mixed = ImagePostProcessor.tensor_to_pil(t_mix)

            meta = {
                "seed_a": config.seed_a,
                "seed_b": config.seed_b,
                "coarse_from_b": config.coarse_from_b,
                "latency_ms": round(elapsed_ms, 2),
            }
            return img_a, img_b, img_mixed, meta
