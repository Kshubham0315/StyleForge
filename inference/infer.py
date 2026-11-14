"""StyleForge High-Performance Production Inference Engine.

Provides an optimized, modular Python API for:
- Seed-controlled photographic face generation
- Truncation trick modulation (psi)
- Raw vs EMA parameter selection
- Latent space interpolation (SLERP in Z, LERP in W)
- Multi-scale style mixing
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from models.style_generator import StyleGenerator

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single (3, H, W) float tensor in [-1, 1] to a PIL Image."""
    np_img = ((tensor * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(np_img)

def make_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
    """Combine a list of PIL Images into a single grid."""
    n = len(images)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))
    return grid

class StyleForgeInference:
    """Production inference wrapper for StyleForge Style-Based Generator."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            candidates = [
                Path("checkpoints/styleforge_final_model.pt"),
                Path("experiments/phase3_convergence/checkpoints/checkpoint_final.pt"),
                Path("experiments/phase2_stylegan_adain/checkpoints/checkpoint_final.pt"),
            ]
            for c in candidates:
                if c.exists():
                    checkpoint_path = c
                    break

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found. Checked default paths: {candidates}")

        self.checkpoint_path = Path(checkpoint_path)
        self._load_models()

    def _load_models(self):
        """Loads both Raw and EMA generator models into evaluation mode."""
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        latent_dim = payload.get("latent_dim", 512)
        style_dim = payload.get("style_dim", 512)
        img_size = payload.get("img_size", 64)

        # 1. Raw Generator
        self.raw_generator = StyleGenerator(latent_dim=latent_dim, style_dim=style_dim, img_size=img_size)
        raw_state = payload.get("generator") or payload.get("model_state_dict")
        cleaned_raw = {k.replace("module.", ""): v for k, v in raw_state.items()}
        self.raw_generator.load_state_dict(cleaned_raw)
        self.raw_generator.to(self.device).eval()

        # 2. EMA Generator (if present, fallback to raw)
        self.ema_generator = StyleGenerator(latent_dim=latent_dim, style_dim=style_dim, img_size=img_size)
        ema_state = payload.get("ema_generator")
        if ema_state is not None:
            cleaned_ema = {k.replace("module.", ""): v for k, v in ema_state.items()}
            self.ema_generator.load_state_dict(cleaned_ema)
        else:
            self.ema_generator.load_state_dict(cleaned_raw)
        self.ema_generator.to(self.device).eval()

        self.metadata = {
            "checkpoint_name": self.checkpoint_path.name,
            "global_step": payload.get("global_step", "N/A"),
            "img_size": img_size,
            "latent_dim": latent_dim,
            "device": str(self.device),
        }

    def _get_active_model(self, use_ema: bool = False) -> StyleGenerator:
        return self.ema_generator if use_ema else self.raw_generator

    @torch.inference_mode()
    def generate(
        self,
        seed: Optional[int] = None,
        num_samples: int = 1,
        truncation_psi: float = 0.8,
        use_ema: bool = False,
    ) -> List[Image.Image]:
        """Generate one or more faces from random or fixed seeds."""
        model = self._get_active_model(use_ema)

        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        z = torch.randn(num_samples, model.latent_dim, device=self.device)
        imgs = model(z, truncation_psi=truncation_psi)

        pil_list = [tensor_to_pil(imgs[i]) for i in range(num_samples)]
        return pil_list

    @torch.inference_mode()
    def interpolate(
        self,
        seed_a: int,
        seed_b: int,
        steps: int = 8,
        space: str = "W",
        truncation_psi: float = 0.8,
        use_ema: bool = False,
    ) -> List[Image.Image]:
        """Generate smooth continuous interpolation sequence between two seeds."""
        model = self._get_active_model(use_ema)

        # Seed A
        torch.manual_seed(seed_a)
        za = torch.randn(1, model.latent_dim, device=self.device)

        # Seed B
        torch.manual_seed(seed_b)
        zb = torch.randn(1, model.latent_dim, device=self.device)

        alphas = np.linspace(0.0, 1.0, steps)
        frames = []

        if space.upper() == "W":
            # Linear Interpolation in intermediate style space W
            wa = model.mapping(za)
            wb = model.mapping(zb)
            if truncation_psi < 1.0:
                wa = model.w_avg + truncation_psi * (wa - model.w_avg)
                wb = model.w_avg + truncation_psi * (wb - model.w_avg)

            for a in alphas:
                w_interp = (1.0 - a) * wa + a * wb
                img = model.synthesis(w_interp)[0]
                frames.append(tensor_to_pil(img))
        else:
            # Spherical Linear Interpolation in latent space Z
            za_norm = za / za.norm(dim=1, keepdim=True)
            zb_norm = zb / zb.norm(dim=1, keepdim=True)
            dot = (za_norm * zb_norm).sum().clamp(-0.9999, 0.9999)
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)

            for a in alphas:
                z_interp = (torch.sin((1.0 - a) * omega) / sin_omega) * za + (torch.sin(a * omega) / sin_omega) * zb
                img = model(z_interp, truncation_psi=truncation_psi)[0]
                frames.append(tensor_to_pil(img))

        return frames

    @torch.inference_mode()
    def mix_styles(
        self,
        seed_a: int,
        seed_b: int,
        coarse_from_b: bool = True,
        truncation_psi: float = 0.8,
        use_ema: bool = False,
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Mix coarse structure from one seed with fine appearance from another."""
        model = self._get_active_model(use_ema)

        torch.manual_seed(seed_a)
        za = torch.randn(1, model.latent_dim, device=self.device)
        torch.manual_seed(seed_b)
        zb = torch.randn(1, model.latent_dim, device=self.device)

        wa = model.mapping(za)
        wb = model.mapping(zb)

        if truncation_psi < 1.0:
            wa = model.w_avg + truncation_psi * (wa - model.w_avg)
            wb = model.w_avg + truncation_psi * (wb - model.w_avg)

        num_layers = model.synthesis.num_style_layers
        split = num_layers // 2

        if coarse_from_b:
            # Coarse from B, Fine from A
            styles = [wb] * split + [wa] * (num_layers - split)
        else:
            # Coarse from A, Fine from B
            styles = [wa] * split + [wb] * (num_layers - split)

        img_a = tensor_to_pil(model.synthesis(wa)[0])
        img_b = tensor_to_pil(model.synthesis(wb)[0])
        img_mixed = tensor_to_pil(model.synthesis(styles)[0])

        return img_a, img_b, img_mixed
