"""Latent Space Exploration and Style Manipulation for StyleForge.

Features:
- Spherical Linear Interpolation (SLERP) in Z-space
- Linear Interpolation (LERP) in W-space
- Truncation trick visualization across various psi values
- Multilevel Style Mixing Matrix (Coarse, Middle, Fine resolution control)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


def slerp(val: float, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation between two vectors."""
    low_norm = low / (torch.norm(low, dim=-1, keepdim=True) + eps)
    high_norm = high / (torch.norm(high, dim=-1, keepdim=True) + eps)
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    # When vectors are nearly colinear, fall back to linear interpolation
    if torch.all(torch.abs(so) < eps):
        return (1.0 - val) * low + val * high

    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def lerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between two vectors."""
    return (1.0 - val) * low + val * high


class LatentExplorer:
    """Exploration and visualization toolkit for generative latent spaces."""

    def __init__(self, generator: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device).eval()
        self.is_style_gen = hasattr(generator, "mapping") and hasattr(generator, "synthesis")

    @torch.no_grad()
    def interpolate_z(
        self,
        z_start: Optional[torch.Tensor] = None,
        z_end: Optional[torch.Tensor] = None,
        steps: int = 10,
        method: str = "slerp",
        latent_dim: int = 512,
        truncation_psi: float = 0.8,
    ) -> torch.Tensor:
        """Interpolates between two latent vectors in Z-space."""
        if z_start is None:
            z_start = torch.randn(1, latent_dim, device=self.device)
        if z_end is None:
            z_end = torch.randn(1, latent_dim, device=self.device)

        interp_images = []
        for alpha in np.linspace(0.0, 1.0, steps):
            if method == "slerp":
                z_interp = slerp(float(alpha), z_start, z_end)
            else:
                z_interp = lerp(float(alpha), z_start, z_end)

            if self.is_style_gen:
                img = self.generator(z_interp, truncation_psi=truncation_psi)
            else:
                img = self.generator(z_interp)

            interp_images.append(img)

        # Shape (steps, C, H, W)
        return torch.cat(interp_images, dim=0)

    @torch.no_grad()
    def interpolate_w(
        self,
        z_start: Optional[torch.Tensor] = None,
        z_end: Optional[torch.Tensor] = None,
        steps: int = 10,
        latent_dim: int = 512,
        truncation_psi: float = 0.8,
    ) -> torch.Tensor:
        """Interpolates linearly in intermediate style space W."""
        if not self.is_style_gen:
            raise ValueError("W-space interpolation requires a StyleGenerator with mapping network.")

        if z_start is None:
            z_start = torch.randn(1, latent_dim, device=self.device)
        if z_end is None:
            z_end = torch.randn(1, latent_dim, device=self.device)

        w_start = self.generator.mapping(z_start)
        w_end = self.generator.mapping(z_end)

        interp_images = []
        for alpha in np.linspace(0.0, 1.0, steps):
            w_interp = lerp(float(alpha), w_start, w_end)
            img = self.generator.generate_from_w(w_interp, truncation_psi=truncation_psi)
            interp_images.append(img)

        return torch.cat(interp_images, dim=0)

    @torch.no_grad()
    def truncation_sweep(
        self,
        z: Optional[torch.Tensor] = None,
        psi_values: Optional[List[float]] = None,
        latent_dim: int = 512,
    ) -> torch.Tensor:
        """Visualizes how the truncation psi parameter influences image quality and fidelity."""
        if not self.is_style_gen:
            raise ValueError("Truncation sweep requires a StyleGenerator.")

        if z is None:
            z = torch.randn(1, latent_dim, device=self.device)
        if psi_values is None:
            psi_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        images = []
        for psi in psi_values:
            img = self.generator(z, truncation_psi=psi)
            images.append(img)

        return torch.cat(images, dim=0)

    @torch.no_grad()
    def style_mixing_matrix(
        self,
        num_src_a: int = 4,
        num_src_b: int = 4,
        coarse_layers: Tuple[int, int] = (0, 4),
        middle_layers: Tuple[int, int] = (4, 8),
        fine_layers: Tuple[int, int] = (8, 12),
        mix_mode: str = "coarse",
        latent_dim: int = 512,
        truncation_psi: float = 0.8,
    ) -> torch.Tensor:
        """Creates a style mixing grid where columns are Source B, rows are Source A,

        and intersecting cells inherit selected layers from Source B and the rest from Source A.
        mix_mode can be 'coarse' (pose/shape), 'middle' (features/hair), or 'fine' (color/texture).
        """
        if not self.is_style_gen:
            raise ValueError("Style mixing matrix requires a StyleGenerator.")

        z_a = torch.randn(num_src_a, latent_dim, device=self.device)
        z_b = torch.randn(num_src_b, latent_dim, device=self.device)

        w_a = self.generator.mapping(z_a)
        w_b = self.generator.mapping(z_b)

        num_layers = self.generator.num_style_layers

        # Target layer indices to override with Source B
        if mix_mode == "coarse":
            override_indices = set(range(coarse_layers[0], min(coarse_layers[1], num_layers)))
        elif mix_mode == "middle":
            override_indices = set(range(middle_layers[0], min(middle_layers[1], num_layers)))
        elif mix_mode == "fine":
            override_indices = set(range(fine_layers[0], min(fine_layers[1], num_layers)))
        else:
            override_indices = set(range(0, num_layers // 2))

        # Blank top-left corner
        dummy_corner = torch.zeros(1, 3, self.generator.img_size, self.generator.img_size, device=self.device)

        # Source B images (top header row)
        img_b = [self.generator.generate_from_w(w_b[j:j+1], truncation_psi=truncation_psi) for j in range(num_src_b)]
        row_header = [dummy_corner] + img_b

        all_rows = [torch.cat(row_header, dim=0)]

        for i in range(num_src_a):
            # Source A image (left column)
            img_a = self.generator.generate_from_w(w_a[i:i+1], truncation_psi=truncation_psi)
            curr_row = [img_a]

            for j in range(num_src_b):
                # Mixed style vector: (1, num_layers, style_dim)
                mixed_styles = []
                for l in range(num_layers):
                    if l in override_indices:
                        mixed_styles.append(w_b[j:j+1])
                    else:
                        mixed_styles.append(w_a[i:i+1])
                mixed_styles = torch.stack(mixed_styles, dim=1)

                mixed_img = self.generator.generate_from_w(mixed_styles, truncation_psi=truncation_psi)
                curr_row.append(mixed_img)

            all_rows.append(torch.cat(curr_row, dim=0))

        # Flatten grid
        full_grid = torch.cat(all_rows, dim=0)
        # Normalize to [0, 1]
        full_grid = (full_grid * 0.5 + 0.5).clamp(0.0, 1.0)
        return full_grid
