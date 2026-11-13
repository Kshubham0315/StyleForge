"""StyleForge Style-Based Generator.

Implements the core StyleGAN generative concepts:
- Mapping Network (f: Z -> W) with PixelNorm, lr_mul=0.01, and equalized linear layers
- Synthesis Network starting from a learned constant 4x4 spatial tensor
- Adaptive Instance Normalization (AdaIN) for style modulation
- Per-channel stochastic noise injection
- Native support for style mixing across resolution layers
- Truncation trick in W-space
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    PixelNorm,
    EqualizedLinear,
    EqualizedConv2d,
    NoiseInjection,
    AdaIN,
    SelfAttention2d,
)


class MappingNetwork(nn.Module):
    """Mapping Network f : Z -> W.

    Maps Gaussian latent vector z to intermediate latent space W.
    Disentangles factors of variation and enables linear style manipulation.
    Uses lr_mul=0.01 to stabilize intermediate manifold convergence.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        num_layers: int = 8,
        lr_mul: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.pixel_norm = PixelNorm()

        layers = []
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else style_dim
            layers.append(EqualizedLinear(in_dim, style_dim, gain=2.0 ** 0.5, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Normalize input latent vector on unit hypersphere
        z = self.pixel_norm(z)
        return self.net(z)


class StyleBlock(nn.Module):
    """Synthesis block at a single resolution.

    Performs:
    1. Convolution
    2. Per-channel Noise Injection
    3. LeakyReLU activation
    4. AdaIN style modulation
    """

    def __init__(self, in_channels: int, out_channels: int, style_dim: int):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise = NoiseInjection(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.adain = AdaIN(style_dim=style_dim, num_features=out_channels)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.noise(x, noise=noise)
        x = self.act(x)
        x = self.adain(x, w)
        return x


class SynthesisNetwork(nn.Module):
    """Synthesis Network g.

    Starts from a learned 4x4 constant tensor and progressively generates
    images via AdaIN-modulated convolutions and upsampling.
    """

    def __init__(
        self,
        style_dim: int = 512,
        out_channels: int = 3,
        img_size: int = 64,
        base_channels: int = 512,
        max_channels: int = 512,
        use_attention: bool = False,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.img_size = img_size
        self.use_attention = use_attention

        # Dynamic channel map with min channel clamp
        channel_map = {
            4: base_channels,
            8: base_channels,
            16: max(16, min(max_channels, base_channels // 2)),
            32: max(16, min(max_channels, base_channels // 4)),
            64: max(16, min(max_channels, base_channels // 8)),
            128: max(16, min(max_channels, base_channels // 16)),
        }

        # Learned constant 4x4 input
        self.const_channels = channel_map[4]
        self.constant_input = nn.Parameter(torch.randn(1, self.const_channels, 4, 4))

        # 4x4 initial block
        self.initial_noise = NoiseInjection(self.const_channels)
        self.initial_adain = AdaIN(style_dim, self.const_channels)
        self.initial_block = StyleBlock(self.const_channels, self.const_channels, style_dim)

        # Multi-scale blocks: 8, 16, 32, 64, ... up to img_size
        self.resolutions = []
        r = 8
        while r <= img_size:
            self.resolutions.append(r)
            r *= 2

        self.blocks = nn.ModuleDict()
        self.attn = nn.ModuleDict()

        in_c = self.const_channels
        # Number of style injection points:
        # 4x4 has 2 styles (initial_adain + initial_block.adain)
        # each subsequent resolution has 2 styles
        # total layers = 2 + 2 * len(resolutions)
        self.num_style_layers = 2 + 2 * len(self.resolutions)

        for res in self.resolutions:
            out_c = channel_map[res]
            b1 = StyleBlock(in_c, out_c, style_dim)
            b2 = StyleBlock(out_c, out_c, style_dim)
            self.blocks[f"res_{res}_b1"] = b1
            self.blocks[f"res_{res}_b2"] = b2

            if use_attention and res == 16:
                self.attn[f"res_{res}"] = SelfAttention2d(out_c)
            in_c = out_c

        # ToRGB projection from final feature map
        self.to_rgb = nn.Sequential(
            EqualizedConv2d(in_c, out_channels, kernel_size=1, padding=0, gain=1.0),
            nn.Tanh(),  # Output range [-1, 1]
        )

    def forward(
        self,
        styles: Union[torch.Tensor, List[torch.Tensor]],
        noise_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Forward pass through synthesis network.

        Args:
            styles: Either a single tensor (B, style_dim) or (B, num_layers, style_dim),
                    or a list of (B, style_dim) tensors for style mixing.
            noise_list: Optional list of explicit spatial noise tensors.
        """
        # Expand styles to (num_layers, B, style_dim)
        if isinstance(styles, list):
            style_layers = styles
        elif styles.dim() == 2:
            style_layers = [styles] * self.num_style_layers
        elif styles.dim() == 3:
            # (B, L, style_dim) -> list of L tensors (B, style_dim)
            style_layers = [styles[:, i, :] for i in range(styles.size(1))]
        else:
            raise ValueError(f"Invalid styles tensor shape: {styles.shape}")

        if len(style_layers) < self.num_style_layers:
            # Pad with the last style if fewer were supplied
            last = style_layers[-1]
            style_layers = style_layers + [last] * (self.num_style_layers - len(style_layers))

        batch_size = style_layers[0].size(0)

        # 4x4 initial constant
        x = self.constant_input.repeat(batch_size, 1, 1, 1)
        n0 = noise_list[0] if noise_list is not None and len(noise_list) > 0 else None
        x = self.initial_noise(x, noise=n0)
        x = self.initial_adain(x, style_layers[0])

        n1 = noise_list[1] if noise_list is not None and len(noise_list) > 1 else None
        x = self.initial_block(x, style_layers[1], noise=n1)

        style_idx = 2
        noise_idx = 2
        for res in self.resolutions:
            # Upsample 2x
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

            b1 = self.blocks[f"res_{res}_b1"]
            b2 = self.blocks[f"res_{res}_b2"]

            s1 = style_layers[style_idx]
            s2 = style_layers[style_idx + 1]
            style_idx += 2

            n_b1 = noise_list[noise_idx] if noise_list is not None and len(noise_list) > noise_idx else None
            n_b2 = noise_list[noise_idx + 1] if noise_list is not None and len(noise_list) > noise_idx + 1 else None
            noise_idx += 2

            x = b1(x, s1, noise=n_b1)
            x = b2(x, s2, noise=n_b2)

            if f"res_{res}" in self.attn:
                x = self.attn[f"res_{res}"](x)

        return self.to_rgb(x)


class StyleGenerator(nn.Module):
    """Complete StyleForge Style-Based Generator.

    Composes Mapping Network + Synthesis Network with built-in:
    - Truncation trick in W-space
    - Style mixing support across layers
    - Running average style tracking
    """

    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        num_mapping_layers: int = 8,
        img_channels: int = 3,
        img_size: int = 64,
        base_channels: int = 512,
        max_channels: int = 512,
        use_attention: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.img_size = img_size

        self.mapping = MappingNetwork(
            latent_dim=latent_dim,
            style_dim=style_dim,
            num_layers=num_mapping_layers,
            lr_mul=0.01,
        )
        self.synthesis = SynthesisNetwork(
            style_dim=style_dim,
            out_channels=img_channels,
            img_size=img_size,
            base_channels=base_channels,
            max_channels=max_channels,
            use_attention=use_attention,
        )

        self.num_style_layers = self.synthesis.num_style_layers

        # Buffer to track running mean style vector w_avg for truncation trick
        self.register_buffer("w_avg", torch.zeros(style_dim))

    def update_w_avg(self, w: torch.Tensor, momentum: float = 0.995):
        """Update running average of W for truncation trick."""
        with torch.no_grad():
            batch_mean = w.detach().mean(dim=0)
            if torch.all(self.w_avg == 0):
                self.w_avg.copy_(batch_mean)
            else:
                self.w_avg.mul_(momentum).add_(batch_mean, alpha=1.0 - momentum)

    def forward(
        self,
        z: torch.Tensor,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        style_mixing_prob: float = 0.0,
        z2: Optional[torch.Tensor] = None,
        noise_list: Optional[List[Optional[torch.Tensor]]] = None,
        update_w_avg: bool = False,
    ) -> torch.Tensor:
        """Forward generation pass.

        Args:
            z: Latent vector (B, latent_dim).
            truncation_psi: Truncation coefficient in [0, 1]. 1.0 = no truncation.
            truncation_cutoff: Number of initial layers to apply truncation to.
            style_mixing_prob: Probability of mixing styles with z2 during training.
            z2: Optional second latent vector for style mixing.
            noise_list: Optional explicit noise tensors.
            update_w_avg: If True, update running w_avg estimate.
        """
        w = self.mapping(z)

        if update_w_avg and self.training:
            self.update_w_avg(w)

        # Style mixing logic
        if (style_mixing_prob > 0 and torch.rand(1).item() < style_mixing_prob) or (z2 is not None):
            if z2 is None:
                z2 = torch.randn_like(z)
            w2 = self.mapping(z2)

            # Random crossover layer
            crossover = torch.randint(1, self.num_style_layers, (1,)).item()
            styles = []
            for i in range(self.num_style_layers):
                styles.append(w if i < crossover else w2)
            styles = torch.stack(styles, dim=1)  # (B, L, D)
        else:
            styles = w.unsqueeze(1).repeat(1, self.num_style_layers, 1)  # (B, L, D)

        # Truncation trick: w' = w_avg + psi * (w - w_avg)
        if truncation_psi < 1.0:
            cutoff = truncation_cutoff if truncation_cutoff is not None else self.num_style_layers
            w_avg_expanded = self.w_avg.view(1, 1, -1)
            styles[:, :cutoff, :] = w_avg_expanded + truncation_psi * (styles[:, :cutoff, :] - w_avg_expanded)

        return self.synthesis(styles, noise_list=noise_list)

    def generate_from_w(
        self,
        w: torch.Tensor,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate directly from intermediate style tensor w.

        Args:
            w: Style tensor of shape (B, style_dim) or (B, num_layers, style_dim).
            truncation_psi: Truncation coefficient.
            truncation_cutoff: Max layer to apply truncation to.
        """
        if w.dim() == 2:
            styles = w.unsqueeze(1).repeat(1, self.num_style_layers, 1)
        else:
            styles = w.clone()

        if truncation_psi < 1.0:
            cutoff = truncation_cutoff if truncation_cutoff is not None else self.num_style_layers
            w_avg_expanded = self.w_avg.view(1, 1, -1)
            styles[:, :cutoff, :] = w_avg_expanded + truncation_psi * (styles[:, :cutoff, :] - w_avg_expanded)

        return self.synthesis(styles)
