"""WGAN-GP Baseline Generator and Critic architectures.

Provides reference implementations following the Gulrajani et al. (2017)
'Improved Training of Wasserstein GANs' guidelines:
- Critic uses LayerNorm / no batch norm to avoid inter-sample coupling
- Configurable image resolutions (64x64 baseline, 128x128 extension)
- Clean functional gradient flow suitable for Wasserstein gradient penalty
"""

from typing import Optional
import torch
import torch.nn as nn
from models.common import SelfAttention2d


class WGANBaselineGenerator(nn.Module):
    """Classical convolutional generator for WGAN-GP baseline."""

    def __init__(
        self,
        latent_dim: int = 128,
        img_channels: int = 3,
        feature_dim: int = 64,
        img_size: int = 64,
        use_attention: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.use_attention = use_attention

        if img_size not in (64, 128):
            raise ValueError(f"Supported img_size values are 64 or 128, got {img_size}")

        # Initial projection: from latent_dim to 4x4 x (feature_dim * 8)
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True),
        )  # 4x4

        # 4x4 -> 8x8
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
        )

        # 8x8 -> 16x16
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
        )

        # Optional self-attention at 16x16
        self.attn = SelfAttention2d(feature_dim * 2) if use_attention else nn.Identity()

        # 16x16 -> 32x32
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
        )

        if img_size == 64:
            # 32x32 -> 64x64
            self.final_block = nn.Sequential(
                nn.ConvTranspose2d(feature_dim, img_channels, kernel_size=4, stride=2, padding=1, bias=True),
                nn.Tanh(),
            )
        else:
            # 32x32 -> 64x64
            self.block4 = nn.Sequential(
                nn.ConvTranspose2d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim // 2),
                nn.ReLU(True),
            )
            # 64x64 -> 128x128
            self.final_block = nn.Sequential(
                nn.ConvTranspose2d(feature_dim // 2, img_channels, kernel_size=4, stride=2, padding=1, bias=True),
                nn.Tanh(),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        x = self.init_block(z)
        x = self.block1(x)
        x = self.block2(x)
        x = self.attn(x)
        x = self.block3(x)
        if self.img_size == 128:
            x = self.block4(x)
        return self.final_block(x)


class WGANBaselineCritic(nn.Module):
    """Classical convolutional Critic for WGAN-GP baseline.

    Uses LayerNorm (or no normalization) in place of BatchNorm to preserve
    the 1-Lipschitz continuity constraint without cross-batch dependence.
    """

    def __init__(
        self,
        img_channels: int = 3,
        feature_dim: int = 64,
        img_size: int = 64,
        use_spectral_norm: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        self.img_size = img_size

        def conv_layer(in_c: int, out_c: int, stride: int = 2) -> nn.Module:
            conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=True)
            if use_spectral_norm:
                from torch.nn.utils.parametrizations import spectral_norm as sn
                return sn(conv)
            return conv

        # First layer never uses LayerNorm
        self.in_block = nn.Sequential(
            conv_layer(img_channels, feature_dim, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )  # 64->32 or 128->64

        if img_size == 128:
            self.extra_down = nn.Sequential(
                conv_layer(feature_dim, feature_dim * 2, stride=2),
                nn.InstanceNorm2d(feature_dim * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )  # 64->32
            c_curr = feature_dim * 2
        else:
            self.extra_down = nn.Identity()
            c_curr = feature_dim

        # 32 -> 16
        self.down1 = nn.Sequential(
            conv_layer(c_curr, feature_dim * 2, stride=2),
            nn.InstanceNorm2d(feature_dim * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.attn = SelfAttention2d(feature_dim * 2, spectral_norm=use_spectral_norm) if use_attention else nn.Identity()

        # 16 -> 8
        self.down2 = nn.Sequential(
            conv_layer(feature_dim * 2, feature_dim * 4, stride=2),
            nn.InstanceNorm2d(feature_dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 8 -> 4
        self.down3 = nn.Sequential(
            conv_layer(feature_dim * 4, feature_dim * 8, stride=2),
            nn.InstanceNorm2d(feature_dim * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 4x4 -> 1x1 scalar critic score
        self.out_conv = conv_layer(feature_dim * 8, 1, stride=1)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)
        if self.img_size == 128:
            x = self.extra_down(x)
        x = self.down1(x)
        x = self.attn(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.out_conv(x)
        x = self.final_pool(x)
        return x.view(x.size(0), 1)
