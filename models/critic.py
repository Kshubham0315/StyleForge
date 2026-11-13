"""StyleForge Critic (Discriminator) architecture.

Features:
- Downsampling convolutional backbone with LeakyReLU activations
- Minibatch Standard Deviation injection at 4x4 resolution to fight mode collapse
- Spectral Normalization toggleable on all convolutional and linear layers for ablations
- SAGAN Self-Attention block toggleable at 16x16 resolution
- Unbounded scalar Wasserstein score output
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import MinibatchStdDev, SelfAttention2d, EqualizedConv2d, EqualizedLinear


class ConvBlock(nn.Module):
    """Convolutional downsampling block with optional spectral normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.downsample = downsample

        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm as sn
            self.conv1 = sn(conv1)
            self.conv2 = sn(conv2)
        else:
            self.conv1 = conv1
            self.conv2 = conv2

        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Residual shortcut
        if downsample or in_channels != out_channels:
            shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if use_spectral_norm:
                from torch.nn.utils.parametrizations import spectral_norm as sn
                self.shortcut = sn(shortcut)
            else:
                self.shortcut = shortcut
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act1(self.conv1(x))
        h = self.act2(self.conv2(h))

        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            res = F.avg_pool2d(res, kernel_size=2, stride=2)

        res = self.shortcut(res)
        return (h + res) * (1.0 / (2.0 ** 0.5))


class StyleCritic(nn.Module):
    """Multi-scale Critic with Minibatch StdDev and optional Self-Attention / Spectral Norm."""

    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        base_channels: int = 64,
        max_channels: int = 512,
        use_minibatch_std: bool = True,
        use_spectral_norm: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_minibatch_std = use_minibatch_std
        self.use_spectral_norm = use_spectral_norm
        self.use_attention = use_attention

        # FromRGB initial projection
        conv_in = nn.Conv2d(img_channels, base_channels, kernel_size=1)
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm as sn
            self.from_rgb = sn(conv_in)
        else:
            self.from_rgb = conv_in

        # Multiscale downsampling:
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.blocks = nn.ModuleDict()

        curr_res = img_size
        c_in = base_channels

        # Build downsampling path down to 4x4
        while curr_res > 4:
            c_out = min(max_channels, c_in * 2)
            block = ConvBlock(
                in_channels=c_in,
                out_channels=c_out,
                downsample=True,
                use_spectral_norm=use_spectral_norm,
            )
            self.blocks[f"down_{curr_res}"] = block

            if use_attention and (curr_res // 2) == 16:
                self.blocks["attn_16"] = SelfAttention2d(c_out, spectral_norm=use_spectral_norm)

            c_in = c_out
            curr_res = curr_res // 2

        # Minibatch std dev at 4x4
        self.minibatch_std = MinibatchStdDev() if use_minibatch_std else nn.Identity()
        extra_c = 1 if use_minibatch_std else 0

        # Final 4x4 conv and scalar score
        final_conv = nn.Conv2d(c_in + extra_c, c_in, kernel_size=3, padding=1)
        final_fc = nn.Linear(c_in * 4 * 4, 1)

        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm as sn
            self.final_conv = sn(final_conv)
            self.final_fc = sn(final_fc)
        else:
            self.final_conv = final_conv
            self.final_fc = final_fc

        self.final_act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)

        curr_res = self.img_size
        while curr_res > 4:
            h = self.blocks[f"down_{curr_res}"](h)
            if self.use_attention and (curr_res // 2) == 16 and "attn_16" in self.blocks:
                h = self.blocks["attn_16"](h)
            curr_res = curr_res // 2

        # At 4x4 resolution:
        h = self.minibatch_std(h)
        h = self.final_act(self.final_conv(h))
        h = h.view(h.size(0), -1)
        score = self.final_fc(h)
        return score
