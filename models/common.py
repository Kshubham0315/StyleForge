"""StyleForge Common Building Blocks.

Contains standard generative layers:
- PixelNorm
- EqualizedLinear / EqualizedConv2d (Equalized Learning Rate)
- Adaptive Instance Normalization (AdaIN)
- Per-channel Noise Injection
- Minibatch Standard Deviation
- Self-Attention 2D (SAGAN)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Pixel Normalization layer.

    Normalizes the feature vector in each pixel to unit length across channels.
    Prevents signal magnitudes from escalating out of control in early layers.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class EqualizedWeight(nn.Module):
    """Runtime weight scaling according to He (Kaiming) initialization.

    Preserves dynamic range and stabilizes gradient flow across different layers.
    """

    def __init__(self, weight: nn.Parameter, gain: float = 2.0 ** 0.5):
        super().__init__()
        self.gain = gain
        fan_in = weight.data[0][0].numel() if weight.dim() > 2 else weight.size(1)
        self.scale = gain / math.sqrt(fan_in)

    def forward(self, weight: nn.Parameter) -> torch.Tensor:
        return weight * self.scale


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, gain: float = 2.0 ** 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.scale = gain / math.sqrt(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale
        return F.linear(x, w, self.bias)


class EqualizedConv2d(nn.Module):
    """2D Convolutional layer with equalized learning rate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        gain: float = 2.0 ** 0.5,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        fan_in = in_channels * (kernel_size ** 2)
        self.scale = gain / math.sqrt(fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)


class NoiseInjection(nn.Module):
    """Stochastic noise injection layer.

    Injects learned per-channel scaled Gaussian noise into spatial feature maps
    to promote generation of fine stochastic details (e.g., hair strands, skin pores).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = torch.randn(batch_size, 1, height, width, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Normalizes input feature map x to zero-mean and unit variance, then modulates
    it using scale (y_s) and bias (y_b) derived from intermediate style vector w.
    """

    def __init__(self, style_dim: int, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        # Style modulation produces [scale, bias] for each feature channel
        self.style_mod = EqualizedLinear(style_dim, num_features * 2, gain=1.0)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Standard instance normalization (without learnable affine params)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Style modulation: shape (B, 2 * num_features)
        style = self.style_mod(w)
        style = style.view(-1, 2, self.num_features, 1, 1)
        scale = style[:, 0] + 1.0  # Center scale around 1.0
        bias = style[:, 1]

        return normalized * scale + bias


class MinibatchStdDev(nn.Module):
    """Minibatch Standard Deviation Layer.

    Computes standard deviation of features across the minibatch for each spatial location,
    averages over all features and locations, and appends the result as a constant feature map.
    Helps discriminator detect mode collapse.
    """

    def __init__(self, group_size: int = 4, eps: float = 1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        group_size = min(batch_size, self.group_size)

        # Reshape to (groups, group_size, channels, height, width)
        if batch_size % group_size == 0 and batch_size > group_size:
            y = x.view(group_size, -1, channels, height, width)
            # Variance over group
            y = torch.var(y, dim=0, unbiased=False)
            y = torch.sqrt(y + self.eps)
            # Mean over all features and spatial locations -> (group_count, 1, 1, 1)
            y = y.mean(dim=[1, 2, 3], keepdim=True)
            # Replicate over group_size to match batch_size
            y = y.repeat(group_size, 1, height, width)
        else:
            y = torch.var(x, dim=0, unbiased=False)
            y = torch.sqrt(y + self.eps)
            y = y.mean().view(1, 1, 1, 1).repeat(batch_size, 1, height, width)

        return torch.cat([x, y], dim=1)


class SelfAttention2d(nn.Module):
    """Self-Attention block for 2D feature maps (SAGAN-inspired).

    Captures long-range spatial dependencies across the image representation.
    """

    def __init__(self, in_channels: int, spectral_norm: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 8)

        conv_query = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        conv_key = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        if spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm as sn
            self.query = sn(conv_query)
            self.key = sn(conv_key)
            self.value = sn(conv_value)
        else:
            self.query = conv_query
            self.key = conv_key
            self.value = conv_value

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, c, h, w = x.shape
        num_locs = h * w

        proj_query = self.query(x).view(batch_size, self.inter_channels, num_locs).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, self.inter_channels, num_locs)
        energy = torch.bmm(proj_query, proj_key)  # (B, N, N)
        attention = self.softmax(energy)

        proj_value = self.value(x).view(batch_size, c, num_locs)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, c, h, w)

        return self.gamma * out + x
