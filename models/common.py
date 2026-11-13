"""StyleForge Common Building Blocks.

Contains standard generative layers:
- PixelNorm
- EqualizedLinear / EqualizedConv2d (Equalized Learning Rate with runtime He-scaling & lr_mul)
- Adaptive Instance Normalization (AdaIN)
- Per-channel Noise Injection
- Minibatch Standard Deviation with robust group tiling
- Self-Attention 2D (SAGAN)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Pixel Normalization layer.

    Normalizes the feature vector in each pixel to unit length across channels:
    y = x / sqrt(mean(x^2) + eps)
    Prevents signal magnitudes from escalating out of control in early layers.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate (runtime He scaling) and lr_mul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gain: float = 2.0 ** 0.5,
        lr_mul: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_mul = lr_mul

        # Initialize weights ~ N(0, 1) divided by lr_mul so that runtime scaling
        # (scale * weight) preserves unit forward variance across multiple layers
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.scale = (gain / math.sqrt(in_features)) * lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale
        if self.bias is not None:
            b = self.bias * self.lr_mul
            return F.linear(x, w, b)
        return F.linear(x, w)


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
        lr_mul: float = 1.0,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.lr_mul = lr_mul

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) / lr_mul
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        fan_in = in_channels * (kernel_size ** 2)
        self.scale = (gain / math.sqrt(fan_in)) * lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.scale
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)


class NoiseInjection(nn.Module):
    """Stochastic noise injection layer.

    Injects learned per-channel scaled Gaussian noise into spatial feature maps
    to promote generation of fine stochastic details (e.g., hair strands, skin pores).
    """

    def __init__(self, channels: int):
        super().__init__()
        # Initialized to zero so noise does not disturb early structural convergence
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = torch.randn(batch_size, 1, height, width, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Normalizes input feature map x to zero-mean and unit variance, then modulates
    it using scale (y_s) and bias (y_b) derived from intermediate style vector w:
    AdaIN(x, y) = y_s * ( (x - mu(x)) / sigma(x) ) + y_b
    """

    def __init__(self, style_dim: int, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        # Style modulation produces [scale, bias] for each feature channel
        self.style_mod = EqualizedLinear(style_dim, num_features * 2, gain=1.0)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Standard instance normalization across spatial dims (H, W)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Style modulation: chunk into scale and bias
        style = self.style_mod(w)
        scale, bias = style.chunk(2, dim=1)

        scale = (scale + 1.0).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)

        return normalized * scale + bias


class MinibatchStdDev(nn.Module):
    """Minibatch Standard Deviation Layer.

    Computes standard deviation of features across the minibatch for each spatial location,
    averages over all features and locations, and appends the result as an extra feature map.
    Helps discriminator detect mode collapse.
    """

    def __init__(self, group_size: int = 4, eps: float = 1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        G = min(N, self.group_size)
        if N % G != 0:
            G = 1  # Fallback to single group across whole batch

        if G > 1:
            # Group into (num_groups, group_size, C, H, W)
            num_groups = N // G
            y = x.view(num_groups, G, C, H, W)
            # Variance across group members (dim 1)
            y = torch.var(y, dim=1, unbiased=False)
            y = torch.sqrt(y + self.eps)
            # Mean across channels and spatial locations -> (num_groups, 1, 1, 1)
            y = y.mean(dim=[1, 2, 3], keepdim=True)
            # Expand along group members to match batch size N -> (num_groups, G, 1, H, W)
            y = y.view(num_groups, 1, 1, 1, 1).expand(-1, G, -1, H, W)
            # Reshape back to (N, 1, H, W)
            y = y.contiguous().view(N, 1, H, W)
        else:
            y = torch.var(x, dim=0, unbiased=False)
            y = torch.sqrt(y + self.eps)
            y = y.mean().view(1, 1, 1, 1).expand(N, 1, H, W)

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
