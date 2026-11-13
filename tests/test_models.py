"""Unit tests for StyleForge model architectures."""

import pytest
import torch
from models.common import (
    PixelNorm,
    EqualizedLinear,
    EqualizedConv2d,
    NoiseInjection,
    AdaIN,
    MinibatchStdDev,
    SelfAttention2d,
)
from models.wgan_baseline import WGANBaselineGenerator, WGANBaselineCritic
from models.style_generator import StyleGenerator, MappingNetwork, SynthesisNetwork
from models.critic import StyleCritic


def test_pixel_norm():
    norm = PixelNorm()
    x = torch.randn(4, 512) * 5.0
    out = norm(x)
    assert out.shape == x.shape
    # Check that average squared norm across channels is close to 1
    mean_sq = torch.mean(out ** 2, dim=1)
    assert torch.allclose(mean_sq, torch.ones_like(mean_sq), atol=1e-3)


def test_equalized_layers():
    fc = EqualizedLinear(64, 128)
    x = torch.randn(2, 64)
    out = fc(x)
    assert out.shape == (2, 128)

    conv = EqualizedConv2d(16, 32, kernel_size=3, padding=1)
    x_img = torch.randn(2, 16, 16, 16)
    out_conv = conv(x_img)
    assert out_conv.shape == (2, 32, 16, 16)


def test_adain():
    adain = AdaIN(style_dim=64, num_features=32)
    x = torch.randn(2, 32, 16, 16)
    w = torch.randn(2, 64)
    out = adain(x, w)
    assert out.shape == (2, 32, 16, 16)


def test_noise_injection():
    noise_inj = NoiseInjection(channels=16)
    x = torch.zeros(2, 16, 8, 8)
    out = noise_inj(x)
    assert out.shape == x.shape


def test_minibatch_std():
    std_layer = MinibatchStdDev()
    x = torch.randn(4, 32, 4, 4)
    out = std_layer(x)
    assert out.shape == (4, 33, 4, 4)


def test_self_attention():
    attn = SelfAttention2d(in_channels=32)
    x = torch.randn(2, 32, 16, 16)
    out = attn(x)
    assert out.shape == (2, 32, 16, 16)


def test_wgan_baseline():
    g = WGANBaselineGenerator(latent_dim=64, img_size=64)
    c = WGANBaselineCritic(img_size=64)

    z = torch.randn(2, 64)
    fake = g(z)
    assert fake.shape == (2, 3, 64, 64)

    score = c(fake)
    assert score.shape == (2, 1)


def test_style_generator():
    g = StyleGenerator(latent_dim=64, style_dim=64, num_mapping_layers=4, img_size=64, base_channels=128)
    z = torch.randn(2, 64)

    # Standard forward
    fake = g(z)
    assert fake.shape == (2, 3, 64, 64)

    # Truncation forward
    fake_trunc = g(z, truncation_psi=0.5)
    assert fake_trunc.shape == (2, 3, 64, 64)

    # Style mixing forward
    z2 = torch.randn(2, 64)
    fake_mix = g(z, z2=z2, style_mixing_prob=1.0)
    assert fake_mix.shape == (2, 3, 64, 64)


def test_style_critic():
    c = StyleCritic(img_size=64, base_channels=32, use_minibatch_std=True, use_spectral_norm=True)
    img = torch.randn(2, 3, 64, 64)
    score = c(img)
    assert score.shape == (2, 1)
