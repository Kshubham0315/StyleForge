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
    mean_sq = torch.mean(out ** 2, dim=1)
    assert torch.allclose(mean_sq, torch.ones_like(mean_sq), atol=1e-3)


def test_equalized_layers():
    fc = EqualizedLinear(64, 128, lr_mul=0.01)
    x = torch.randn(2, 64)
    out = fc(x)
    assert out.shape == (2, 128)
    # Check that scale incorporates lr_mul
    assert fc.scale == pytest.approx((2.0 ** 0.5 / 8.0) * 0.01)

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
    assert not torch.isnan(out).any()


def test_noise_injection():
    noise_inj = NoiseInjection(channels=16)
    x = torch.zeros(2, 16, 8, 8)
    out = noise_inj(x)
    assert out.shape == x.shape


def test_minibatch_std_various_batch_sizes():
    std_layer = MinibatchStdDev(group_size=4)

    # Standard batch of 8 (2 groups of 4)
    x8 = torch.randn(8, 32, 4, 4)
    out8 = std_layer(x8)
    assert out8.shape == (8, 33, 4, 4)

    # Batch of 4 (1 group of 4)
    x4 = torch.randn(4, 32, 4, 4)
    out4 = std_layer(x4)
    assert out4.shape == (4, 33, 4, 4)

    # Odd batch size: 3 (smaller than group_size)
    x3 = torch.randn(3, 32, 4, 4)
    out3 = std_layer(x3)
    assert out3.shape == (3, 33, 4, 4)

    # Single sample batch: 1
    x1 = torch.randn(1, 32, 4, 4)
    out1 = std_layer(x1)
    assert out1.shape == (1, 33, 4, 4)


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


def test_mapping_network():
    m = MappingNetwork(latent_dim=64, style_dim=64, num_layers=4, lr_mul=0.01)
    z = torch.randn(3, 64)
    w = m(z)
    assert w.shape == (3, 64)
    assert not torch.isnan(w).any()


def test_style_generator():
    g = StyleGenerator(latent_dim=64, style_dim=64, num_mapping_layers=4, img_size=64, base_channels=128)
    z = torch.randn(2, 64)

    # Standard forward
    fake = g(z)
    assert fake.shape == (2, 3, 64, 64)

    # Truncation forward
    fake_trunc = g(z, truncation_psi=0.5, truncation_cutoff=4)
    assert fake_trunc.shape == (2, 3, 64, 64)

    # Style mixing forward
    z2 = torch.randn(2, 64)
    fake_mix = g(z, z2=z2, style_mixing_prob=1.0)
    assert fake_mix.shape == (2, 3, 64, 64)

    # generate_from_w
    w = g.mapping(z)
    fake_w = g.generate_from_w(w, truncation_psi=0.7)
    assert fake_w.shape == (2, 3, 64, 64)


def test_style_critic():
    c = StyleCritic(img_size=64, base_channels=32, use_minibatch_std=True, use_spectral_norm=True)
    img = torch.randn(2, 3, 64, 64)
    score = c(img)
    assert score.shape == (2, 1)
