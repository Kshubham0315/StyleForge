"""Unit tests for training losses and gradient penalties."""

import pytest
import torch
import torch.nn as nn
from training.losses import compute_gradient_penalty, compute_r1_penalty, GANLoss


class SimpleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = self.pool(h).view(h.size(0), -1)
        return self.fc(h)


def test_gradient_penalty():
    critic = SimpleCritic()
    real = torch.randn(2, 3, 32, 32)
    fake = torch.randn(2, 3, 32, 32)

    gp, norm = compute_gradient_penalty(critic, real, fake, gp_weight=10.0)
    assert gp.ndim == 0
    assert norm.ndim == 0
    assert gp.item() >= 0.0

    # Ensure backprop through gradient penalty computes gradients for weights
    gp.backward()
    assert critic.conv.weight.grad is not None
    assert critic.fc.weight.grad is not None


def test_r1_penalty():
    critic = SimpleCritic()
    real = torch.randn(2, 3, 32, 32)

    r1, norm = compute_r1_penalty(critic, real, r1_gamma=10.0)
    assert r1.ndim == 0
    assert norm.ndim == 0
    assert r1.item() >= 0.0

    r1.backward()
    assert critic.conv.weight.grad is not None
    assert critic.fc.weight.grad is not None


def test_gan_loss_modes():
    critic = SimpleCritic()
    real = torch.randn(2, 3, 32, 32)
    fake = torch.randn(2, 3, 32, 32)

    # WGAN-GP mode
    loss_wgan = GANLoss(mode="wgan_gp")
    d_loss, d_stats = loss_wgan.critic_loss(critic, real, fake)
    g_loss, g_stats = loss_wgan.generator_loss(critic, fake)
    assert d_loss.ndim == 0
    assert g_loss.ndim == 0
    assert "wasserstein_distance" in d_stats

    # R1 mode
    loss_r1 = GANLoss(mode="r1")
    d_loss_r1, d_stats_r1 = loss_r1.critic_loss(critic, real, fake)
    g_loss_r1, g_stats_r1 = loss_r1.generator_loss(critic, fake)
    assert d_loss_r1.ndim == 0
    assert g_loss_r1.ndim == 0
