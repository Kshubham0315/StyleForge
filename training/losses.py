"""Loss functions and regularizers for GAN training stability.

Includes:
- Wasserstein GAN loss with gradient penalty (WGAN-GP)
- Non-saturating logistic GAN loss with R1 regularization
- Wasserstein distance estimator and critic diagnostics
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    gp_weight: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute WGAN-GP 2-sided gradient penalty.

    Interpolates between real and fake images along straight lines,
    evaluates critic gradients in float32 without autocast for numerical stability,
    and penalizes deviation of gradient norm from 1.
    """
    dev_type = real_images.device.type
    with torch.amp.autocast(dev_type, enabled=False):
        real = real_images.detach().float()
        fake = fake_images.detach().float()
        batch_size = real.size(0)

        # Uniform random interpolation coefficient
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolated = (epsilon * real + (1.0 - epsilon) * fake).requires_grad_(True)

        critic_interpolated = critic(interpolated).float()

        grad_outputs = torch.ones_like(critic_interpolated, device=real.device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten spatial dims to compute L2 norm per sample
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gradient_penalty = gp_weight * torch.mean((grad_norm - 1.0) ** 2)

        return gradient_penalty, grad_norm.mean().detach()


def compute_r1_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    r1_gamma: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute R1 gradient penalty on real images.

    Evaluated in float32 without autocast for numerical stability:
    R1 = (gamma / 2) * E_{x ~ p_data} [ || grad_x D(x) ||^2 ]
    """
    dev_type = real_images.device.type
    with torch.amp.autocast(dev_type, enabled=False):
        real_images = real_images.detach().float().requires_grad_(True)
        real_scores = critic(real_images).float()

        grad_outputs = torch.ones_like(real_scores, device=real_images.device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=real_scores.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        grad_sq_norm = gradients.pow(2).sum(dim=1)
        r1_penalty = (r1_gamma * 0.5) * grad_sq_norm.mean()
        grad_norm = gradients.norm(2, dim=1).mean().detach()

        return r1_penalty, grad_norm


class GANLoss(nn.Module):
    """Unified GAN loss module supporting WGAN-GP and Non-Saturating R1 modes."""

    def __init__(self, mode: str = "wgan_gp", penalty_weight: float = 10.0):
        super().__init__()
        self.mode = mode.lower()
        self.penalty_weight = penalty_weight

        if self.mode not in ("wgan_gp", "r1", "logistic"):
            raise ValueError(f"Unknown loss mode: {mode}. Choose 'wgan_gp', 'r1', or 'logistic'")

    def critic_loss(
        self,
        critic: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute critic loss and diagnostics."""
        real_scores = critic(real_images)
        fake_scores = critic(fake_images.detach())

        stats = {
            "real_score_mean": real_scores.mean().item(),
            "real_score_std": real_scores.std().item() if real_scores.size(0) > 1 else 0.0,
            "fake_score_mean": fake_scores.mean().item(),
            "fake_score_std": fake_scores.std().item() if fake_scores.size(0) > 1 else 0.0,
            "wasserstein_distance": (real_scores.mean() - fake_scores.mean()).item(),
        }

        if self.mode == "wgan_gp":
            base_loss = fake_scores.mean() - real_scores.mean()
            gp, grad_norm = compute_gradient_penalty(
                critic, real_images, fake_images, gp_weight=self.penalty_weight
            )
            total_loss = base_loss + gp
            stats["base_loss"] = base_loss.item()
            stats["penalty"] = gp.item()
            stats["grad_norm"] = grad_norm.item()

        elif self.mode in ("r1", "logistic"):
            # Non-saturating logistic loss
            loss_real = F.softplus(-real_scores).mean()
            loss_fake = F.softplus(fake_scores).mean()
            base_loss = loss_real + loss_fake
            r1, grad_norm = compute_r1_penalty(
                critic, real_images, r1_gamma=self.penalty_weight
            )
            total_loss = base_loss + r1
            stats["base_loss"] = base_loss.item()
            stats["penalty"] = r1.item()
            stats["grad_norm"] = grad_norm.item()

        return total_loss, stats

    def generator_loss(
        self,
        critic: nn.Module,
        fake_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute generator loss."""
        fake_scores = critic(fake_images)

        if self.mode == "wgan_gp":
            loss = -fake_scores.mean()
        elif self.mode in ("r1", "logistic"):
            loss = F.softplus(-fake_scores).mean()

        stats = {
            "g_loss": loss.item(),
            "g_fake_score_mean": fake_scores.mean().item(),
        }
        return loss, stats
