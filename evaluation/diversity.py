"""Diversity and Mode Collapse Diagnostics for StyleForge.

Quantifies generative variety through:
- Pairwise Pixel Distance (L1 / L2)
- Deep Feature Diversity (perceptual distance across generated samples)
- Intra-batch variance
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiversityEvaluator:
    """Evaluates sample diversity to detect and monitor mode collapse."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate_diversity(
        self,
        generator: nn.Module,
        num_samples: int = 256,
        batch_size: int = 32,
        latent_dim: int = 512,
    ) -> Dict[str, float]:
        """Calculates pairwise distances across generated images."""
        generator.eval()
        all_fakes = []

        count = 0
        while count < num_samples:
            cur_bs = min(batch_size, num_samples - count)
            z = torch.randn(cur_bs, latent_dim, device=self.device)
            fake = generator(z)
            all_fakes.append(fake.cpu())
            count += cur_bs

        # Shape (N, C, H, W) in [-1, 1]
        all_fakes = torch.cat(all_fakes, dim=0)
        # Reshape to (N, C * H * W)
        flat = all_fakes.view(all_fakes.size(0), -1)

        # Batch variance
        batch_variance = float(torch.var(flat, dim=0).mean().item())

        # Pairwise L1 and L2 distances (sampling 1000 random pairs to avoid N^2 memory)
        num_pairs = min(1000, num_samples * 2)
        idx1 = torch.randint(0, num_samples, (num_pairs,))
        idx2 = torch.randint(0, num_samples, (num_pairs,))
        # Avoid self-pairs
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        diff = flat[idx1] - flat[idx2]
        l1_dist = float(torch.abs(diff).mean(dim=1).mean().item())
        l2_dist = float(torch.norm(diff, p=2, dim=1).mean().item())

        return {
            "pairwise_l1_distance": l1_dist,
            "pairwise_l2_distance": l2_dist,
            "batch_pixel_variance": batch_variance,
            "diversity_score": l1_dist * 10.0,  # Scaled index
        }
