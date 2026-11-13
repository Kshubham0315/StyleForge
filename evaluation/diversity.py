"""Diversity, Mode Collapse Diagnostics, and Precision/Recall for StyleForge.

Quantifies generative variety through:
- Pairwise Pixel Distance (L1 / L2)
- Intra-batch variance (Mode collapse indicator)
- Precision and Recall for Generative Models (Kynkäänniemi et al. / Sajjadi et al.)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pairwise_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Computes Euclidean distance matrix between rows of X and Y."""
    dist = torch.cdist(X, Y, p=2.0)
    return dist


def compute_precision_recall(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    k: int = 3,
) -> Tuple[float, float]:
    """Computes Precision and Recall for Generative Models (k-NN manifold estimation).

    - Precision: Fraction of generated samples falling within the support manifold of real data.
    - Recall: Fraction of real data falling within the support manifold of generated samples.
    """
    N_real = real_features.size(0)
    N_fake = fake_features.size(0)

    # 1. Real-to-real pairwise distance
    real_to_real = compute_pairwise_distances(real_features, real_features)
    # Distance to k-th nearest neighbor (ignoring self at index 0)
    k_real = min(k, N_real - 1)
    real_radii, _ = torch.topk(real_to_real, k=k_real + 1, dim=1, largest=False)
    real_k_dist = real_radii[:, -1]  # (N_real,)

    # 2. Fake-to-fake pairwise distance
    fake_to_fake = compute_pairwise_distances(fake_features, fake_features)
    k_fake = min(k, N_fake - 1)
    fake_radii, _ = torch.topk(fake_to_fake, k=k_fake + 1, dim=1, largest=False)
    fake_k_dist = fake_radii[:, -1]  # (N_fake,)

    # 3. Distance between real and fake
    real_to_fake = compute_pairwise_distances(real_features, fake_features)  # (N_real, N_fake)

    # Precision: For each fake, check if distance to any real_i is <= real_k_dist[i]
    is_in_real_manifold = real_to_fake <= real_k_dist.unsqueeze(1)  # (N_real, N_fake)
    precision = float(is_in_real_manifold.any(dim=0).float().mean().item())

    # Recall: For each real, check if distance to any fake_j is <= fake_k_dist[j]
    is_in_fake_manifold = real_to_fake <= fake_k_dist.unsqueeze(0)  # (N_real, N_fake)
    recall = float(is_in_fake_manifold.any(dim=1).float().mean().item())

    return precision, recall


class DiversityEvaluator:
    """Evaluates sample diversity, mode collapse, and manifold coverage."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate_diversity(
        self,
        generator: nn.Module,
        num_samples: int = 256,
        batch_size: int = 32,
        latent_dim: int = 512,
        real_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Calculates pairwise distances, batch variance, and precision/recall across generated images."""
        generator.eval()
        all_fakes = []

        count = 0
        while count < num_samples:
            cur_bs = min(batch_size, num_samples - count)
            z = torch.randn(cur_bs, latent_dim, device=self.device)
            if hasattr(generator, "style_dim"):
                fake = generator(z, truncation_psi=0.8)
            else:
                fake = generator(z)
            all_fakes.append(fake.cpu())
            count += cur_bs

        all_fakes = torch.cat(all_fakes, dim=0)  # (N, C, H, W) in [-1, 1]
        flat = all_fakes.view(all_fakes.size(0), -1)

        # Batch pixel variance (mode collapse metric)
        batch_variance = float(torch.var(flat, dim=0).mean().item())

        # Pairwise L1 and L2 distances (subsampling pairs for efficiency)
        num_pairs = min(1000, num_samples * 2)
        idx1 = torch.randint(0, num_samples, (num_pairs,))
        idx2 = torch.randint(0, num_samples, (num_pairs,))
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        diff = flat[idx1] - flat[idx2]
        l1_dist = float(torch.abs(diff).mean(dim=1).mean().item())
        l2_dist = float(torch.norm(diff, p=2, dim=1).mean().item())

        results = {
            "pairwise_l1_distance": l1_dist,
            "pairwise_l2_distance": l2_dist,
            "batch_pixel_variance": batch_variance,
            "diversity_score": l1_dist * 10.0,
        }

        # If real images provided, compute Precision & Recall
        if real_images is not None:
            real_flat = real_images.cpu().view(real_images.size(0), -1)[:num_samples]
            p, r = compute_precision_recall(real_flat, flat[:real_flat.size(0)])
            results["precision"] = p
            results["recall"] = r

        return results
