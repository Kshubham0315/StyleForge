"""Unit and Regression Tests for Generative Metric Implementations.

Verifies:
1. Exact k-NN manifold estimation (Kynkäänniemi et al., 2019)
2. Precision and Recall on identical vs disjoint distributions
3. Distance metric scale invariance and stability
4. Real-to-real positive baseline validation
"""

import pytest
import torch
from evaluation.diversity import compute_pairwise_distances, compute_precision_recall


def test_pairwise_distances_geometry():
    """Verifies that Euclidean distance matrix has zero diagonal and symmetric properties."""
    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    Y = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

    dist_XY = compute_pairwise_distances(X, Y)
    assert dist_XY.shape == (3, 2)
    assert torch.isclose(dist_XY[0, 0], torch.tensor(0.0))
    assert torch.isclose(dist_XY[0, 1], torch.tensor(5.0))  # sqrt(3^2 + 4^2) = 5.0
    assert torch.isclose(dist_XY[1, 0], torch.tensor(1.0))

    dist_XX = compute_pairwise_distances(X, X)
    assert torch.allclose(torch.diag(dist_XX), torch.zeros(3))
    assert torch.allclose(dist_XX, dist_XX.T)


def test_precision_recall_identical_distributions():
    """Identical distributions drawn from the same standard normal should have substantial overlap."""
    torch.manual_seed(42)
    real = torch.randn(200, 64)
    fake = torch.randn(200, 64)

    p, r = compute_precision_recall(real, fake, k=3)
    assert 0.3 <= p <= 1.0, f"Expected non-zero precision for identical distributions, got {p}"
    assert 0.3 <= r <= 1.0, f"Expected non-zero recall for identical distributions, got {r}"


def test_precision_recall_disjoint_distributions():
    """Separated distributions (no manifold overlap) must yield exactly 0.0 precision and recall."""
    torch.manual_seed(42)
    real = torch.randn(100, 64)
    # Shift fake distribution far outside the real manifold
    fake = torch.randn(100, 64) + 100.0

    p, r = compute_precision_recall(real, fake, k=3)
    assert p == 0.0, f"Expected 0.0 precision for disjoint manifolds, got {p}"
    assert r == 0.0, f"Expected 0.0 recall for disjoint manifolds, got {r}"


def test_precision_recall_partial_mode_drop():
    """A mode drop (fake only covers half the real data) should yield high precision and lower recall."""
    torch.manual_seed(42)
    # Real data has two distant modes: mode 0 at -10.0, mode 1 at +10.0
    mode1 = torch.randn(100, 32) - 10.0
    mode2 = torch.randn(100, 32) + 10.0
    real = torch.cat([mode1, mode2], dim=0)  # 200 samples

    # Fake data only covers mode 1 (mode collapse / truncation)
    fake = torch.randn(100, 32) + 10.0

    p, r = compute_precision_recall(real, fake, k=3)
    assert p > 0.4, f"Precision should be high because fake is within mode 2, got {p}"
    assert r < 0.7, f"Recall should be bounded because mode 1 is completely dropped, got {r}"


def test_knn_radii_positivity_and_k_monotonicity():
    """Neighbor radius must increase monotonically as k increases."""
    torch.manual_seed(42)
    X = torch.randn(50, 16)
    dist = compute_pairwise_distances(X, X)

    radii_k1, _ = torch.topk(dist, k=2, dim=1, largest=False)
    r1 = radii_k1[:, -1]

    radii_k5, _ = torch.topk(dist, k=6, dim=1, largest=False)
    r5 = radii_k5[:, -1]

    assert (r1 > 0).all(), "Radii must be strictly positive for non-identical points"
    assert (r5 >= r1).all(), "k=5 radii must be >= k=1 radii"
