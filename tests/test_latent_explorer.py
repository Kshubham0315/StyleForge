"""Unit tests for latent explorer and interpolation."""

import pytest
import torch
from models.style_generator import StyleGenerator
from evaluation.latent_explorer import LatentExplorer, slerp, lerp


def test_slerp_and_lerp():
    v1 = torch.tensor([[1.0, 0.0]])
    v2 = torch.tensor([[0.0, 1.0]])

    interp_slerp = slerp(0.5, v1, v2)
    assert interp_slerp.shape == (1, 2)
    # Norm should be preserved in spherical interpolation
    assert torch.allclose(torch.norm(interp_slerp), torch.tensor(1.0), atol=1e-3)

    interp_lerp = lerp(0.5, v1, v2)
    assert torch.allclose(interp_lerp, torch.tensor([[0.5, 0.5]]))


def test_explorer_methods():
    g = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=64, base_channels=64)
    explorer = LatentExplorer(g, device=torch.device("cpu"))

    # Test Z interpolation
    z_interp = explorer.interpolate_z(steps=5, latent_dim=32)
    assert z_interp.shape == (5, 3, 64, 64)

    # Test W interpolation
    w_interp = explorer.interpolate_w(steps=5, latent_dim=32)
    assert w_interp.shape == (5, 3, 64, 64)

    # Test Truncation sweep
    trunc = explorer.truncation_sweep(psi_values=[0.0, 0.5, 1.0], latent_dim=32)
    assert trunc.shape == (3, 3, 64, 64)

    # Test Style mixing matrix
    matrix = explorer.style_mixing_matrix(num_src_a=2, num_src_b=2, latent_dim=32)
    # (1 + 2) rows * (1 + 2) columns = 9 images
    assert matrix.shape == (9, 3, 64, 64)
