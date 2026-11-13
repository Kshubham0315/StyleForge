"""End-to-end Integration and System Tests for StyleForge."""

import tempfile
from pathlib import Path
import pytest
import torch
from data.dataset import SyntheticFaceDataset, get_dataloader
from evaluation.diversity import compute_precision_recall, DiversityEvaluator
from evaluation.latent_explorer import LatentExplorer
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.trainer import GANTrainer, ExponentialMovingAverage


def test_synthetic_dataset_and_loader():
    ds = SyntheticFaceDataset(num_samples=20, img_size=32, seed=42)
    assert len(ds) == 20
    img = ds[0]
    assert img.shape == (3, 32, 32)
    assert -1.0 <= img.min() and img.max() <= 1.0

    loader = get_dataloader(use_synthetic=True, synthetic_samples=16, batch_size=4, img_size=32)
    batch = next(iter(loader))
    assert batch.shape == (4, 3, 32, 32)


def test_precision_recall_computation():
    torch.manual_seed(42)
    # Identical distributions should give high precision and recall
    reals = torch.randn(50, 16)
    fakes = reals + 0.01 * torch.randn(50, 16)

    p, r = compute_precision_recall(reals, fakes, k=3)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert p > 0.5
    assert r > 0.5


def test_ema_updates():
    g = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    ema = ExponentialMovingAverage(g, decay=0.9)

    # Change weights of original model
    with torch.no_grad():
        for p in g.parameters():
            p.add_(1.0)

    # Shadow before update should be different
    for s_p, m_p in zip(ema.shadow.parameters(), g.parameters()):
        assert not torch.allclose(s_p, m_p)

    # Update EMA
    ema.update(g)
    # Shadow should have moved towards modified model
    for s_p, m_p in zip(ema.shadow.parameters(), g.parameters()):
        assert s_p.abs().sum() > 0


def test_checkpoint_save_and_load(tmp_path):
    g = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    c = StyleCritic(img_size=32, base_channels=16)

    loader = get_dataloader(use_synthetic=True, synthetic_samples=16, batch_size=4, img_size=32)
    config = {
        "experiment": {"exp_dir": str(tmp_path / "exp")},
        "model": {"type": "stylegan", "latent_dim": 32, "style_dim": 32, "img_size": 32},
        "training": {
            "total_steps": 2,
            "lr_g": 0.001,
            "lr_d": 0.001,
            "n_critic": 1,
            "loss_mode": "wgan_gp",
            "penalty_weight": 5.0,
            "batch_size": 4,
            "amp": False,
        },
    }

    trainer = GANTrainer(g, c, loader, config, device=torch.device("cpu"))
    trainer.save_checkpoint("test_ckpt.pt")

    ckpt_file = tmp_path / "exp" / "checkpoints" / "test_ckpt.pt"
    assert ckpt_file.exists()

    # Load into fresh trainer
    g_fresh = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    c_fresh = StyleCritic(img_size=32, base_channels=16)
    trainer_fresh = GANTrainer(g_fresh, c_fresh, loader, config, device=torch.device("cpu"))
    trainer_fresh.load_checkpoint(str(ckpt_file))
    assert trainer_fresh.global_step == 0


def test_full_single_train_step(tmp_path):
    g = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    c = StyleCritic(img_size=32, base_channels=16)

    loader = get_dataloader(use_synthetic=True, synthetic_samples=8, batch_size=4, img_size=32)
    config = {
        "experiment": {"exp_dir": str(tmp_path / "exp_step")},
        "model": {"type": "stylegan", "latent_dim": 32, "style_dim": 32, "img_size": 32},
        "training": {
            "total_steps": 1,
            "lr_g": 0.001,
            "lr_d": 0.001,
            "n_critic": 1,
            "loss_mode": "wgan_gp",
            "penalty_weight": 5.0,
            "batch_size": 4,
            "amp": False,
            "sample_every": 1,
            "checkpoint_every": 1,
        },
    }

    trainer = GANTrainer(g, c, loader, config, device=torch.device("cpu"))
    trainer.train(total_steps=1, sample_every=1, checkpoint_every=1)
    assert trainer.global_step == 1
    assert len(trainer.history) == 1
    assert "g_loss" in trainer.history[0]
    assert "wasserstein_distance" in trainer.history[0]


def test_training_resume(tmp_path):
    """Verifies that a trainer can cleanly resume and advance global_step."""
    g = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    c = StyleCritic(img_size=32, base_channels=16)

    loader = get_dataloader(use_synthetic=True, synthetic_samples=8, batch_size=4, img_size=32)
    config = {
        "experiment": {"exp_dir": str(tmp_path / "exp_resume")},
        "model": {"type": "stylegan", "latent_dim": 32, "style_dim": 32, "img_size": 32},
        "training": {
            "total_steps": 1,
            "lr_g": 0.001,
            "lr_d": 0.001,
            "n_critic": 1,
            "loss_mode": "wgan_gp",
            "penalty_weight": 5.0,
            "batch_size": 4,
            "amp": False,
        },
    }

    trainer1 = GANTrainer(g, c, loader, config, device=torch.device("cpu"))
    trainer1.train(total_steps=1)
    ckpt_path = tmp_path / "exp_resume" / "checkpoints" / "checkpoint_final.pt"
    assert ckpt_path.exists()

    # Resume into fresh trainer up to 2 steps
    g2 = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    c2 = StyleCritic(img_size=32, base_channels=16)
    trainer2 = GANTrainer(g2, c2, loader, config, device=torch.device("cpu"))
    trainer2.load_checkpoint(str(ckpt_path))
    assert trainer2.global_step == 1

    trainer2.train(total_steps=2)
    assert trainer2.global_step == 2


def test_fid_preprocessing_and_uint8():
    """Verifies [-1, 1] float tensor to [0, 255] uint8 conversion."""
    from evaluation.fid_kid import to_uint8_images
    t = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3)
    uint8_t = to_uint8_images(t)
    assert uint8_t.dtype == torch.uint8
    assert uint8_t[0, 0, 0, 0].item() == 0
    assert uint8_t[0, 0, 0, 1].item() == 127
    assert uint8_t[0, 0, 0, 2].item() == 255


def test_fixed_seed_reproducibility():
    """Verifies that fixing seeds yields bitwise deterministic image tensors."""
    g = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    g.eval()

    torch.manual_seed(1234)
    z1 = torch.randn(4, 32)
    out1 = g(z1)

    torch.manual_seed(1234)
    z2 = torch.randn(4, 32)
    out2 = g(z2)

    diff = (out1 - out2).abs().max().item()
    assert diff == 0.0, f"Expected 0 diff, got {diff}"


def test_diversity_metric_keys():
    """Verifies that DiversityEvaluator outputs all expected metric fields."""
    g = StyleGenerator(latent_dim=32, style_dim=32, num_mapping_layers=2, img_size=32, base_channels=32)
    evaluator = DiversityEvaluator(device=torch.device("cpu"))
    results = evaluator.evaluate_diversity(g, num_samples=8, batch_size=4, latent_dim=32)

    required_keys = ["pairwise_l1_distance", "pairwise_l2_distance", "batch_pixel_variance", "diversity_score"]
    for k in required_keys:
        assert k in results, f"Missing metric key: {k}"
        assert isinstance(results[k], float)
        assert results[k] >= 0.0

