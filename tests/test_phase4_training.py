import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn
import tempfile
import copy

from training.trainer import GANTrainer, ExponentialMovingAverage
from training.losses import GANLoss, compute_r1_penalty
from training.augmentation import AugmentationConfig, build_face_transforms, AdaptiveAugmentationController
from models.style_generator import StyleGenerator
from models.critic import StyleCritic


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=32):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 64, 64)


@pytest.fixture
def dummy_setup():
    gen = StyleGenerator(latent_dim=64, style_dim=64, img_size=64, base_channels=64, max_channels=64)
    critic = StyleCritic(img_size=64, base_channels=64, max_channels=64)
    loader = torch.utils.data.DataLoader(DummyDataset(16), batch_size=4)
    config = {
        "experiment": {"exp_dir": tempfile.mkdtemp()},
        "training": {
            "lr_g": 0.0002,
            "lr_d": 0.0002,
            "loss_mode": "logistic",
            "penalty_weight": 10.0,
            "lazy_r1": True,
            "lazy_interval": 4,
            "amp": False,
        },
        "model": {"latent_dim": 64},
    }
    return gen, critic, loader, config


def test_augmentation_transforms():
    cfg = AugmentationConfig(
        horizontal_flip=True,
        rotation=True,
        rotation_degrees=5.0,
        color_jitter=True,
        brightness=0.1,
    )
    t_train = build_face_transforms(img_size=64, is_train=True, config=cfg)
    t_val = build_face_transforms(img_size=64, is_train=False, config=cfg)

    from PIL import Image
    dummy_pil = Image.fromarray(torch.randint(0, 255, (180, 180, 3), dtype=torch.uint8).numpy())

    tensor_train = t_train(dummy_pil)
    tensor_val = t_val(dummy_pil)

    assert tensor_train.shape == (3, 64, 64)
    assert tensor_val.shape == (3, 64, 64)
    assert tensor_train.min() >= -1.01 and tensor_train.max() <= 1.01


def test_ada_controller():
    ada = AdaptiveAugmentationController(target=0.6, speed_kimg=1, batch_size=16, update_interval=2)
    assert ada.get_p() == 0.0

    # Feed saturated discriminator scores (all positive)
    scores_high = torch.ones(16) * 5.0
    p1 = ada.update(scores_high)
    p2 = ada.update(scores_high)
    # p should have increased
    assert p2 > 0.0

    # Feed real scores below zero
    scores_low = torch.ones(16) * -5.0
    for _ in range(4):
        p_dec = ada.update(scores_low)
    assert p_dec < p2


def test_lazy_r1_regularization_interval():
    critic = StyleCritic(img_size=64, base_channels=64, max_channels=64)
    real = torch.randn(4, 3, 64, 64)
    fake = torch.randn(4, 3, 64, 64)

    loss_fn = GANLoss(mode="logistic", penalty_weight=10.0, lazy_r1=True, lazy_interval=4)

    # Step 1: non-regularized step
    _, stats_1 = loss_fn.critic_loss(critic, real, fake, step=1)
    assert stats_1["penalty"] == 0.0

    # Step 4: regularized step
    _, stats_4 = loss_fn.critic_loss(critic, real, fake, step=4)
    assert stats_4["penalty"] >= 0.0
    assert stats_4["grad_norm"] >= 0.0


def test_safe_checkpoint_serialization_and_resume(dummy_setup):
    gen, critic, loader, config = dummy_setup
    trainer = GANTrainer(gen, critic, loader, config, device=torch.device("cpu"))
    trainer.global_step = 100
    trainer.best_metric = 42.5

    # Save checkpoint
    trainer.save_checkpoint("test_ckpt.pt")
    ckpt_path = trainer.checkpoints_dir / "test_ckpt.pt"
    assert ckpt_path.exists()

    # Load into fresh trainer with TTUR override
    new_config = copy.deepcopy(config)
    new_config["training"]["lr_g"] = 0.001
    new_config["training"]["lr_d"] = 0.002

    gen2 = StyleGenerator(latent_dim=64, style_dim=64, img_size=64, base_channels=64, max_channels=64)
    critic2 = StyleCritic(img_size=64, base_channels=64, max_channels=64)
    trainer2 = GANTrainer(gen2, critic2, loader, new_config, device=torch.device("cpu"))

    trainer2.load_checkpoint(str(ckpt_path))
    assert trainer2.global_step == 100
    assert trainer2.best_metric == 42.5
    # TTUR learning rates correctly overridden
    assert trainer2.opt_g.param_groups[0]["lr"] == 0.001
    assert trainer2.opt_d.param_groups[0]["lr"] == 0.002
