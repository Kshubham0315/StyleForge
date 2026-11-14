"""Unit and Regression Tests for Mathematically Correct EMA Bias Correction & Scheduling."""

import pytest
import torch
import torch.nn as nn
from training.trainer import ExponentialMovingAverage


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=True)
        self.register_buffer("running_mean", torch.tensor([1.0, 2.0]))

    def forward(self, x):
        return self.fc(x)


def test_standard_ema_recurrence():
    """Verifies that standard EMA update matches theta_t = beta * theta_(t-1) + (1-beta) * theta_t."""
    torch.manual_seed(42)
    model = SimpleToyModel()
    ema = ExponentialMovingAverage(model, decay=0.9, use_warmup=False, bias_correction=False)

    init_weight = model.fc.weight.clone()
    assert torch.allclose(ema.shadow.fc.weight, init_weight)

    # Step 1: change model weight to all 10s
    with torch.no_grad():
        model.fc.weight.fill_(10.0)
    ema.update(model)

    expected_w1 = 0.9 * init_weight + 0.1 * 10.0
    assert torch.allclose(ema.shadow.fc.weight, expected_w1, atol=1e-5)
    assert ema.num_updates == 1


def test_ema_model_init_bias_correction():
    """Verifies that debiased weights eliminate the initial condition theta^(0)."""
    torch.manual_seed(42)
    model = SimpleToyModel()
    init_weight = model.fc.weight.clone()

    ema = ExponentialMovingAverage(model, decay=0.99, use_warmup=False, bias_correction=True, init_mode="model")

    # Update 5 times with constant parameter W = 5.0
    with torch.no_grad():
        model.fc.weight.fill_(5.0)

    for _ in range(5):
        ema.update(model)

    # In uncorrected shadow, theta_shadow = (0.99^5) * init_weight + (1 - 0.99^5) * 5.0
    # The debiased shadow should be EXACTLY 5.0!
    debiased_model = ema.get_debiased_shadow()
    assert torch.allclose(debiased_model.fc.weight, torch.full_like(debiased_model.fc.weight, 5.0), atol=1e-5)
    # The raw shadow should still contain residual of init_weight
    assert not torch.allclose(ema.shadow.fc.weight, torch.full_like(ema.shadow.fc.weight, 5.0))


def test_ema_zero_init_bias_correction():
    """Verifies that zero-initialized EMA matches the normalized weighted sum."""
    torch.manual_seed(42)
    model = SimpleToyModel()
    ema = ExponentialMovingAverage(model, decay=0.95, use_warmup=False, bias_correction=True, init_mode="zero")

    # Initial shadow must be all zeros
    assert torch.allclose(ema.shadow.fc.weight, torch.zeros_like(ema.shadow.fc.weight))

    # Update with W = 8.0 for 10 steps
    with torch.no_grad():
        model.fc.weight.fill_(8.0)

    for _ in range(10):
        ema.update(model)

    # Debiased shadow must be exactly 8.0
    debiased_model = ema.get_debiased_shadow()
    assert torch.allclose(debiased_model.fc.weight, torch.full_like(debiased_model.fc.weight, 8.0), atol=1e-5)


def test_ema_warmup_schedule():
    """Verifies that warmup schedule starts low and monotonically approaches asymptotic decay."""
    model = SimpleToyModel()
    ema = ExponentialMovingAverage(model, decay=0.999, use_warmup=True)

    decays = []
    for step in range(1, 100):
        ema.update(model)
        decays.append(ema.get_effective_decay())

    # Effective decay must be strictly non-decreasing
    for i in range(len(decays) - 1):
        assert decays[i] <= decays[i + 1]

    # At step 1, beta_eff should be (1+1)/(10+1) = 2/11 = ~0.1818
    assert abs(decays[0] - (2.0 / 11.0)) < 0.05
    # As t -> inf, beta_eff approaches target decay 0.999
    assert decays[-1] <= 0.999


def test_ema_metadata_checkpoint_serialization():
    """Verifies that metadata and update count are correctly extracted and restored."""
    model = SimpleToyModel()
    ema = ExponentialMovingAverage(model, decay=0.995, use_warmup=True, bias_correction=True, init_mode="zero")

    for _ in range(25):
        ema.update(model)

    meta = ema.get_metadata()
    assert meta["num_updates"] == 25
    assert meta["decay"] == 0.995
    assert meta["use_warmup"] is True
    assert meta["bias_correction"] is True
    assert meta["init_mode"] == "zero"

    # Restore into fresh EMA
    fresh_ema = ExponentialMovingAverage(model, decay=0.9)
    fresh_ema.load_metadata(meta)
    assert fresh_ema.num_updates == 25
    assert fresh_ema.decay == 0.995
    assert fresh_ema.use_warmup is True
    assert fresh_ema.bias_correction is True
