"""Configurable Data Augmentation and Adaptive Discriminator Augmentation (ADA) Controller."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class AugmentationConfig:
    """Settings for training data augmentation pipeline."""

    def __init__(
        self,
        horizontal_flip: bool = True,
        flip_prob: float = 0.5,
        rotation: bool = False,
        rotation_degrees: float = 5.0,
        color_jitter: bool = False,
        brightness: float = 0.1,
        contrast: float = 0.05,
        use_ada: bool = False,
        ada_target: float = 0.6,
        ada_kimg: int = 10,
        ada_interval: int = 4,
    ):
        self.horizontal_flip = horizontal_flip
        self.flip_prob = flip_prob
        self.rotation = rotation
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.use_ada = use_ada
        self.ada_target = ada_target
        self.ada_kimg = ada_kimg
        self.ada_interval = ada_interval

    @classmethod
    def from_dict(cls, d: Dict) -> "AugmentationConfig":
        return cls(
            horizontal_flip=d.get("horizontal_flip", True),
            flip_prob=float(d.get("flip_prob", 0.5)),
            rotation=d.get("rotation", False),
            rotation_degrees=float(d.get("rotation_degrees", 5.0)),
            color_jitter=d.get("color_jitter", False),
            brightness=float(d.get("brightness", 0.1)),
            contrast=float(d.get("contrast", 0.05)),
            use_ada=d.get("use_ada", False),
            ada_target=float(d.get("ada_target", 0.6)),
            ada_kimg=int(d.get("ada_kimg", 10)),
            ada_interval=int(d.get("ada_interval", 4)),
        )


def build_face_transforms(
    img_size: int = 64,
    is_train: bool = True,
    config: Optional[AugmentationConfig] = None,
) -> transforms.Compose:
    """Builds a deterministic or augmented torchvision transform pipeline.

    Validation and testing sets are strictly unaugmented (CenterCrop + Resize + Normalize only).
    """
    cfg = config or AugmentationConfig()

    transforms_list = [
        transforms.CenterCrop(140),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    ]

    if is_train:
        # Candidate A: Random Horizontal Flip (p=0.5)
        if cfg.horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=cfg.flip_prob))

        # Candidate B: Subtle Rotation (-5° to +5°)
        if cfg.rotation:
            transforms_list.append(
                transforms.RandomRotation(
                    degrees=(-cfg.rotation_degrees, cfg.rotation_degrees),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )

        # Candidate C: Color / Brightness Jitter (approx 10%)
        if cfg.color_jitter:
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                )
            )

    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return transforms.Compose(transforms_list)


class AdaptiveAugmentationController:
    """Adaptive Discriminator Augmentation (ADA) Controller (Karras et al. NeurIPS 2020).

    Monitors discriminator overfitting using the heuristic r_t = E[sign(D(x_real))].
    Adjusts augmentation probability p dynamically:
      - If r_t > target (discriminator saturating / memorizing): increase p
      - If r_t < target: decrease p
    """

    def __init__(
        self,
        target: float = 0.6,
        speed_kimg: int = 10,
        batch_size: int = 16,
        update_interval: int = 4,
    ):
        self.target = target
        self.p = 0.0  # Initial augmentation probability
        self.batch_size = batch_size
        self.update_interval = update_interval
        # delta_p adjustment rate per batch: batch_size / (speed_kimg * 1000)
        self.delta_p = (batch_size * update_interval) / (speed_kimg * 1000.0)
        self.sign_accum = 0.0
        self.num_samples = 0
        self.step_count = 0

    def update(self, real_scores: torch.Tensor) -> float:
        """Update controller heuristic with discriminator real sample predictions."""
        with torch.no_grad():
            signs = (real_scores > 0).float()
            self.sign_accum += signs.sum().item()
            self.num_samples += real_scores.size(0)
            self.step_count += 1

            if self.step_count % self.update_interval == 0 and self.num_samples > 0:
                r_t = self.sign_accum / self.num_samples
                # Adjust probability p towards target
                if r_t > self.target:
                    self.p = min(1.0, self.p + self.delta_p)
                else:
                    self.p = max(0.0, self.p - self.delta_p)

                # Reset accumulator for next interval
                self.sign_accum = 0.0
                self.num_samples = 0

        return self.p

    def get_p(self) -> float:
        return self.p
