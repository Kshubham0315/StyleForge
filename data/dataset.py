"""StyleForge Data Loading and Preprocessing Pipeline.

Provides:
- Custom ImageFolder and CelebA dataset loading with face center cropping
- Synthetic procedural face dataset generator for zero-setup smoke testing
- DataLoader factory with configurable augmentations and caching
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceTransform:
    """Prepares face images by cropping to central face region and resizing."""

    def __init__(self, img_size: int = 64, horizontal_flip: bool = True):
        transforms_list = [
            # For standard CelebA (178x218), central 140x140 or 178x178 crops the face
            transforms.CenterCrop(140),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transforms_list.extend([
            transforms.ToTensor(),
            # Normalize to [-1, 1] range expected by GAN tanh generator
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.transform = transforms.Compose(transforms_list)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class LocalImageFolderDataset(Dataset):
    """Loads images from any directory containing JPG/PNG face images."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        img_size: int = 64,
        max_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transform = transform or FaceTransform(img_size=img_size)

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in valid_exts and not p.name.startswith(".")
        ]

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


class SyntheticFaceDataset(Dataset):
    """Generates procedural face-like synthetic images in memory.

    Enables immediate testing and verification without downloading large datasets.
    Renders geometric face ovals, eyes, mouth, and random skin tones with noise.
    """

    def __init__(self, num_samples: int = 500, img_size: int = 64, seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return self.num_samples

    def _render_synthetic_face(self, index: int) -> Image.Image:
        rng = np.random.RandomState(index)
        size = 128
        img = Image.new("RGB", (size, size), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)

        # Random skin color
        r = rng.randint(180, 255)
        g = rng.randint(130, 210)
        b = rng.randint(100, 180)
        skin_color = (r, g, b)

        # Head oval
        head_w = rng.randint(50, 70)
        head_h = rng.randint(65, 85)
        cx, cy = size // 2, size // 2
        draw.ellipse([cx - head_w // 2, cy - head_h // 2, cx + head_w // 2, cy + head_h // 2], fill=skin_color)

        # Eyes
        eye_y = cy - rng.randint(5, 15)
        eye_spacing = rng.randint(15, 25)
        draw.ellipse([cx - eye_spacing - 4, eye_y - 3, cx - eye_spacing + 4, eye_y + 3], fill=(30, 30, 30))
        draw.ellipse([cx + eye_spacing - 4, eye_y - 3, cx + eye_spacing + 4, eye_y + 3], fill=(30, 30, 30))

        # Mouth
        mouth_y = cy + rng.randint(15, 25)
        mouth_w = rng.randint(12, 22)
        draw.rectangle([cx - mouth_w // 2, mouth_y - 2, cx + mouth_w // 2, mouth_y + 2], fill=(160, 50, 50))

        return img.resize((self.img_size, self.img_size), Image.BILINEAR)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._render_synthetic_face(idx)
        return self.transform(img)


def get_dataloader(
    dataset_path: Optional[str] = None,
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    use_synthetic: bool = False,
    synthetic_samples: int = 500,
) -> DataLoader:
    """Builds and returns a DataLoader for training or evaluation."""
    if use_synthetic or dataset_path is None or not os.path.exists(dataset_path):
        dataset = SyntheticFaceDataset(num_samples=synthetic_samples, img_size=img_size)
    else:
        dataset = LocalImageFolderDataset(
            root_dir=dataset_path,
            img_size=img_size,
            max_samples=max_samples,
        )
        if len(dataset) == 0:
            print(f"Warning: No valid images found in {dataset_path}. Falling back to synthetic dataset.")
            dataset = SyntheticFaceDataset(num_samples=synthetic_samples, img_size=img_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader
