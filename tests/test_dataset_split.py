"""Unit tests for CelebA dataset partitions, manifest integrity, and split disjointness."""

import json
from pathlib import Path
import pytest
import torch

from data.dataset import CelebADataset, get_dataloader


def test_split_manifests_exist_and_disjoint():
    base_dir = Path(__file__).resolve().parent.parent / "data" / "celeba"
    train_manifest = base_dir / "train_manifest.json"
    val_manifest = base_dir / "val_manifest.json"
    test_manifest = base_dir / "test_manifest.json"

    if not train_manifest.exists():
        pytest.skip("Dataset manifests not present (environment without downloaded dataset).")

    with open(train_manifest, "r") as f:
        train_data = json.load(f)
    with open(val_manifest, "r") as f:
        val_data = json.load(f)
    with open(test_manifest, "r") as f:
        test_data = json.load(f)

    train_set = set(train_data["images"])
    val_set = set(val_data["images"])
    test_set = set(test_data["images"])

    # Strict disjointness: No image may appear in more than one partition
    assert len(train_set.intersection(val_set)) == 0, "Train and Val splits overlap!"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test splits overlap!"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test splits overlap!"

    # Total sample integrity
    assert len(train_set) == 4912, f"Expected 4912 train images, got {len(train_set)}"
    assert len(val_set) == 614, f"Expected 614 val images, got {len(val_set)}"
    assert len(test_set) == 614, f"Expected 614 test images, got {len(test_set)}"


def test_celeba_dataset_partition_loading():
    img_dir = Path(__file__).resolve().parent.parent / "data" / "celeba" / "img_align_celeba"
    if not img_dir.exists() or len(list(img_dir.glob("*.jpg"))) == 0:
        pytest.skip("Dataset images not found on disk.")

    val_dataset = CelebADataset(
        root_dir=str(img_dir),
        split="val",
        img_size=64,
    )
    assert len(val_dataset) == 614
    sample = val_dataset[0]
    assert sample.shape == (3, 64, 64)
    assert -1.0 <= sample.min() and sample.max() <= 1.0
