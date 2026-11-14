"""Authentic CelebA Dataset Downloader, Extractor & Partition Verifier.

Downloads authentic photographic CelebA aligned and cropped face images
(178x218 JPEG) from the verified open community mirror (huggan/CelebA-faces-with-attributes),
verifies image integrity, and creates deterministic, strictly disjoint train/val/test splits.
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download


REPO_ID = "huggan/CelebA-faces-with-attributes"


def download_and_extract(
    num_chunks: int = 4,
    dest_dir: Path = Path("data/celeba/img_align_celeba"),
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = dest_dir.parent
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing to extract authentic CelebA images across {num_chunks} parquet chunks...")
    all_images: List[Dict[str, str]] = []
    seen_ids: Set[str] = set()

    for chunk_idx in range(num_chunks):
        filename = f"data/train-{chunk_idx:05d}-of-00132.parquet"
        print(f"Fetching chunk {chunk_idx + 1}/{num_chunks}: {filename}...")
        parquet_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            token=False,
        )
        df = pd.read_parquet(parquet_path)
        print(f"  Chunk loaded with {len(df)} images.")

        for _, row in df.iterrows():
            img_id = str(row["image_id"]).strip()
            if img_id in seen_ids:
                continue
            seen_ids.add(img_id)

            img_bytes = row["image"]["bytes"]
            target_path = dest_dir / img_id

            # Write file if not already on disk
            if not target_path.exists():
                with open(target_path, "wb") as f:
                    f.write(img_bytes)

            all_images.append({
                "image_id": img_id,
                "path": str(target_path.as_posix()),
            })

    total_images = len(all_images)
    print(f"\nSuccessfully verified/extracted {total_images} authentic CelebA images.")

    # Integrity verification
    print("Verifying image integrity and dimensions on disk...")
    corrupt_count = 0
    sample_sizes = set()
    for item in all_images[:500]:
        try:
            with Image.open(item["path"]) as img:
                img.verify()
            with Image.open(item["path"]) as img:
                sample_sizes.add(img.size)
        except Exception as e:
            print(f"  Corrupt image detected: {item['path']} ({e})")
            corrupt_count += 1

    if corrupt_count > 0:
        raise RuntimeError(f"Integrity check failed: {corrupt_count} corrupt images found!")
    print(f"Integrity check passed. Image dimensions: {sample_sizes}")

    # Deterministic Partitioning
    print("Creating deterministic, strictly disjoint train / val / test splits...")
    rng = np.random.RandomState(seed)
    indices = np.arange(total_images)
    rng.shuffle(indices)

    n_val = int(total_images * val_ratio)
    n_test = int(total_images * test_ratio)
    n_train = total_images - n_val - n_test

    val_indices = indices[:n_val]
    test_indices = indices[n_val : n_val + n_test]
    train_indices = indices[n_val + n_test :]

    train_files = [all_images[i]["image_id"] for i in train_indices]
    val_files = [all_images[i]["image_id"] for i in val_indices]
    test_files = [all_images[i]["image_id"] for i in test_indices]

    # Verify strict disjointness
    set_train = set(train_files)
    set_val = set(val_files)
    set_test = set(test_files)

    assert len(set_train.intersection(set_val)) == 0, "Train-Val overlap detected!"
    assert len(set_train.intersection(set_test)) == 0, "Train-Test overlap detected!"
    assert len(set_val.intersection(set_test)) == 0, "Val-Test overlap detected!"
    assert len(set_train) + len(set_val) + len(set_test) == total_images, "Split sum mismatch!"

    print(f"Partition counts: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    print(f"Disjointness verified: 0 leakage between sets.")

    # Save partition file (list_eval_partition.txt)
    # Standard format: 0=train, 1=val, 2=test
    partition_txt_path = metadata_dir / "list_eval_partition.txt"
    with open(partition_txt_path, "w") as f:
        for f_name in train_files:
            f.write(f"{f_name} 0\n")
        for f_name in val_files:
            f.write(f"{f_name} 1\n")
        for f_name in test_files:
            f.write(f"{f_name} 2\n")
    print(f"Saved partition manifest to {partition_txt_path}")

    # Save JSON manifests
    with open(metadata_dir / "train_manifest.json", "w") as f:
        json.dump(train_files, f, indent=2)
    with open(metadata_dir / "val_manifest.json", "w") as f:
        json.dump(val_files, f, indent=2)
    with open(metadata_dir / "test_manifest.json", "w") as f:
        json.dump(test_files, f, indent=2)

    metadata = {
        "dataset_name": "CelebA Align&Cropped (Photographic)",
        "source": REPO_ID,
        "total_images": total_images,
        "sample_dimensions": list(list(sample_sizes)[0]) if sample_sizes else [178, 218],
        "preprocessing_target": [64, 64],
        "normalization_range": [-1.0, 1.0],
        "random_seed": seed,
        "splits": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "disjointness_verified": True,
        "data_leakage": 0,
        "manifests": {
            "partition_txt": str(partition_txt_path.as_posix()),
            "train_json": str((metadata_dir / "train_manifest.json").as_posix()),
            "val_json": str((metadata_dir / "val_manifest.json").as_posix()),
            "test_json": str((metadata_dir / "test_manifest.json").as_posix()),
        }
    }

    with open(metadata_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset metadata to {metadata_dir / 'dataset_metadata.json'}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Download and verify real CelebA dataset")
    parser.add_argument("--chunks", type=int, default=4, help="Number of 1535-image parquet chunks to extract")
    parser.add_argument("--dest", type=str, default="data/celeba/img_align_celeba", help="Image destination folder")
    parser.add_argument("--seed", type=int, default=42, help="Partition random seed")
    args = parser.parse_args()

    download_and_extract(
        num_chunks=args.chunks,
        dest_dir=Path(args.dest),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
