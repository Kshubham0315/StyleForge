"""CelebA Dataset Download & Preparation Helper.

Provides:
- Step-by-step guidance and download options for CelebA face dataset
- Utility to generate local mock/sample dataset for instant training validation
"""

import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def generate_mock_faces(target_dir: Path, count: int = 500, img_size: int = 128):
    """Generates synthetic face dataset on disk for testing pipelines."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {count} mock face images in {target_dir}...")

    for i in range(count):
        rng = np.random.RandomState(i)
        img = Image.new("RGB", (img_size, img_size), color=(30, 30, 45))
        draw = ImageDraw.Draw(img)

        # Random skin color
        r = rng.randint(180, 255)
        g = rng.randint(130, 210)
        b = rng.randint(100, 180)
        skin_color = (r, g, b)

        # Head oval
        head_w = rng.randint(int(img_size * 0.4), int(img_size * 0.6))
        head_h = rng.randint(int(img_size * 0.5), int(img_size * 0.75))
        cx, cy = img_size // 2, img_size // 2
        draw.ellipse([cx - head_w // 2, cy - head_h // 2, cx + head_w // 2, cy + head_h // 2], fill=skin_color)

        # Eyes
        eye_y = cy - rng.randint(int(img_size * 0.05), int(img_size * 0.15))
        eye_spacing = rng.randint(int(img_size * 0.12), int(img_size * 0.2))
        eye_r = max(2, int(img_size * 0.03))
        draw.ellipse([cx - eye_spacing - eye_r, eye_y - eye_r, cx - eye_spacing + eye_r, eye_y + eye_r], fill=(20, 20, 20))
        draw.ellipse([cx + eye_spacing - eye_r, eye_y - eye_r, cx + eye_spacing + eye_r, eye_y + eye_r], fill=(20, 20, 20))

        # Mouth
        mouth_y = cy + rng.randint(int(img_size * 0.12), int(img_size * 0.22))
        mouth_w = rng.randint(int(img_size * 0.1), int(img_size * 0.2))
        draw.rectangle([cx - mouth_w // 2, mouth_y - 2, cx + mouth_w // 2, mouth_y + 2], fill=(160, 40, 40))

        img.save(target_dir / f"{i:06d}.jpg", quality=95)

    print(f"Done! {count} mock images saved to {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="CelebA preparation helper")
    parser.add_argument("--mock", action="store_true", help="Generate mock faces dataset locally for quick testing")
    parser.add_argument("--count", type=int, default=500, help="Number of mock faces to generate")
    parser.add_argument("--dest", type=str, default="data/celeba/img_align_celeba", help="Destination folder")
    args = parser.parse_args()

    dest_path = Path(args.dest)

    if args.mock:
        generate_mock_faces(dest_path, count=args.count)
        return

    print("==================================================")
    print("CelebA Dataset Setup Instructions")
    print("==================================================")
    print("""
To train StyleForge on real celebrity faces:
1. Download 'img_align_celeba.zip' from the official CelebA page:
   https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   or via Kaggle:
   https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

2. Unzip into:
   StyleForge/data/celeba/img_align_celeba/
   (so that images are at StyleForge/data/celeba/img_align_celeba/000001.jpg, etc.)

3. To practice or test immediately without downloading 1.4 GB, run:
   python scripts/download_celeba.py --mock --count 500
""")


if __name__ == "__main__":
    main()
