"""CelebA Dataset Download & Preparation Helper.

Provides:
- Step-by-step guidance and automatic extraction for CelebA face dataset
- Dataset terms of use and non-commercial licensing acknowledgment
- Option to extract existing zip archive (--zip_path)
- Utility to generate local mock/sample dataset for explicit smoke testing (--mock)
"""

import argparse
import os
from pathlib import Path
import zipfile
from PIL import Image, ImageDraw
import numpy as np


def generate_mock_faces(target_dir: Path, count: int = 500, img_size: int = 128):
    """Generates synthetic face dataset on disk for explicit smoke tests."""
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


def extract_zip(zip_path: str, target_dir: Path):
    """Extracts CelebA zip archive to destination folder."""
    zip_p = Path(zip_path)
    if not zip_p.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} into {target_dir}...")
    with zipfile.ZipFile(zip_p, 'r') as z:
        z.extractall(target_dir)
    print(f"Extraction complete! Files extracted to {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="CelebA preparation helper")
    parser.add_argument("--mock", action="store_true", help="Generate mock faces dataset locally for smoke testing")
    parser.add_argument("--count", type=int, default=500, help="Number of mock faces to generate")
    parser.add_argument("--dest", type=str, default="data/celeba/img_align_celeba", help="Destination folder")
    parser.add_argument("--zip_path", type=str, default=None, help="Path to local img_align_celeba.zip archive to extract")
    args = parser.parse_args()

    dest_path = Path(args.dest)

    if args.zip_path:
        extract_zip(args.zip_path, dest_path)
        return

    if args.mock:
        generate_mock_faces(dest_path, count=args.count)
        return

    print("=" * 65)
    print("CelebA Dataset Setup & Licensing Instructions")
    print("=" * 65)
    print("""
TERMS OF USE:
The CelebFaces Attributes Dataset (CelebA) is available strictly for
non-commercial research purposes only. By using CelebA, you agree to
respect the original copyright and terms outlined by MMLab, CUHK.

Reference:
Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang,
"Deep Learning Face Attributes in the Wild", ICCV 2015.

HOW TO DOWNLOAD & CONFIGURE:
1. Download 'img_align_celeba.zip' (approx. 1.34 GB) from:
   - Official CUHK MMLab page: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   - Or Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

2. Extract into your StyleForge directory:
   python scripts/download_celeba.py --zip_path path/to/img_align_celeba.zip --dest data/celeba/img_align_celeba

3. Verify dataset integrity:
   python scripts/verify_dataset.py --data_dir data/celeba/img_align_celeba

4. For quick pipeline validation without large downloads:
   python scripts/download_celeba.py --mock --count 500
""")


if __name__ == "__main__":
    main()
