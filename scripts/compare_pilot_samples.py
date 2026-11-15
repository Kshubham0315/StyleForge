from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def make_comparison_grid():
    conditions = ["Exp_A", "Exp_B", "Exp_C", "Exp_D"]
    labels = [
        "A: Baseline WGAN-GP (1.19s/step)",
        "B: AMP + Augmentation (1.16s/step)",
        "C: Aug + TTUR (1.21s/step)",
        "D: Aug + TTUR + Lazy R1 (0.72s/step)",
    ]

    images = []
    for c in conditions:
        img_path = Path(f"experiments/phase4_controlled_pilots/{c}/samples/step_001530.png")
        if img_path.exists():
            im = Image.open(img_path)
            images.append(im)
        else:
            print(f"Missing {img_path}")
            return

    # Each step_001530.png is a 4x4 grid of 64x64 faces = 256x256 (or similar)
    w, h = images[0].size
    header_h = 30
    composite = Image.new("RGB", (w * 4, h + header_h), color=(20, 20, 25))
    draw = ImageDraw.Draw(composite)

    for i, (im, lbl) in enumerate(zip(images, labels)):
        composite.paste(im, (i * w, header_h))
        draw.text((i * w + 10, 8), lbl, fill=(255, 255, 255))

    out_dir = Path("reports/visual_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase4_pilot_comparison.png"
    composite.save(out_path)
    print(f"Saved pilot comparison grid to {out_path}")

    # Copy to brain artifacts
    brain_dir = Path("C:/Users/Shubham Kumar/.gemini/antigravity/brain/5b587fc8-3c27-4ac2-8b80-9ce8669cd5db")
    if brain_dir.exists():
        import shutil
        shutil.copy(out_path, brain_dir / "phase4_pilot_comparison.png")
        print("Copied to brain artifacts directory.")

if __name__ == "__main__":
    make_comparison_grid()
