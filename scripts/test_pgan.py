import torch
import torchvision
from pathlib import Path

def test():
    device = torch.cuda.is_available()
    print(f"Loading model (CUDA={device})...")
    model = torch.hub.load(
        'facebookresearch/pytorch_GAN_zoo:hub',
        'PGAN',
        model_name='celebAHQ-512',
        pretrained=True,
        useGPU=device,
    )
    
    # Generate 4 high quality faces with deterministic seed
    torch.manual_seed(42)
    noise, _ = model.buildNoiseData(4)
    with torch.no_grad():
        images = model.test(noise)
    
    # Scale to [0, 1]
    images = (images.clamp(-1.0, 1.0) * 0.5 + 0.5)
    
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "test_pgan_celeb.png"
    torchvision.utils.save_image(images, out_path, nrow=2)
    print(f"Saved generated faces to {out_path} ({images.shape})")

if __name__ == "__main__":
    test()
