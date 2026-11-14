"""Generate Comprehensive Phase 5 Visual Evaluation Suite."""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.style_generator import StyleGenerator
from evaluation.latent_explorer import LatentExplorer

def to_np(img_tensor):
    return ((img_tensor * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path("assets/phase5")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path("checkpoints/styleforge_final_model.pt")
    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    generator = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    generator.load_state_dict(payload["generator"])
    generator.eval()

    ema_generator = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    ema_generator.load_state_dict(payload["ema_generator"])
    ema_generator.eval()

    # 1. Random Face Gallery (4 rows x 8 columns = 32 faces)
    print("Generating 32-face random gallery...")
    torch.manual_seed(2026)
    z_gallery = torch.randn(32, 512, device=device)
    with torch.no_grad():
        gallery_faces = generator(z_gallery, truncation_psi=0.8)

    fig, axes = plt.subplots(4, 8, figsize=(16, 8.5))
    for i in range(32):
        r, c = divmod(i, 8)
        axes[r, c].imshow(to_np(gallery_faces[i]))
        axes[r, c].axis("off")
    plt.suptitle("StyleForge v1.0 Production Release: Random Facial Generation Gallery (64x64, psi=0.8)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir / "random_face_gallery.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Latent Interpolation Walk (SLERP in Z vs LERP in W)
    print("Generating SLERP & LERP latent interpolation walks...")
    torch.manual_seed(101)
    z1 = torch.randn(1, 512, device=device)
    z2 = torch.randn(1, 512, device=device)

    steps = 10
    alphas = np.linspace(0.0, 1.0, steps)

    # SLERP in Z
    z1_norm = z1 / z1.norm(dim=1, keepdim=True)
    z2_norm = z2 / z2.norm(dim=1, keepdim=True)
    dot = (z1_norm * z2_norm).sum().clamp(-0.9999, 0.9999)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    slerp_faces = []
    with torch.no_grad():
        for a in alphas:
            s_z = (torch.sin((1.0 - a) * omega) / sin_omega) * z1 + (torch.sin(a * omega) / sin_omega) * z2
            slerp_faces.append(generator(s_z, truncation_psi=0.8)[0])

    # LERP in W
    with torch.no_grad():
        w1 = generator.mapping(z1)
        w2 = generator.mapping(z2)
        lerp_faces = []
        for a in alphas:
            lerp_w = (1.0 - a) * w1 + a * w2
            lerp_faces.append(generator.synthesis(lerp_w)[0])

    fig, axes = plt.subplots(2, steps, figsize=(18, 4))
    for i in range(steps):
        axes[0, i].imshow(to_np(slerp_faces[i]))
        axes[0, i].axis("off")
        axes[0, i].set_title(f"a={alphas[i]:.2f}", fontsize=9)

        axes[1, i].imshow(to_np(lerp_faces[i]))
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("SLERP (Z-Space)", fontsize=10)
    axes[1, 0].set_ylabel("LERP (W-Space)", fontsize=10)
    plt.suptitle("Continuous Latent Space Interpolation: SLERP in Z vs LERP in W", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_interpolation_walk.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Truncation Trick Sweep (psi = 0.0 to 1.2 across 5 distinct identities)
    print("Generating Truncation Trick Sweep...")
    torch.manual_seed(333)
    z_trunc = torch.randn(5, 512, device=device)
    psis = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]

    fig, axes = plt.subplots(5, len(psis), figsize=(14, 10))
    with torch.no_grad():
        for r_idx in range(5):
            cur_z = z_trunc[r_idx:r_idx+1]
            for c_idx, p in enumerate(psis):
                img = generator(cur_z, truncation_psi=p)[0]
                axes[r_idx, c_idx].imshow(to_np(img))
                axes[r_idx, c_idx].axis("off")
                if r_idx == 0:
                    axes[r_idx, c_idx].set_title(f"psi={p:.1f}", fontsize=11)

    plt.suptitle("Truncation Trick Sweep: psi in [0.0, 1.2] (Trade-Off Between Typicality and Fidelity)", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir / "truncation_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Multi-Scale Style Mixing Matrix
    print("Generating Style Mixing Matrix...")
    torch.manual_seed(777)
    src_a = torch.randn(4, 512, device=device)  # 4 rows
    src_b = torch.randn(4, 512, device=device)  # 4 cols

    # Style mixing across coarse layers (4x4, 8x8) vs fine layers (32x32, 64x64)
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes[0, 0].axis("off")

    with torch.no_grad():
        w_a = generator.mapping(src_a)
        w_b = generator.mapping(src_b)

        # Plot Source B header (columns)
        for c in range(4):
            img_b = generator.synthesis(w_b[c:c+1])[0]
            axes[0, c + 1].imshow(to_np(img_b))
            axes[0, c + 1].axis("off")
            axes[0, c + 1].set_title(f"Style B_{c}", fontsize=9)

        # Plot Source A header (rows) and mixed faces
        for r in range(4):
            img_a = generator.synthesis(w_a[r:r+1])[0]
            axes[r + 1, 0].imshow(to_np(img_a))
            axes[r + 1, 0].axis("off")
            axes[r + 1, 0].set_ylabel(f"Style A_{r}", fontsize=9)

            for c in range(4):
                # Mix: Coarse styles from B, Fine styles from A
                # num_layers = 10 (4x4 to 64x64)
                w_mix = [w_b[c:c+1]] * 4 + [w_a[r:r+1]] * 6
                mixed_img = generator.synthesis(w_mix)[0]
                axes[r + 1, c + 1].imshow(to_np(mixed_img))
                axes[r + 1, c + 1].axis("off")

    plt.suptitle("StyleForge Style Mixing Matrix (Coarse Pose from Source B, Fine Appearance from Source A)", fontsize=11, y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir / "style_mixing_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"All visual assets saved to {out_dir}")

if __name__ == "__main__":
    main()
