"""Export Lightweight Production Checkpoint and Compare Candidates."""

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
from models.style_generator import StyleGenerator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    source_path = Path("experiments/phase3_convergence/checkpoints/checkpoint_final.pt")
    assert source_path.exists(), f"Source checkpoint not found at {source_path}"

    data = torch.load(source_path, map_location="cpu", weights_only=False)
    raw_state = {k.replace("module.", ""): v for k, v in data["generator"].items()}
    ema_state = {k.replace("module.", ""): v for k, v in data["ema_generator"].items()}

    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "styleforge_final_model.pt"

    export_payload = {
        "model_name": "StyleForge Production Face Generator",
        "global_step": 1500,
        "img_size": 64,
        "latent_dim": 512,
        "style_dim": 512,
        "base_channels": 512,
        "max_channels": 512,
        "generator": raw_state,
        "ema_generator": ema_state,
        "metrics": {
            "raw_fid": 297.76,
            "ema_fid": 336.98,
            "raw_kid": 0.3285,
            "ema_kid": 0.4167,
        },
        "license": "MIT",
        "format_version": "1.0",
    }

    torch.save(export_payload, out_path)
    sz_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported production checkpoint to {out_path} ({sz_mb:.2f} MB)")

    # Qualitative comparison between Raw and EMA on fixed seeds
    g_raw = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    g_raw.load_state_dict(raw_state)
    g_raw.eval()

    g_ema = StyleGenerator(latent_dim=512, style_dim=512, img_size=64).to(device)
    g_ema.load_state_dict(ema_state)
    g_ema.eval()

    torch.manual_seed(999)
    z = torch.randn(8, 512, device=device)

    with torch.no_grad():
        raw_faces = g_raw(z, truncation_psi=0.8).cpu()
        ema_faces = g_ema(z, truncation_psi=0.8).cpu()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    for i in range(8):
        # Raw
        img_raw = (raw_faces[i] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_raw)
        axes[0, i].axis("off")
        if i == 0: axes[0, i].set_title("Raw Generator (Step 1500)\nFID: 297.76", fontsize=10, loc="left")

        # EMA
        img_ema = (ema_faces[i] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[1, i].imshow(img_ema)
        axes[1, i].axis("off")
        if i == 0: axes[1, i].set_title("EMA Generator (Step 1500)\nFID: 336.98", fontsize=10, loc="left")

    plt.suptitle("StyleForge Final Model Candidates: Raw vs EMA (Step 1500, Fixed Latent Seeds)", fontsize=13, y=1.02)
    plt.tight_layout()
    comp_path = Path("reports/candidate_comparison_raw_vs_ema.png")
    plt.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visual candidate comparison saved to {comp_path}")

if __name__ == "__main__":
    main()
