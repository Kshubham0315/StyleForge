"""Controlled Pilot Training Experiments across Configurations A, B, C, and D."""

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from data.dataset import CelebADataset
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.trainer import GANTrainer

def run_pilot_experiments(pilot_steps: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"STYLEFORGE PHASE 4 CONTROLLED PILOT EXPERIMENTS ({device})")
    print(f"Pilot Step Budget per Condition: {pilot_steps} generator steps")
    print("=" * 70)

    base_ckpt_path = Path("experiments/phase3_convergence/checkpoints/checkpoint_final.pt")
    assert base_ckpt_path.exists(), f"Source checkpoint not found at {base_ckpt_path}"
    base_state = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)

    conditions = [
        {
            "id": "Exp_A",
            "name": "Baseline Continuation (AMP, WGAN-GP, No Extra Aug)",
            "loss_mode": "wgan_gp",
            "penalty_weight": 10.0,
            "lazy_r1": False,
            "rotation": False,
            "color_jitter": False,
            "lr_g": 0.0001,
            "lr_d": 0.0001,
        },
        {
            "id": "Exp_B",
            "name": "AMP + Data Augmentation (Flip + Rot +-5° + Jitter 10%)",
            "loss_mode": "wgan_gp",
            "penalty_weight": 10.0,
            "lazy_r1": False,
            "rotation": True,
            "color_jitter": True,
            "lr_g": 0.0001,
            "lr_d": 0.0001,
        },
        {
            "id": "Exp_C",
            "name": "AMP + Augmentation + TTUR (lr_G=0.0005, lr_D=0.001)",
            "loss_mode": "wgan_gp",
            "penalty_weight": 10.0,
            "lazy_r1": False,
            "rotation": True,
            "color_jitter": True,
            "lr_g": 0.0005,
            "lr_d": 0.0010,
        },
        {
            "id": "Exp_D",
            "name": "AMP + Augmentation + TTUR + Lazy R1 (Interval=16)",
            "loss_mode": "logistic",
            "penalty_weight": 10.0,
            "lazy_r1": True,
            "lazy_interval": 16,
            "rotation": True,
            "color_jitter": True,
            "lr_g": 0.0005,
            "lr_d": 0.0010,
        },
    ]

    results = []
    base_exp_dir = Path("experiments/phase4_controlled_pilots")
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        cid = cond["id"]
        cname = cond["name"]
        print(f"\n---> Running [{cid}]: {cname}")

        exp_dir = base_exp_dir / cid
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Build fresh dataset for condition with appropriate augmentations
        train_ds = CelebADataset(
            "data/celeba/img_align_celeba",
            split="train",
            rotation=cond["rotation"],
            color_jitter=cond["color_jitter"],
        )
        loader = DataLoader(
            train_ds,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        # Fresh model instances
        gen = StyleGenerator(latent_dim=512, style_dim=512, img_size=64, base_channels=512, max_channels=512)
        critic = StyleCritic(img_size=64, base_channels=64, max_channels=512, use_minibatch_std=True)

        config = {
            "experiment": {"name": cid, "exp_dir": str(exp_dir), "seed": 42},
            "training": {
                "amp": True,
                "batch_size": 16,
                "n_critic": 4,
                "lr_g": cond["lr_g"],
                "lr_d": cond["lr_d"],
                "loss_mode": cond["loss_mode"],
                "penalty_weight": cond["penalty_weight"],
                "lazy_r1": cond.get("lazy_r1", False),
                "lazy_interval": cond.get("lazy_interval", 16),
                "style_mixing_prob": 0.3,
                "ema_decay": 0.999,
                "max_grad_norm": 10.0,
            },
            "model": {"latent_dim": 512, "style_dim": 512, "img_size": 64},
        }

        trainer = GANTrainer(gen, critic, loader, config, device=device)
        trainer.load_checkpoint(str(base_ckpt_path))
        start_step = trainer.global_step
        target_step = start_step + pilot_steps

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        trainer.train(total_steps=target_step, sample_every=pilot_steps // 2, checkpoint_every=pilot_steps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_vram_mb = torch.cuda.max_memory_reserved() / (1024**2)
        sec_per_step = elapsed / pilot_steps
        steps_per_sec = pilot_steps / elapsed

        # Extract recent metric averages
        recent = trainer.history[-min(10, len(trainer.history)):]
        avg_g_loss = sum(h.get("g_loss", 0.0) for h in recent) / len(recent)
        avg_penalty = sum(h.get("penalty", 0.0) for h in recent) / len(recent)
        avg_d_grad = sum(h.get("critic_grad_norm", 0.0) for h in recent) / len(recent)
        avg_g_grad = sum(h.get("gen_grad_norm", 0.0) for h in recent) / len(recent)

        res_record = {
            "id": cid,
            "name": cname,
            "steps": pilot_steps,
            "elapsed_sec": round(elapsed, 2),
            "sec_per_step": round(sec_per_step, 4),
            "steps_per_sec": round(steps_per_sec, 2),
            "peak_allocated_vram_mb": round(peak_vram_mb, 1),
            "peak_reserved_vram_mb": round(reserved_vram_mb, 1),
            "avg_g_loss": round(avg_g_loss, 4),
            "avg_penalty": round(avg_penalty, 4),
            "avg_critic_grad_norm": round(avg_d_grad, 4),
            "avg_gen_grad_norm": round(avg_g_grad, 4),
            "samples_path": str(exp_dir / "samples"),
        }
        results.append(res_record)
        print(f"Finished {cid}: {sec_per_step:.3f} s/step ({steps_per_sec:.2f} it/s), VRAM: {peak_vram_mb:.0f} MB")

    out_json = Path("reports/phase4_pilot_experiment_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll pilot experiments completed. Summary written to {out_json}")

if __name__ == "__main__":
    run_pilot_experiments(pilot_steps=30)
