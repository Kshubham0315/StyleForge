"""StyleForge Phase 4 Training Continuation Script.

Runs training continuation using the winning configuration (Exp_D):
- Logistic GAN loss + Lazy R1 Regularization (k=16, gamma=10.0)
- TTUR (lr_g=0.0005, lr_d=0.0010, betas=(0.0, 0.99))
- Data Augmentation (Horizontal Flip 0.5, Rotation +-5 deg, Color Jitter 10%)
- Mixed Precision (torch.amp FP16) on CUDA
- Restores checkpoint_final.pt from Phase 3 (Step 1,500)
- Configurable target steps via CLI argument
- Automatic checkpointing: latest, best, and periodic steps
- Fixed-seed visual snapshots saved every interval
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from data.dataset import CelebADataset, FaceTransform
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.trainer import GANTrainer



def parse_args():
    parser = argparse.ArgumentParser(description="StyleForge Phase 4 Training Continuation")
    parser.add_argument("--source_ckpt", type=str, default="experiments/phase3_convergence/checkpoints/checkpoint_final.pt",
                        help="Path to starting checkpoint")
    parser.add_argument("--exp_dir", type=str, default="experiments/phase4_training_continuation",
                        help="Output experiment directory")
    parser.add_argument("--target_steps", type=int, default=2000,
                        help="Target total optimizer steps (e.g. 2000, 2500, 5000, 20000)")
    parser.add_argument("--sample_every", type=int, default=100,
                        help="Interval for saving visual grid samples")
    parser.add_argument("--checkpoint_every", type=int, default=250,
                        help="Interval for saving periodic checkpoints")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Minibatch size")
    parser.add_argument("--lr_g", type=float, default=0.0005,
                        help="Generator learning rate (TTUR)")
    parser.add_argument("--lr_d", type=float, default=0.0010,
                        help="Critic learning rate (TTUR)")
    parser.add_argument("--lazy_interval", type=int, default=16,
                        help="Lazy R1 interval k")
    parser.add_argument("--r1_gamma", type=float, default=10.0,
                        help="R1 regularization strength")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== StyleForge Phase 4 Training Continuation ===")
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Source Checkpoint: {args.source_ckpt}")
    print(f"Target Steps: {args.target_steps}")
    print(f"Batch Size: {args.batch_size}")
    print(f"TTUR Learning Rates: lr_g={args.lr_g}, lr_d={args.lr_d}")
    print(f"Regularization: Lazy R1 (interval={args.lazy_interval}, gamma={args.r1_gamma})")

    # 1. Dataset & Augmented DataLoader
    train_transform = FaceTransform(
        img_size=64,
        horizontal_flip=True,
        rotation=True,
        rotation_degrees=5.0,
        color_jitter=True,
        brightness=0.10,
        contrast=0.05,
    )
    dataset = CelebADataset(
        "data/celeba/img_align_celeba",
        split="train",
        rotation=True,
        color_jitter=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )
    print(f"Training Dataset: {len(dataset)} images ({len(dataloader)} batches/epoch)")

    # 2. Architectures
    generator = StyleGenerator(
        latent_dim=512,
        style_dim=512,
        num_mapping_layers=8,
        img_size=64,
        base_channels=512,
        max_channels=512,
    ).to(device)

    critic = StyleCritic(
        img_size=64,
        base_channels=64,
        max_channels=512,
        use_minibatch_std=True,
    ).to(device)

    # 3. Training Config
    config = {
        "experiment": {
            "name": "phase4_training_continuation",
            "exp_dir": args.exp_dir,
            "seed": 42,
        },
        "model": {
            "image_size": 64,
            "latent_dim": 512,
            "base_channels": 512,
            "max_channels": 512,
        },
        "training": {
            "batch_size": args.batch_size,
            "n_critic": 1,  # 1:1 balance for Logistic + Lazy R1
            "loss_type": "logistic",
            "lazy_r1": True,
            "lazy_interval": args.lazy_interval,
            "r1_gamma": args.r1_gamma,
            "lr_g": args.lr_g,
            "lr_d": args.lr_d,
            "beta1": 0.0,
            "beta2": 0.99,
            "amp": True,
            "max_grad_norm": 10.0,
        },
    }

    # 4. Initialize Trainer & Load Checkpoint
    trainer = GANTrainer(generator, critic, dataloader, config, device)

    # Check if a continuation checkpoint already exists in exp_dir
    resume_ckpt = Path(args.exp_dir) / "checkpoints" / "checkpoint_latest.pt"
    if resume_ckpt.exists():
        print(f"Found existing continuation checkpoint at {resume_ckpt}. Resuming from here...")
        trainer.load_checkpoint(str(resume_ckpt))
    else:
        print(f"Resuming from verified source checkpoint: {args.source_ckpt}")
        trainer.load_checkpoint(args.source_ckpt)

    print(f"Starting training continuation from Step {trainer.global_step} to {args.target_steps}...")

    # 5. Run Training
    trainer.train(
        total_steps=args.target_steps,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
    )

    print(f"Phase 4 continuation run completed successfully! Final Step: {trainer.global_step}")


if __name__ == "__main__":
    main()
