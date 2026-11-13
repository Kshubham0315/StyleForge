"""StyleForge GAN Trainer.

Provides a unified, robust training loop featuring:
- Exponential Moving Average (EMA) of Generator parameters
- Gradient Penalty / R1 regularization (FP32 safe)
- n_critic updates per generator step
- Mixed precision AMP training (fp16) on CUDA
- NaN / Inf gradient anomaly detection & optional gradient clipping
- TensorBoard logging + JSON metrics history
- VRAM usage & training throughput monitoring
- Git commit hash & environment snapshot in checkpoint metadata
"""

import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from training.losses import GANLoss


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash if in a git repo."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


class ExponentialMovingAverage:
    """Maintains an exponential moving average of model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

        for s_buffer, m_buffer in zip(self.shadow.buffers(), model.buffers()):
            s_buffer.copy_(m_buffer)


class GANTrainer:
    """Modular GAN Trainer for Baseline WGAN-GP and Style-Based Generators."""

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        dataloader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader

        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)

        # EMA generator
        ema_decay = float(config.get("training", {}).get("ema_decay", 0.999))
        self.ema_generator = ExponentialMovingAverage(self.generator, decay=ema_decay)

        # Optimizers
        lr_g = float(config.get("training", {}).get("lr_g", 0.0002))
        lr_d = float(config.get("training", {}).get("lr_d", 0.0002))
        beta1 = float(config.get("training", {}).get("beta1", 0.0))
        beta2 = float(config.get("training", {}).get("beta2", 0.99))

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.opt_d = torch.optim.Adam(self.critic.parameters(), lr=lr_d, betas=(beta1, beta2))
        self.lr_g = lr_g
        self.lr_d = lr_d

        # Loss
        loss_mode = config.get("training", {}).get("loss_mode", "wgan_gp")
        penalty_weight = float(config.get("training", {}).get("penalty_weight", 10.0))
        r1_gamma = float(config.get("training", {}).get("r1_gamma", 10.0))
        self.loss_fn = GANLoss(mode=loss_mode, penalty_weight=penalty_weight, r1_gamma=r1_gamma)

        # Training parameters
        self.n_critic = int(config.get("training", {}).get("n_critic", 5))
        self.latent_dim = int(config.get("model", {}).get("latent_dim", 512))
        self.style_mixing_prob = float(config.get("training", {}).get("style_mixing_prob", 0.0))
        self.max_grad_norm = config.get("training", {}).get("max_grad_norm", None)

        # Output dirs
        self.exp_dir = Path(config.get("experiment", {}).get("exp_dir", "experiments/default"))
        self.samples_dir = self.exp_dir / "samples"
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer (if available)
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.exp_dir / "tb_logs"))
        except Exception:
            pass

        # Save config snapshot with git commit
        config_snapshot = copy.deepcopy(config)
        config_snapshot["git_commit"] = get_git_commit_hash()
        config_snapshot["device"] = str(self.device)
        if torch.cuda.is_available():
            config_snapshot["gpu_name"] = torch.cuda.get_device_name(0)

        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(config_snapshot, f, indent=2)

        # Fixed latents for progress snapshots (seed fixed)
        g_cpu = torch.Generator().manual_seed(42)
        n_fixed = min(16, config.get("training", {}).get("batch_size", 16))
        self.fixed_z = torch.randn(n_fixed, self.latent_dim, generator=g_cpu).to(self.device)

        # AMP
        use_amp = config.get("training", {}).get("amp", True) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.use_amp = use_amp

        self.global_step = 0
        self.history = []
        self.start_time = None

    def save_checkpoint(self, filename: str = "checkpoint_latest.pt"):
        """Save model, EMA, and optimizer states."""
        ckpt_path = self.checkpoints_dir / filename
        state = {
            "global_step": self.global_step,
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "ema_generator": self.ema_generator.shadow.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "config": self.config,
            "git_commit": get_git_commit_hash(),
            "timestamp": time.time(),
        }
        torch.save(state, ckpt_path)
        print(f"Checkpoint saved at step {self.global_step}: {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str):
        """Restore model, EMA, and optimizer states."""
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(state["generator"])
        self.critic.load_state_dict(state["critic"])
        self.ema_generator.shadow.load_state_dict(state["ema_generator"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.global_step = state["global_step"]
        print(f"Resumed from checkpoint: {ckpt_path} at step {self.global_step}")

    @torch.no_grad()
    def save_sample_grid(self, step: int):
        """Generate and save image grid using EMA generator."""
        self.ema_generator.shadow.eval()
        fake = self.ema_generator.shadow(self.fixed_z)
        # Unnormalize [-1, 1] to [0, 1]
        fake = (fake * 0.5 + 0.5).clamp(0.0, 1.0)
        grid_path = self.samples_dir / f"step_{step:06d}.png"
        save_image(fake, grid_path, nrow=4)

        if self.tb_writer is not None:
            self.tb_writer.add_images("EMA_Generated_Samples", fake, global_step=step)

    def train_step_critic(self, real_images: torch.Tensor) -> Dict[str, float]:
        """Single training step for critic."""
        batch_size = real_images.size(0)
        self.opt_d.zero_grad(set_to_none=True)

        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            if hasattr(self.generator, "style_dim"):
                fake_images = self.generator(z, style_mixing_prob=self.style_mixing_prob, update_w_avg=True)
            else:
                fake_images = self.generator(z)

            loss_d, d_stats = self.loss_fn.critic_loss(self.critic, real_images, fake_images)

        # Gradient anomaly detection
        if torch.isnan(loss_d) or torch.isinf(loss_d):
            print(f"[Warning] Critic loss NaN/Inf at step {self.global_step}. Skipping step.")
            return d_stats

        self.scaler.scale(loss_d).backward()

        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.scaler.step(self.opt_d)
        self.scaler.update()

        return d_stats

    def train_step_generator(self, batch_size: int) -> Dict[str, float]:
        """Single training step for generator."""
        self.opt_g.zero_grad(set_to_none=True)

        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            if hasattr(self.generator, "style_dim"):
                fake_images = self.generator(z, style_mixing_prob=self.style_mixing_prob)
            else:
                fake_images = self.generator(z)

            loss_g, g_stats = self.loss_fn.generator_loss(self.critic, fake_images)

        if torch.isnan(loss_g) or torch.isinf(loss_g):
            print(f"[Warning] Generator loss NaN/Inf at step {self.global_step}. Skipping step.")
            return g_stats

        self.scaler.scale(loss_g).backward()

        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)

        self.scaler.step(self.opt_g)
        self.scaler.update()

        # Update EMA weights
        self.ema_generator.update(self.generator)

        return g_stats

    def train(self, total_steps: int, sample_every: int = 250, checkpoint_every: int = 1000):
        """Main training loop driven by total generator steps."""
        self.start_time = time.time()
        print(f"Starting training on {self.device} for {total_steps} steps...")
        data_iter = iter(self.dataloader)

        pbar = tqdm(total=total_steps, initial=self.global_step, desc="Training StyleForge")

        while self.global_step < total_steps:
            # Train Critic n_critic times
            d_stats_accum = {}
            for _ in range(self.n_critic):
                try:
                    real_images = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    real_images = next(data_iter)

                if isinstance(real_images, (list, tuple)):
                    real_images = real_images[0]

                real_images = real_images.to(self.device, non_blocking=True)
                d_stats = self.train_step_critic(real_images)

                for k, v in d_stats.items():
                    d_stats_accum[k] = d_stats_accum.get(k, 0.0) + v / self.n_critic

            # Train Generator once
            batch_size = real_images.size(0)
            g_stats = self.train_step_generator(batch_size)
            self.global_step += 1
            pbar.update(1)

            # Performance stats
            elapsed = time.time() - self.start_time
            max_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if self.device.type == "cuda" else 0.0

            step_metrics = {
                "step": self.global_step,
                "elapsed_sec": round(elapsed, 2),
                "vram_mb": round(max_vram_mb, 1),
                "lr_g": self.lr_g,
                "lr_d": self.lr_d,
                **d_stats_accum,
                **g_stats,
            }
            self.history.append(step_metrics)

            # TensorBoard logging
            if self.tb_writer is not None:
                for k, v in step_metrics.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f"train/{k}", v, global_step=self.global_step)

            # Update progress bar
            pbar.set_postfix({
                "W-Dist": f"{d_stats_accum.get('wasserstein_distance', 0.0):.3f}",
                "G_Loss": f"{g_stats.get('g_loss', 0.0):.3f}",
                "GP/R1": f"{d_stats_accum.get('penalty', 0.0):.3f}",
                "VRAM": f"{max_vram_mb:.0f}M",
            })

            # Periodic visual snapshots
            if self.global_step % sample_every == 0 or self.global_step == 1:
                self.save_sample_grid(self.global_step)

            # Periodic checkpointing
            if self.global_step % checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step:06d}.pt")
                self.save_checkpoint("checkpoint_latest.pt")
                with open(self.exp_dir / "metrics.json", "w") as f:
                    json.dump(self.history, f, indent=2)

        pbar.close()
        self.save_sample_grid(self.global_step)
        self.save_checkpoint("checkpoint_final.pt")
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)

        if self.tb_writer is not None:
            self.tb_writer.close()

        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.1f}s ({total_time / 60.0:.2f} min)!")
