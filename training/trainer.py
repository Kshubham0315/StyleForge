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
    """Maintains an exponential moving average of model parameters and buffers.

    Supports:
    - Standard exponential decay: theta_ema = beta * theta_ema + (1 - beta) * theta
    - Initialization modes:
        * 'model': Initialized to initial generator weights theta^(0)
        * 'zero': Initialized to zero tensors
    - Mathematical bias correction:
        * For 'model' init: theta_debiased = (theta_shadow - beta^t * theta^(0)) / (1 - beta^t)
        * For 'zero' init:  theta_debiased = theta_shadow / (1 - beta^t)
    - Warm-up scheduling:
        * beta(t) = min(beta, (1 + t) / (10 + t)) or step-based warmup
    - Update counter tracking and full state checkpoint serialization
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        use_warmup: bool = False,
        bias_correction: bool = False,
        init_mode: str = "model",
    ):
        self.decay = float(decay)
        self.use_warmup = use_warmup
        self.bias_correction = bias_correction
        self.init_mode = init_mode.lower()
        self.num_updates = 0

        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad = False

        if self.init_mode == "zero":
            for param in self.shadow.parameters():
                param.data.zero_()
            self.initial_params = None
        else:
            # Save clone of initial parameters for exact debiasing if requested
            self.initial_params = [p.clone().detach().cpu() for p in model.parameters()]

    def get_effective_decay(self) -> float:
        """Computes effective decay factor, taking into account warm-up scheduling."""
        if not self.use_warmup or self.num_updates == 0:
            return self.decay
        # Canonical StyleGAN / PyTorch AveragedModel warmup schedule
        warmup_decay = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Updates shadow weights and buffers with effective decay."""
        self.num_updates += 1
        beta = self.get_effective_decay()

        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(beta).add_(m_param.data, alpha=1.0 - beta)

        for s_buffer, m_buffer in zip(self.shadow.buffers(), model.buffers()):
            s_buffer.copy_(m_buffer)

    @torch.no_grad()
    def get_debiased_shadow(self, target_model: Optional[nn.Module] = None) -> nn.Module:
        """Returns a generator module with bias-corrected weights applied."""
        out_model = copy.deepcopy(target_model if target_model is not None else self.shadow).eval()
        if self.num_updates == 0 or not self.bias_correction:
            out_model.load_state_dict(self.shadow.state_dict())
            return out_model

        beta_t = self.decay ** self.num_updates
        correction_factor = max(1e-7, 1.0 - beta_t)

        out_params = list(out_model.parameters())
        shadow_params = list(self.shadow.parameters())

        if self.init_mode == "zero":
            for o_param, s_param in zip(out_params, shadow_params):
                o_param.data.copy_(s_param.data / correction_factor)
        else:
            for i, (o_param, s_param) in enumerate(zip(out_params, shadow_params)):
                init_p = self.initial_params[i].to(s_param.device)
                # theta_debiased = (theta_shadow - beta^t * theta^(0)) / (1 - beta^t)
                debiased = (s_param.data - beta_t * init_p) / correction_factor
                o_param.data.copy_(debiased)

        for o_buffer, s_buffer in zip(out_model.buffers(), self.shadow.buffers()):
            o_buffer.copy_(s_buffer)

        return out_model

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata for checkpoint serialization."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "use_warmup": self.use_warmup,
            "bias_correction": self.bias_correction,
            "init_mode": self.init_mode,
        }

    def load_metadata(self, meta: Dict[str, Any]):
        """Restores metadata from checkpoint."""
        self.decay = meta.get("decay", self.decay)
        self.num_updates = meta.get("num_updates", self.num_updates)
        self.use_warmup = meta.get("use_warmup", self.use_warmup)
        self.bias_correction = meta.get("bias_correction", self.bias_correction)
        self.init_mode = meta.get("init_mode", self.init_mode)


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
        t_cfg = config.get("training", {})
        ema_decay = float(t_cfg.get("ema_decay", 0.999))
        ema_warmup = bool(t_cfg.get("ema_warmup", False))
        ema_bias_corr = bool(t_cfg.get("ema_bias_correction", False))
        ema_init_mode = str(t_cfg.get("ema_init_mode", "model"))
        self.ema_generator = ExponentialMovingAverage(
            self.generator,
            decay=ema_decay,
            use_warmup=ema_warmup,
            bias_correction=ema_bias_corr,
            init_mode=ema_init_mode,
        )

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
            "ema_metadata": self.ema_generator.get_metadata(),
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
        if "ema_generator" in state:
            self.ema_generator.shadow.load_state_dict(state["ema_generator"])
        if "ema_metadata" in state:
            self.ema_generator.load_metadata(state["ema_metadata"])
        else:
            self.ema_generator.num_updates = state.get("global_step", 0)
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.global_step = state["global_step"]
        metrics_file = self.exp_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    self.history = json.load(f)
            except Exception:
                pass
        print(f"Resumed from checkpoint: {ckpt_path} at step {self.global_step} (history entries: {len(self.history)})")

    @torch.no_grad()
    def save_sample_grid(self, step: int):
        """Generate and save image grid using debiased EMA generator."""
        eval_gen = self.ema_generator.get_debiased_shadow()
        eval_gen.eval()
        fake = eval_gen(self.fixed_z)
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

        self.scaler.unscale_(self.opt_d)
        grad_norm_d = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm if self.max_grad_norm is not None else float("inf"),
        )
        d_stats["critic_grad_norm"] = float(grad_norm_d.item() if hasattr(grad_norm_d, "item") else grad_norm_d)

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

        self.scaler.unscale_(self.opt_g)
        grad_norm_g = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            self.max_grad_norm if self.max_grad_norm is not None else float("inf"),
        )
        g_stats["gen_grad_norm"] = float(grad_norm_g.item() if hasattr(grad_norm_g, "item") else grad_norm_g)

        # Track intermediate W-space dispersion if style generator
        if hasattr(self.generator, "mapping"):
            with torch.no_grad():
                w_sample = self.generator.mapping(z[:8])
                g_stats["w_space_std"] = float(w_sample.std().item())

        self.scaler.step(self.opt_g)
        self.scaler.update()

        # Update EMA weights only if generator parameters are clean (no NaN/Inf)
        has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in self.generator.parameters())
        if not has_nan:
            self.ema_generator.update(self.generator)
        else:
            print(f"[Warning] Generator parameters contain NaN/Inf at step {self.global_step}. Skipping EMA update.")

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
            steps_done = len(self.history) + 1
            it_per_sec = steps_done / (elapsed + 1e-6)
            max_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if self.device.type == "cuda" else 0.0

            # Estimate epoch based on dataloader length
            epoch = (self.global_step * batch_size) / max(1, len(self.dataloader.dataset))

            step_metrics = {
                "step": self.global_step,
                "epoch": round(epoch, 2),
                "elapsed_sec": round(elapsed, 2),
                "steps_per_sec": round(it_per_sec, 2),
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
        self.save_checkpoint("checkpoint_latest.pt")
        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)

        self.export_training_plots()

        if self.tb_writer is not None:
            self.tb_writer.close()

        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.1f}s ({total_time / 60.0:.2f} min)!")

    def export_training_plots(self):
        """Exports 4-panel training trajectory figures as PNG."""
        if not self.history:
            return
        try:
            import matplotlib.pyplot as plt
            steps = [m["step"] for m in self.history]
            g_loss = [m.get("g_loss", 0.0) for m in self.history]
            w_dist = [m.get("wasserstein_distance", 0.0) for m in self.history]
            penalty = [m.get("penalty", 0.0) for m in self.history]
            vram = [m.get("vram_mb", 0.0) for m in self.history]
            g_norm = [m.get("gen_grad_norm", 0.0) for m in self.history]
            d_norm = [m.get("critic_grad_norm", 0.0) for m in self.history]

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. Losses & Wasserstein Distance
            axes[0, 0].plot(steps, g_loss, label="Generator Loss", color="#1f77b4", alpha=0.8)
            axes[0, 0].plot(steps, w_dist, label="Wasserstein Distance", color="#ff7f0e", alpha=0.8)
            axes[0, 0].set_title("Generator Loss & Wasserstein Estimate")
            axes[0, 0].set_xlabel("Generator Step")
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            # 2. Gradient Penalty / Regularizer
            axes[0, 1].plot(steps, penalty, label="Penalty (GP / R1)", color="#2ca02c", alpha=0.8)
            axes[0, 1].set_title("Regularization Penalty Trajectory")
            axes[0, 1].set_xlabel("Generator Step")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

            # 3. Gradient Norms
            axes[1, 0].plot(steps, g_norm, label="Gen Grad Norm", color="#9467bd", alpha=0.7)
            axes[1, 0].plot(steps, d_norm, label="Critic Grad Norm", color="#8c564b", alpha=0.7)
            axes[1, 0].set_title("Parameter Gradient Norms")
            axes[1, 0].set_xlabel("Generator Step")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

            # 4. GPU Memory Utilization
            axes[1, 1].plot(steps, vram, label="CUDA VRAM (MB)", color="#d62728", alpha=0.8)
            axes[1, 1].set_title("GPU VRAM Utilization")
            axes[1, 1].set_xlabel("Generator Step")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

            plt.tight_layout()
            out_plot = self.exp_dir / "training_curves.png"
            plt.savefig(out_plot, dpi=150)
            plt.close(fig)
            print(f"Training trajectory plots saved to: {out_plot}")
        except Exception as e:
            print(f"[Warning] Failed to generate training plots: {e}")
