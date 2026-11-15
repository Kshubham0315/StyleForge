import os
import sys
import time
import torch
from pathlib import Path

# Insert project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import CelebADataset, get_dataloader
from torch.utils.data import DataLoader
from models.style_generator import StyleGenerator
from models.critic import StyleCritic
from training.losses import GANLoss
from training.trainer import GANTrainer

def run_audit():
    print("=" * 70)
    print("STYLEFORGE PHASE 4 AUDIT: ENVIRONMENT, CHECKPOINT & VRAM")
    print("=" * 70)

    # 1. Device and Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_mem_gb:.2f} GB")
    else:
        print("Running on CPU!")
        gpu_name = "CPU"
        total_mem_gb = 0

    # 2. Checkpoint Inspection
    ckpt_path = Path("experiments/phase3_convergence/checkpoints/checkpoint_final.pt")
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print("\n[CHECKPOINT INSPECTION]")
    print(f"Source Checkpoint: {ckpt_path}")
    print(f"File Size: {ckpt_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Keys present: {list(state.keys())}")
    print(f"Global Step: {state.get('global_step')}")
    print(f"Has Generator: {'generator' in state} (tensors: {len(state['generator'])})")
    print(f"Has Critic: {'critic' in state} (tensors: {len(state['critic'])})")
    print(f"Has EMA Generator: {'ema_generator' in state} (tensors: {len(state['ema_generator'])})")
    print(f"Has Optimizer G: {'opt_g' in state}")
    print(f"Has Optimizer D: {'opt_d' in state}")
    print(f"Has Scaler State: {'scaler' in state}")
    print(f"Has RNG State: {'rng_state' in state}")
    print(f"Has EMA Metadata: {'ema_metadata' in state}")

    # Inspect optimizer state
    opt_g_state = state["opt_g"]
    opt_d_state = state["opt_d"]
    lr_g = opt_g_state["param_groups"][0]["lr"]
    lr_d = opt_d_state["param_groups"][0]["lr"]
    betas_g = opt_g_state["param_groups"][0]["betas"]
    betas_d = opt_d_state["param_groups"][0]["betas"]
    print(f"Recorded LR - G: {lr_g}, D: {lr_d}")
    print(f"Recorded Betas - G: {betas_g}, D: {betas_d}")

    # 3. Model Parameters and Shapes
    gen = StyleGenerator(latent_dim=512, style_dim=512, img_size=64, base_channels=512, max_channels=512)
    gen.load_state_dict(state["generator"])
    critic = StyleCritic(img_size=64, base_channels=64, max_channels=512, use_minibatch_std=True)
    critic.load_state_dict(state["critic"])
    
    gen_params = sum(p.numel() for p in gen.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"\n[MODEL CAPACITY]")
    print(f"Generator Parameters: {gen_params:,} ({gen_params * 4 / (1024*1024):.2f} MB in FP32)")
    print(f"Critic Parameters: {critic_params:,} ({critic_params * 4 / (1024*1024):.2f} MB in FP32)")

    # 4. Dataset Inspection
    train_ds = CelebADataset("data/celeba/img_align_celeba", split="train")
    val_ds = CelebADataset("data/celeba/img_align_celeba", split="val")
    print(f"\n[DATASET METRICS]")
    print(f"Training Samples: {len(train_ds)}")
    print(f"Validation Samples: {len(val_ds)}")
    print(f"Sample Tensor Shape: {train_ds[0].shape}, Range: [{train_ds[0].min():.2f}, {train_ds[0].max():.2f}]")

    # 5. Measure Baseline VRAM and Step Throughput on CUDA (Short 5-step probe)
    if torch.cuda.is_available():
        print("\n[PROBING VRAM AND THROUGHPUT: 5-step warm baseline]")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        cfg = state["config"]
        # Set short test
        cfg["experiment"]["exp_dir"] = "experiments/audit_probe"
        cfg["training"]["amp"] = True
        
        loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        trainer = GANTrainer(gen, critic, loader, cfg, device=device)
        trainer.load_checkpoint(str(ckpt_path))
        
        # Measure 5 steps
        t0 = time.perf_counter()
        data_iter = iter(loader)
        for step in range(5):
            for _ in range(trainer.n_critic):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                batch = batch.to(device, non_blocking=True)
                trainer.train_step_critic(batch)
            trainer.train_step_generator(batch.size(0))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_vram_mb = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"5 Steps Elapsed: {elapsed:.2f} s ({elapsed/5:.3f} s/step, {5/elapsed:.2f} steps/s)")
        print(f"Peak Allocated VRAM: {peak_vram_mb:.1f} MB")
        print(f"Peak Reserved VRAM: {reserved_vram_mb:.1f} MB")
        print(f"Available VRAM Margin: {(total_mem_gb * 1024) - reserved_vram_mb:.1f} MB")

        # Cleanup probe dir
        import shutil
        probe_dir = Path("experiments/audit_probe")
        if probe_dir.exists():
            shutil.rmtree(probe_dir)

if __name__ == "__main__":
    run_audit()
