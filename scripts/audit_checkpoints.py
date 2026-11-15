import torch
from pathlib import Path

paths = [
    'checkpoints/styleforge_final_model.pt',
    'experiments/phase3_convergence/checkpoints/checkpoint_final.pt',
    'experiments/phase3_improvement_logistic_r1/checkpoints/checkpoint_final.pt'
]

for p in paths:
    print('=' * 60)
    print(f"Path: {p}")
    if not Path(p).exists():
        print("  FILE NOT FOUND")
        continue
    d = torch.load(p, map_location='cpu', weights_only=False)
    print(f"  Keys in checkpoint: {list(d.keys())}")
    for k, v in d.items():
        if k in ['generator', 'critic', 'ema_generator']:
            print(f"  {k}: dict with {len(v)} parameter tensors")
        elif k in ['opt_g', 'opt_d']:
            param_groups = v.get('param_groups', [])
            lr = param_groups[0].get('lr', 'N/A') if param_groups else 'N/A'
            betas = param_groups[0].get('betas', 'N/A') if param_groups else 'N/A'
            print(f"  {k}: state dict (param_groups={len(param_groups)}, lr={lr}, betas={betas})")
        elif k == 'config':
            print(f"  config: {v}")
        else:
            print(f"  {k}: {v}")
