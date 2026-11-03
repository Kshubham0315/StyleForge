# StyleForge

A StyleGAN-inspired face generation project built with PyTorch, focused on GAN training stability, style-based generation, latent space exploration, and quantitative evaluation.

## Overview

StyleForge explores how modern GAN techniques influence image quality, diversity, and training stability. The project starts with a WGAN-GP baseline and progressively introduces style-based generation components and controlled experiments.

## Planned Features

- WGAN-GP baseline implementation
- Style mapping network and Adaptive Instance Normalization (AdaIN)
- Style mixing, latent interpolation, and truncation
- Self-attention and spectral normalization experiments
- R1 regularization and mode collapse analysis
- FID, KID, precision-recall, and diversity evaluation
- Architecture ablations and reproducible experiment tracking

## Tech Stack

- Python
- PyTorch
- NumPy
- Pillow
- Matplotlib
- TorchMetrics or equivalent evaluation tools

## Dataset

CelebA is the initial target dataset. Training will begin at 64×64 resolution, with 128×128 as a potential extension depending on hardware and training stability.

Dataset access and usage will follow the dataset's applicable terms.

## Project Structure

```text
StyleForge/
├── configs/
├── data/
├── models/
├── training/
├── evaluation/
├── scripts/
├── experiments/
├── tests/
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

Python and PyTorch environment setup instructions will be added alongside the initial training implementation.

## Experiments

Results will be documented with model configurations, training budgets, evaluation metrics, and sample visualizations. Comparisons will use consistent evaluation settings wherever possible.

## Project Status

In development. Architecture implementation, training experiments, and evaluation are planned.

## Goals

- Understand modern GAN architecture and optimization
- Investigate mode collapse and training instability
- Evaluate image quality and diversity quantitatively
- Build a reproducible, modular generative modeling pipeline

## License

MIT License, if enabled for this repository.
