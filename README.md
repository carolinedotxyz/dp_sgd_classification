![Status: WIP](https://img.shields.io/badge/status-WIP-orange)

_This project is a work in progress; details may change._

## DP‑SGD Image Classifier (CelebA - Eyeglasses)

This is an educational repository. It walks through the full journey: selecting and balancing a subset of CelebA, preprocessing images, and training a small CNN with Differentially Private SGD (DP‑SGD) using PyTorch and Opacus.

In short: simple ideas, executed well. The goal is clarity over complexity.

### What you’ll find
- **Data analysis and subset building**: small, well‑balanced `Eyeglasses` vs `No Eyeglasses` dataset with stratification to reduce confounders.
- **Preprocessing**: deterministic center‑crop, resize, and dataset statistics for consistent normalization.
- **Training**: baseline (non‑DP) and DP‑SGD variants with clean, minimal training loops.
- **Reproducible outputs**: configs, metrics, and model artifacts saved under `runs/`.