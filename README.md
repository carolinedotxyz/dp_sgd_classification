![Status: Active](https://img.shields.io/badge/status-active-green)

_This project is actively maintained. The core API is stable, but new features may be added._

# Educational SGD vs. DP-SGD Image Classifier
CelebA — Eyeglasses Attribute

## 📖 Overview

This repo guides an **end-to-end, educational workflow**:  
from raw [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) images → balanced subset creation → preprocessing → baseline CNN → differentially private training with DP-SGD (via [Opacus](https://opacus.ai/)).

Instead of diving straight into opaque training loops, we prioritize **clarity over complexity**. Each stage is observable, testable, and modifiable — designed as a teaching tool.

![Training Dynamics: Baseline vs DP-SGD](docs/assets/training_dynamics_comparison.gif)

**The primary entry point is** [`notebooks/celeba_eyeglasses_workflow.ipynb`](notebooks/celeba_eyeglasses_workflow.ipynb).  
This notebook provides an **end-to-end educational workflow** where you:
- Analyze the CelebA dataset and create balanced subsets
- Preprocess images with transparent, observable steps
- Train both baseline and DP-SGD models with matched hyperparameters
- Visualize and compare privacy-accuracy trade-offs

All steps are designed for learning: each stage is observable, testable, and modifiable. Training can also be done programmatically using the modules in `src/training/`, but the notebook is recommended for understanding the complete workflow.

---

### 🔗 Recommended Reading for New Users

If you're new to **Differential Privacy**, **DP-SGD**, or the **CelebA dataset**, the following resources provide helpful background before running this notebook:

#### **Differential Privacy Fundamentals**

* **"The Algorithmic Foundations of Differential Privacy" (Dwork & Roth)** — canonical introduction

  [https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

* **OpenMined: Differential Privacy Overview** — beginner-friendly conceptual guide

  [https://docs.openmined.org/differential-privacy](https://docs.openmined.org/differential-privacy)

#### **DP-SGD and Practical Implementations**

* **Original DP-SGD Paper: "Deep Learning with Differential Privacy"**

  [https://arxiv.org/abs/1607.00133](https://arxiv.org/abs/1607.00133)

* **Opacus (PyTorch) — DP-SGD Documentation**

  [https://opacus.ai/docs](https://opacus.ai/docs)

* **TensorFlow Privacy — DP-SGD Overview**

  [https://github.com/tensorflow/privacy](https://github.com/tensorflow/privacy)

#### **CelebA Dataset Background**

* **CelebA Dataset Paper ("Deep Learning Face Attributes…")**

  [https://arxiv.org/abs/1411.7766](https://arxiv.org/abs/1411.7766)

* **CelebA Dataset Homepage & Documentation**

  [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### **General DP Resources (Optional but Helpful)**

* **Apple Differential Privacy Technical Overview**

  [https://www.apple.com/privacy/docs/Differential_Privacy_Overview.pdf](https://www.apple.com/privacy/docs/Differential_Privacy_Overview.pdf)

* **Google's RAPPOR (Local DP technique)**

  [https://research.google/pubs/pub42852/](https://research.google/pubs/pub42852/)

---

## Quick Start

### 1. Install (editable)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### 2. Download the Data (CelebA)

* Download from Kaggle: CelebA Dataset ([https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)).
* Place the extracted archive under:

  ```
  data/celeba/archive/
  ```
* The notebook will validate the expected files/structure.

### 3. Configure (optional)

The notebook uses a centralized YAML configuration system. Edit `notebooks/config.yaml` to customize:
- Dataset paths and subset sizes
- Preprocessing parameters
- Training hyperparameters
- Privacy parameters (for DP-SGD)

See [`notebooks/README_config.md`](notebooks/README_config.md) for configuration details.

### 4. Create the subset and preprocess (in the notebook)

Open and run: `notebooks/celeba_eyeglasses_workflow.ipynb`  
The notebook will:
- build a balanced Eyeglasses subset (train/val/test),
- preprocess images (crop/resize/normalize), and
- save dataset statistics needed by loaders.

These artifacts are required before running training. We keep this step in-notebook to make the process transparent and learnable.

### 5. Train Models

After preprocessing, you can train models by:

- Continuing in `notebooks/celeba_eyeglasses_workflow.ipynb`
- The notebook includes training cells for both baseline and DP-SGD models
- Uses matched hyperparameters for fair privacy-accuracy comparison
- Includes visualization cells for analyzing results

---

## Project Structure

```
├── data/              # Dataset storage (CelebA archive, subsets, processed images)
├── notebooks/         # Main workflow notebook, configs, and utilities
│   ├── celeba_eyeglasses_workflow.ipynb  # Primary entry point
│   ├── config.yaml    # Centralized configuration
│   └── README_config.md  # Configuration guide
├── scripts/           # Standalone CLI scripts for data processing
├── src/               # Core Python package
│   ├── config/        # Configuration management (notebook, platform, training)
│   ├── core/          # Core ML components (models, data loaders, utils)
│   ├── datasets/      # Dataset-specific code (CelebA workflow, analysis)
│   ├── notebooks/     # Notebook utilities (display, setup, helpers)
│   ├── training/      # Training loops, sweeps, visualization
│   └── visualization/ # Plotting utilities
├── tests/             # Test suite
└── runs/              # Training run outputs (configs, checkpoints, metrics)
```

---

## Tests

Run tests with:

```bash
pytest -q
```

or for a specific file:

```bash
pytest tests/test_data.py
pytest tests/test_train_baseline.py
pytest tests/test_train_dp.py
```

---

## What you'll find

- **Data Processing Scripts** (`scripts/`):
  - `celeba_analyze.py`: Analyze CelebA attribute balance
  - `celeba_build_subset.py`: Create balanced subsets with stratification
  - `celeba_preprocess.py`: Preprocess images (crop/resize/normalize)
  - `celeba_centering.py`: Analyze face centering using landmarks

- **Core Functionality** (`src/`):
  - **Configuration** (`src/config/`): Centralized YAML-based config system with platform-specific workarounds (e.g., M1 Mac OpenMP fixes)
  - **Subset building** (`src/datasets/celeba/`): Balanced Eyeglasses vs No Eyeglasses, with options to reduce confounding
  - **Preprocessing** (`src/datasets/celeba/`): Deterministic center-crop/resize, normalization, and saved dataset stats
  - **Training** (`src/training/`): Clear baseline (non-DP) and DP-SGD loops using PyTorch + Opacus
  - **Hyperparameter sweeps** (`src/training/`): Automated grid searches for baseline and DP-SGD
  - **Visualization** (`src/visualization/`): Training curves, privacy-utility tradeoffs, and comparisons
  - **Notebook utilities** (`src/notebooks/`): Helper functions for timestamps, config printing, validation

- **Notebook Features** (`notebooks/`):
  - **Centralized configuration**: YAML-based config system (`config.yaml`) for easy experimentation
  - **Helper utilities**: Reusable functions (`generate_timestamp`, `print_config`, validation helpers)
  - **Cell dependencies**: Clear documentation of global state and dependencies in notebook header
  - **Code quality**: Well-documented, maintainable code following best practices
  - **Matched-pair methodology**: Identical hyperparameters for baseline and DP-SGD enable direct privacy cost quantification

- **Reproducibility**: Configs, metrics, and artifacts tracked under `runs/`
- **Documentation**: Comprehensive guides in `docs/` for data processing, training, and notebooks

---

## Documentation

- **Configuration**: [`notebooks/README_config.md`](notebooks/README_config.md) - Configuration system guide

## Roadmap

[WIP]

---

## Acknowledgments

* CelebA Dataset: [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* Opacus: [https://opacus.ai/](https://opacus.ai/)

