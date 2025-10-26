![Status: WIP](https://img.shields.io/badge/status-WIP-orange)

_This project is under active development; details and APIs may change._

# Educational SGD vs. DP-SGD Image Classifier
CelebA — Eyeglasses Attribute

## 📖 Overview

This repo guides an **end-to-end, educational workflow**:  
from raw [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) images → balanced subset creation → preprocessing → baseline CNN → differentially private training with DP-SGD (via [Opacus](https://opacus.ai/)).

Instead of diving straight into opaque training loops, we prioritize **clarity over complexity**. Each stage is observable, testable, and modifiable — designed as a teaching tool, not just a benchmark script.

**Educational analysis, subset creation, and preprocessing happen inside** [`notebooks/celeba_eyeglasses_workflow.ipynb`](notebooks/celeba_eyeglasses_workflow.ipynb).  
This is intentional (for teaching): you will create the subset and processed data in the notebook.  
Training scripts under `src/` (e.g., `src/train_baseline.py`, `src/train_dp.py`) build directly on these artifacts.

---

## Quick Start

### 1. Install (editable)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
````

### 2. Download the Data (CelebA)

* Download from Kaggle: CelebA Dataset ([https://www.kaggle.com/datasets/jessicali9530/celeba-dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)).
* Place the extracted archive under:

  ```
  data/celeba/archive/
  ```
* The notebook will validate the expected files/structure.

### 3. Create the subset and preprocess (in the notebook)

Open and run: `notebooks/celeba_eyeglasses_workflow.ipynb`  
The notebook will:
- build a balanced Eyeglasses subset (train/val/test),
- preprocess images (crop/resize/normalize), and
- save dataset statistics needed by loaders.

These artifacts are required before running training scripts. We keep this step in-notebook to make the process transparent and learnable.


---

## Tests

Run tests with:

```bash
pytest -q
```

or for a specific file:

```bash
pytest tests/test_celeba_preprocess.py
```

---

## What you’ll find

- **Subset building**: Balanced Eyeglasses vs No Eyeglasses, with options to reduce confounding.
- **Preprocessing**: Deterministic center-crop/resize, normalization, and saved dataset stats.
- **Training**: Clear baseline (non-DP) and DP-SGD loops using PyTorch + Opacus.
- **Reproducibility**: Configs, metrics, and artifacts tracked under `runs/`.

---

## Principles (design philosophy)

- **Small steps**: improve one thing at a time.
- **Determinism where it matters**: fixed transforms, stored stats.
- **Seams for experimentation**: easy swap-in points (optimizers, samplers, transforms).
- **Tight feedback**: fast dry-runs + tests before long jobs.

---

## Roadmap

- [ ] Add gap tests for `docs/test_docs/*.md` (data, utils, baseline, DP training).
- [ ] Implement Notebook #1 (guided preprocessing + stats).
- [ ] Extend to other CelebA attributes and multi-attribute tasks.

---

## Acknowledgments

* CelebA Dataset: [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* Opacus: [https://opacus.ai/](https://opacus.ai/)

