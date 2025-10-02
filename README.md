![Status: WIP](https://img.shields.io/badge/status-WIP-orange)

_This project is a work in progress; details may change._

## Educational DP‑SGD Image Classifier (CelebA — Eyeglasses)

When you step into an unfamiliar codebase, it’s easy to see what’s missing and forget what’s working. This repository takes the opposite stance. We keep what works, add seams where we need change, and move in safe, observable steps. The result: a small, clear path from raw CelebA images to a baseline CNN and a DP‑SGD variant using PyTorch and Opacus.

In short: simple ideas, executed well. Clarity over complexity.

### What you’ll find
- **Data subset building**: a balanced `Eyeglasses` vs `No Eyeglasses` split with options to reduce confounding.
- **Preprocessing**: deterministic center‑crop/resize and computed dataset statistics for stable normalization.
- **Training**: baseline (non‑DP) and DP‑SGD loops designed to be readable and easy to change.
- **Reproducibility**: configs, metrics, and artifacts written under `runs/`.

### Principles (how we keep change safe)
- **Small steps**: make one thing better at a time.
- **Determinism where it matters**: fixed transforms and stored stats.
- **Seams for experimentation**: clear points to swap samplers, optimizers, or transforms.
- **Tight feedback**: fast dry‑runs and tests before long training jobs.

### Install (editable) and quick start
1) Create the environment (or use your own)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

2) Get the data (CelebA)
- Download from [Kaggle: CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and place it under `data/celeba/archive/`.
- See `docs/data/subset_strategy.md` for the exact files and filenames expected.

3) Build a balanced subset (CLI)
- Pick a target size (e.g., 20,000). Splits are 80/10/10 and balanced 50/50 by class. Missing images are backfilled so you can hit requested caps when alternatives exist.
```bash
celeba-build-subset \
  --archive-dir data/celeba/archive \
  --attribute Eyeglasses \
  --output-dir data/celeba/subsets/eyeglasses_balanced_20k \
  --images-root data/celeba/archive/img_align_celeba/img_align_celeba \
  --link-mode copy --overwrite \
  --max-per-class-train 8000 --max-per-class-val 1000 --max-per-class-test 1000
```

- Dry‑run to preview achievable counts without writing files:
```bash
celeba-build-subset \
  --archive-dir data/celeba/archive \
  --attribute Eyeglasses \
  --output-dir /tmp/dry_run --dry-run \
  --images-root data/celeba/archive/img_align_celeba/img_align_celeba \
  --max-per-class-train 8000 --max-per-class-val 1000 --max-per-class-test 1000
```

- Optional distribution matching via inter‑ocular distance (may reduce caps): add
  `--use-landmarks --stratify-by interocular --iod-bins 2`

- Strict mode (no top‑up/backfill for missing files): add `--no-fill-missing`

4) Preprocess the images (recommended before training)
- Classroom notebook: `notebooks/celeba_eyeglasses_workflow.ipynb` (see `docs/notebooks/celeba_eyeglasses_workflow.md`). Or run the script:
```bash
celeba-preprocess \
  --subset-root data/celeba/subsets/eyeglasses_balanced_20k \
  --out-root data/celeba/processed/eyeglasses_balanced_20k_64 \
  --size 64 --center-crop --normalize-01 --compute-stats
```
This writes `processed_index.csv` and `stats/stats.json` with `train_mean` and `train_std` used by the loaders.

### Using the library in notebooks
Prefer importing from `src` (single source of truth). Example:
```python
from src import WorkflowConfig, review_archive, build_balance_display_table
from src import analyze_processed, preprocess_images

cfg = WorkflowConfig(
    archive_dir=Path("data/celeba/archive"),
    images_root=Path("data/celeba/archive/img_align_celeba/img_align_celeba"),
    output_dir=Path("data/celeba/subsets/eyeglasses_balanced_20k"),
    subset_root=Path("data/celeba/subsets/eyeglasses_balanced_20k"),
    out_root=Path("data/celeba/processed/eyeglasses_balanced_20k_64"),
)
review_archive(cfg)
preprocess_images(cfg)
analyze_processed(cfg)
```

### Module map (public)
- `src/celeba_io.py`: archive validation, merge/load CSVs, write outputs
- `src/celeba_index.py`: processed/subset index read/augment/discover
- `src/celeba_analysis.py`: balance summaries, comparisons, display tables
- `src/celeba_diagnostics.py`: sampling, image size/channel stats
- `src/celeba_plots.py`: domain plotting wrappers
- `src/celeba_workflow.py`: orchestration for build/preprocess/analyze
- `src/celeba_builder.py`: programmatic subset builder
- `src/celeba_preprocess_core.py`: programmatic preprocessing

### Tests
```bash
pytest -q
```

### Roadmap
1) Implement gap tests for `docs/test_docs/test_data.md`, `docs/test_docs/test_utils.md`, `docs/test_docs/test_train_baseline.md`, `docs/test_docs/test_train_dp.md`
2) Implement notebook #1 for guided data preprocessing and stats
3) Implement notebook #2 for end‑to‑end training (baseline and DP‑SGD)

