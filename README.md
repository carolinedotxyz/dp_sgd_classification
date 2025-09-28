![Status: WIP](https://img.shields.io/badge/status-WIP-orange)

_This project is a work in progress; details may change._

## Educational DP‑SGD Image Classifier (CelebA - Eyeglasses)


This is an educational repository. It walks through the full journey: selecting and balancing a subset of CelebA, preprocessing images, and training a small CNN with Differentially Private SGD (DP‑SGD) using PyTorch and Opacus.

In short: simple ideas, executed well. The goal is clarity over complexity.

### What you’ll find
- **Data analysis and subset building**: small, well‑balanced `Eyeglasses` vs `No Eyeglasses` dataset with stratification to reduce confounders.
- **Preprocessing**: deterministic center‑crop, resize, and dataset statistics for consistent normalization.
- **Training**: baseline (non‑DP) and DP‑SGD variants with clean, minimal training loops.
- **Reproducible outputs**: configs, metrics, and model artifacts saved under `runs/`.

### Quick start
1) Create the environment
```bash
conda env create -f env/environment.yml
conda activate dp-sgd-py312-env
```

2) Get the data (CelebA)
- Download CelebA from [Kaggle: CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and place it under `data/celeba/archive/` (see `docs/data/subset_strategy.md` for the required files and filenames).


3) Build a balanced subset (required)
- Choose your total N (e.g., 20,000). Per-class caps are computed via the ratio policy (80/10/10 splits; 50/50 class balance per split). The example below uses N = 20,000.
```bash
python scripts/celeba_build_subset.py \
  --archive-dir data/celeba/archive \
  --attribute Eyeglasses --use-landmarks --stratify-by interocular --iod-bins 4 \
  --output-dir data/celeba/subsets/eyeglasses_balanced_20k_strat \
  --link-mode copy --max-per-class-train 8000 --max-per-class-val 1000 --max-per-class-test 1000
```


### Unit tests
```
$ pytest -q
```




### Next up
1) Implement script + unit tests for - Preprocess dataset 
2) Implement script + unit tests for - Train the model (baseline and DP)
3) Implement notebook #1 for guided data preprocessing and stats
4) Implement notebook #2 for training baseline and DP‑SGD end‑to‑end

