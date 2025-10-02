## CelebA — Eyeglasses classroom notebook

This short guide accompanies `notebooks/celeba_eyeglasses_workflow.ipynb`.

### Prerequisites
- Place the CelebA archive under `data/celeba/archive/` as described in `README.md`.
- Ensure you have `matplotlib`, `pandas`, `numpy`, `Pillow`, and `tqdm` installed.

### What the notebook does
1. Reviews the archive and shows attribute balance
2. Builds a balanced Eyeglasses subset (train/val/test)
3. Preprocesses images (center-crop/resize) and computes TRAIN channel stats
4. Analyzes the processed dataset

### Running
Open and run all cells in `notebooks/celeba_eyeglasses_workflow.ipynb` on a fresh kernel.

### Outputs
- Subset under `data/celeba/subsets/eyeglasses_balanced_20k/`
- Processed under `data/celeba/processed/eyeglasses_balanced_20k_64/`
- Plots and CSVs saved alongside the data roots


