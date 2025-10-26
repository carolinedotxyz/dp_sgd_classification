## CelebA — Eyeglasses classroom notebook

This short guide accompanies `notebooks/celeba_eyeglasses_workflow.ipynb`.
It is an educational notebook: you will actively create the subset and processed data by running its cells, so the steps are transparent and easy to learn.

### Prerequisites
- Download the CelebA dataset and place the extracted archive under `data/celeba/archive/` (see `README.md`).
- Ensure you have `matplotlib`, `pandas`, `numpy`, `Pillow`, and `tqdm` installed.

### What the notebook does
1. Reviews the archive and shows attribute balance
2. Builds a balanced Eyeglasses subset (train/val/test)
3. Preprocesses images (center-crop/resize) and computes TRAIN channel stats
4. Analyzes the processed dataset

### Running
Open and run all cells in `notebooks/celeba_eyeglasses_workflow.ipynb` on a fresh kernel.  
You must run this notebook before launching training scripts; it produces required artifacts (subset, processed images, and stats).

### Outputs
- Subset under `data/celeba/subsets/eyeglasses_balanced_20k/`
- Processed under `data/celeba/processed/eyeglasses_balanced_20k_64/`
- Stats files and plots saved alongside the data roots

These are intentionally produced in-notebook to keep the workflow observable and instructional.


