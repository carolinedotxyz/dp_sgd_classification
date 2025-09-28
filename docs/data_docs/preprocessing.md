## Preprocessing Strategy (CelebA → Eyeglasses, DP‑SGD Ready)

### Objective
Prepare a small, consistent image dataset for an educational DP‑SGD classifier, minimizing memory and variance while preserving facial signal for the `Eyeglasses` vs `No Eyeglasses` task.

### Data sources
- Balanced subset with ratio policy (50/50 class balance per split; 80/10/10 splits)
- Current repo default example (N = 20,000):
  - `data/celeba/subsets/eyeglasses_balanced_20k_strat`
- Derived from official CelebA splits with stratification by inter‑ocular distance to balance pose/scale across classes (see `subset_strategy.md`).

### Process
- Resize: 64×64
  - Smaller images → faster training, lower per‑sample memory, easier DP accounting.
- Crop: center crop before resize (square)
  - Centering analysis using landmarks shows faces are near the image center (median radius ≈ 0.04 normalized), so a center crop keeps the face and trims background.
  - If a future subset shows larger offsets, prefer bbox/landmark‑based crops.
- Normalize: compute train split mean/std on [0,1]
  - Images saved as uint8 RGB; stats computed on [0,1] arrays for training transforms.
  - Store at `processed/.../stats/stats.json`. Use the same stats for val/test/inference.

### Commands used 
- Preprocess with center crop → 64×64 and compute stats (20k example):
```bash
python scripts/celeba_preprocess.py \
  --subset-root data/celeba/subsets/eyeglasses_balanced_20k_strat \
  --out-root data/celeba/processed/eyeglasses_balanced_20k_strat_64 \
  --size 64 --center-crop --normalize-01 --compute-stats
```

### Outputs
- Directory: `data/celeba/processed/eyeglasses_balanced_20k_strat_64/<split>/{eyeglasses,no_eyeglasses}/...`
- Index CSV: `.../processed_index.csv`
- Stats JSON: `.../stats/stats.json`
  - Keys: `size`, `normalize_01`, `train_mean`, `train_std` (RGB channel order)

### Training transform recommendation
- If using PyTorch:
  - Convert to tensor scaling to [0,1], then apply `Normalize(mean, std)` from `stats.json`.
  - Keep the same preprocessing for train/val/test/inference to avoid distribution shift.

### DP‑SGD considerations
- Smaller images reduce the memory and runtime overhead of per‑sample gradients.
- Consistent normalization stabilizes gradient clipping behavior.
- Balanced per‑split sampling and stratification reduce spurious correlations, helping under DP noise.

### Validation of centering (what we checked)
- Script: `scripts/celeba_centering.py`
- Artifacts: `.../centering_analysis/`
  - `centering_metrics.csv`, `center_distance_hist.png`, `center_offset_hex.png`, `outliers_grid.png`
- Interpretation: most faces are close to center; center crop is appropriate for this subset.

### Variations (when to change)
- Off‑center faces: switch to bbox/landmark crop.
- Different model capacity: adjust `--size` (e.g., 96, 128) and recompute stats.
- Grayscale experiments: convert to L mode before stats if desired and store per‑channel stats accordingly.

### Other subset sizes
- If you use a different total N, preprocessing is identical; only the input/output directories change to match your chosen subset name.


