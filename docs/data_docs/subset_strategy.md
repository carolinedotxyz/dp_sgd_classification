## Training Subset Creation Strategy (CelebA → Eyeglasses)

### Objective
Create an educational, differentially private-friendly image classification subset for the `Eyeglasses` attribute with a ratio-driven policy:
- Balanced classes (eyeglasses vs no_eyeglasses) within each split (50/50 per split)
- Split ratios: 80/10/10 (train/val/test)
- Total size is configurable; caps are computed from total N using the ratios
- Stratified sampling by a key confounder (pose/scale proxy) using inter-ocular distance

### Data sources
- Attributes: `data/celeba/archive/list_attr_celeba.csv`
- Partitions: `data/celeba/archive/list_eval_partition.csv`
- Landmarks (for inter-ocular distance): `data/celeba/archive/list_landmarks_align_celeba.csv`
- Images: `data/celeba/archive/img_align_celeba/` (autodetected nested folder variant handled)

### Process
1) Respect CelebA’s official split boundaries
- We never move items across train/val/test; balance is done within each split.

2) Merge and label
- Merge `list_attr_celeba.csv` with `list_eval_partition.csv` by `image_id`.
- Generate binary label for `Eyeglasses` (1 = eyeglasses, 0 = no_eyeglasses).

3) Landmarks and inter-ocular distance
- Load landmarks and compute the Euclidean distance between eye centers (`interocular`).
- When used for stratification, `interocular` is binned into quantiles (number of bins configurable).

4) Stratified, balanced sampling per split
- For each split, sample an equal number from each class up to caps derived from N:
  - Given total N and two classes, per-class caps are:
    - `max_per_class_train = floor(0.8 * N / 2)`
    - `max_per_class_val   = floor(0.1 * N / 2)`
    - `max_per_class_test  = floor(0.1 * N / 2)`
- Sampling is stratified by the requested confounders so class distributions match across strata. In this run we stratify by `interocular` (pose/scale), using 4 quantile bins.
- Missing files are skipped and backfilled from the same stratum/class to meet the cap (fail-fast available via `--strict-missing`).

5) Materialization and index CSV
- Output directory structure: `<output_dir>/<split>/<class_name>/<image_id>.jpg` with `class_name ∈ {eyeglasses, no_eyeglasses}`.
- Index CSV: `<output_dir>/subset_index_eyeglasses.csv` with at least:
  - `image_id, label, class_name, partition_name, source_path, dest_path`
  - `interocular` when landmarks are used
- Copy mode records `source_path` and `dest_path` relative to `<output_dir>`; symlink mode records absolute paths.

### Example: N = 20,000 (current repo default)
- Per-class caps resolve to: 8,000 (train), 1,000 (val), 1,000 (test)

### Command used
```bash
python scripts/celeba_build_subset.py \
  --archive-dir data/celeba/archive \
  --attribute Eyeglasses --use-landmarks --stratify-by interocular --iod-bins 4 \
  --output-dir data/celeba/subsets/eyeglasses_balanced_20k_strat \
  --link-mode copy --max-per-class-train 8000 --max-per-class-val 1000 --max-per-class-test 1000
```

### Reproducibility and variations
- Determinism: controlled by `--seed` (default 1337). Changing seed changes the specific samples but preserves the policy.
- Add more confounders to `--stratify-by` (e.g., `Male Young Smiling interocular`) to further harmonize class distributions.
- To include more metadata in the index CSV, use:
  - `--include-attrs <Attr1> <Attr2> ...` or `--include-all-attrs`
  - `--include-bbox` to append `x_1,y_1,width,height`
  - `--include-landmarks-raw` to append raw landmark coordinates

  ### Outputs
  [WIP]

### Notes for DP-SGD
- Balanced per-split sampling improves learning stability under DP noise.
- Prefer balancing and, if needed, loss reweighting over heavy oversampling to reduce distribution shift.
- Keep confounder distributions similar across classes to reduce spurious correlations (e.g., pose, gender, expression).

### Using other sizes
- To use a different total N, recompute caps with the same ratios and update the command flags accordingly.[WIP: will make this dynamic in upcoming update]
