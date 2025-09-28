### Unit test documentation — tests/test_data.py

This file documents what each test in `tests/test_data.py` validates, the inputs/assumptions, and notable gaps/edges not yet covered.

---

#### Fixture: tmp_imagefolder
- **Purpose**: Creates a minimal `ImageFolder`-style directory with RGB squares for classes `class_a` and `class_b` across `train/val/test` splits.
  - Train: 5 images per class → 10 total
  - Val: 2 images per class → 4 total
  - Test: 2 images per class → 4 total
- **Assumptions**: JPEG files with shape 10×10×3; simple uniform color per split/class.
- **Not covered / edges**:
  - Non-RGB inputs, corrupt files, varied sizes/aspect ratios, nested classes, empty classes.
  - Class names with special characters, single-class datasets.

#### test_make_transforms_no_aug
- **Validates**: `make_transforms(mean, std, aug_flip=False, aug_jitter=False)` returns a callable transform.
- **Assumptions**: Only existence/callability; does not assert content/order of composed transforms or normalization behavior.
- **Not covered / edges**:
  - Input type handling (`PIL.Image` vs `Tensor`), channel order, dtype conversions.
  - Normalization using provided `mean/std` (length=3, value ranges).
  - Determinism and error handling for invalid `mean/std` shapes.

#### test_make_transforms_with_aug
- **Validates**: With `aug_flip=True, aug_jitter=True`, the returned transform is callable.
- **Assumptions**: Presence of augmentation steps but not their parameters or stochasticity.
- **Not covered / edges**:
  - Flip probability, jitter ranges, and composition order with normalization.
  - Deterministic behavior under fixed seed; CPU vs GPU transforms.
  - Behavior on grayscale or single-channel images.

#### test_load_stats
- **Validates**: `load_stats(path)` parses a JSON file and returns `train_mean`, `train_std` as sequences identical to the JSON payload.
- **Assumptions**: File exists, JSON has expected keys with length-3 lists.
- **Not covered / edges**:
  - Missing file, invalid JSON, missing/extra keys, wrong lengths/dtypes.
  - `Path` vs `str` inputs; numeric types (`int` vs `float`).
  - Value validation (ranges, NaNs/inf) and defaulting behavior.

#### test_build_dataloaders_shapes
- **Validates**: `build_dataloaders(...)` returns loaders with dataset sizes matching the fixture and yields image/target tensors.
  - Checks dataset lengths: train=10, val=4, test=4.
  - Fetches one train batch and asserts: tensor type, `NCHW` with `C=3`.
- **Assumptions**: `batch_size=4`, `num_workers=0`, `aug_*` disabled, mean/std of ones/zeros.
- **Not covered / edges**:
  - Non-divisible batch sizes (last partial batch), `drop_last`, `shuffle`, `pin_memory`.
  - Device placement (CPU/GPU), memory format, dtype, and normalization range correctness.
  - Class-to-index mapping stability and alphabetical ordering under `ImageFolder`.
  - Error handling for empty splits or missing directories.

#### test_build_dataloaders_aug_flags
- **Validates**: With `aug_flip=True` and `aug_jitter=False`, construction works and a batch can be iterated (`images.shape[0] <= 4`).
- **Assumptions**: Only flip augmentation path exercised; no checks on targets or image value distributions.
- **Not covered / edges**:
  - `aug_jitter=True` path, combined augmentations, and deterministic behavior under seeding.
  - Consistency of labels with augmented images; potential geometric transforms changing shape.
  - Multi-worker loading (`num_workers>0`) and worker seeding.

---

### Coverage summary
- **Covered**:
  - Creation of augmentation/no-augmentation transforms (existence/callability).
  - Loading dataset normalization stats from a JSON file (happy path).
  - Building dataloaders from an `ImageFolder`-style structure; basic shapes and dataset sizes.
  - Minimal augmentation flag path (flip enabled) constructs a working loader.

- **Gaps to implement**:
  - Transform correctness: verify normalization (per-channel mean/std), output ranges/dtypes, composition order.
  - Augmentation behavior: test jitter on, combined flips+jitter, and determinism with a fixed seed.
  - Robustness: invalid/missing stats, malformed datasets (empty classes, corrupt files, grayscale), diverse image sizes.
  - DataLoader options: `num_workers>0`, `shuffle`, `drop_last`, `pin_memory`, non-divisible batch sizes, empty datasets.
  - Device and performance: move to CUDA/MPS where available; ensure tensor device/dtype are correct.
  - Label mapping: assert stable class-to-index mapping and correct target ranges across splits.


