### Unit test documentation — tests/test_utils.py

This file documents what each test in `tests/test_utils.py` validates, the inputs/assumptions, and gaps/edges that are not yet covered.

---

#### test_get_device_reports_available_backend
- **Validates**: `get_device()` returns a `torch.device` whose `type` is one of `{"cpu", "cuda", "mps"}`.
- **Assumptions**: Does not require a specific accelerator to be present; only checks the reported type is recognized by PyTorch.
- **Not covered / edges**:
  - Behavior when CUDA/MPS is available but not usable (e.g., driver mismatch).
  - Selection preference/order when multiple backends exist (e.g., CUDA vs MPS vs CPU).
  - Correct handling of specific device indices (e.g., `cuda:0`).

#### test_set_seed_makes_results_reproducible
- **Validates**: `set_seed(123)` synchronizes RNGs across Python `random`, NumPy, and CPU Torch.
  - Compares two sequences of draws for equality: `random.randint`, `np.random.rand`, `torch.randn`.
- **Assumptions**: CPU RNGs only; single-process execution.
- **Not covered / edges**:
  - CUDA/MPS RNG seeding and determinism.
  - DataLoader worker seeding (`num_workers>0`).
  - Determinism settings (e.g., `torch.backends.cudnn.deterministic`).
  - Sequence continuity (advancing RNG after seeding vs re-seeding between calls).

#### test_accuracy_matches_argmax_mean
- **Validates**: `accuracy(logits, targets)` equals the fraction of correct `argmax` predictions.
  - Uses a small 3×2 logits tensor and matching integer class targets.
- **Assumptions**: Logits and targets on CPU with matching batch size.
- **Not covered / edges**:
  - Ties in logits (equal max values) and expected tie-breaking.
  - Non-integer or out-of-range targets; one-hot targets; different dtypes/devices.
  - Empty batch behavior and return type/range guarantees (0.0–1.0).
  - Multi-class with >2 classes and larger batch sizes.

#### test_evaluate_iterates_loader_and_averages
- **Validates**: `evaluate(model, loader, criterion, device)`
  - Iterates over a `DataLoader` and returns dataset-size–weighted average loss and accuracy.
  - Compares to a manual reference using `CrossEntropyLoss` and `argmax`-based accuracy.
  - Uses a tiny linear model on CPU with batch size 4 and dataset size 8 (exactly divisible).
- **Assumptions**: CPU execution; criterion reduction semantics as used in the test; no shuffling or AMP.
- **Not covered / edges**:
  - Non-divisible batch sizes (last partial batch) and empty loaders.
  - Device transfers (inputs/targets/model on CUDA/MPS), mixed precision, and `torch.no_grad()` handling.
  - Model train/eval mode management and gradient disabling during evaluation.
  - Return dtypes and numerical stability across devices.

---

### Coverage summary
- **Covered**:
  - Device type detection shape (CPU/CUDA/MPS) at a high level.
  - Cross-library seeding for reproducibility (Python/NumPy/Torch CPU).
  - Core classification accuracy definition (argmax match rate).
  - Evaluation loop aggregates loss and accuracy over batches on CPU.

- **Gaps to implement**:
  - Exercise GPU/MPS paths where available, including RNG seeding and tensor/device movement in `evaluate`.
  - Add tests for empty loaders and datasets with sizes not divisible by batch size.
  - Validate behavior with DataLoader `num_workers>0` and worker seeding.
  - Add edge cases for `accuracy`: ties, invalid targets, dtype/device mismatches, empty batch.
  - Confirm `evaluate` sets `model.eval()`, uses `torch.no_grad()`, and restores state if needed.
  - Check deterministic behavior under different PyTorch determinism flags and backends.


