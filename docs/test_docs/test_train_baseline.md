### Unit test documentation — tests/test_train_baseline.py

This file documents what each test in `tests/test_train_baseline.py` validates, its assumptions, and gaps/edges not yet covered.

---

#### Helpers and fakes
- **_make_tiny_loaders(...)**: Produces train/val/test loaders of synthetic tensors with random labels. Uses different seeds per split via a `torch.Generator`. Shuffled loaders, default `batch_size=4`.
- **_clone_state_dict(...)**: Deep clones a state dict to CPU for stable comparisons.
- **_state_dicts_allclose(...)**: Compares two state dicts with `atol=5e-2, rtol=0.0` and exact key sets.
- **FakeEvaluate**: Callable that mimics `evaluate`, returning deterministic `(loss, acc)` for val/test loaders. Also snapshots model weights when called with the val loader to capture per-epoch state.

---

#### test_get_optimizer_variants
- **Validates**: `_get_optimizer(name, params, lr, weight_decay)` returns correct optimizer instance.
  - Supports: `adam`, `adamw`, `sgd`; unknown name falls back to Adam.
- **Assumptions**: Only type checks; does not assert hyperparameters are applied or error handling for invalid args.
- **Not covered / edges**:
  - Momentum, betas, eps, and other hyperparameter wiring for each optimizer.
  - Handling of parameter groups, different learning rates/weight decays per group.

#### test_train_baseline_saves_best_and_outputs
- **Validates**: With `save_best=True`, the best-performing checkpoint is saved and final model equals the on-disk `best.pth`.
  - History length equals epochs; summary keys include `best_val_acc` and `test_acc` reflecting `FakeEvaluate`.
  - Output artifacts created in run dir: `config.json`, `history.csv`, `summary.json`, `best.pth`.
  - Confirms `config.json` content for `run_name` and `optimizer`.
- **Assumptions**: CPU training with `SimpleCNN`, synthetic loaders, monkeypatched `evaluate`.
- **Not covered / edges**:
  - Device handling (CUDA/MPS), AMP, gradient clipping, and scheduler integration.
  - Behavior on ties for best validation accuracy (first vs last best).
  - Robustness to I/O failures, partial writes, or interrupted runs.
  - Determinism and seeding across epochs and shuffling.

#### test_train_baseline_no_save_best_keeps_last_state_and_no_best_file
- **Validates**: With `save_best=False`, the final model weights remain from the last epoch; `best.pth` is not written.
  - Still tracks `best_val_acc` in summary; history length equals epochs.
  - Confirms presence of `config.json`, `history.csv`, `summary.json` and absence of `best.pth`.
  - Verifies final state equals the last validation snapshot captured by `FakeEvaluate`.
- **Assumptions**: Same synthetic setup and fake evaluation as prior test.
- **Not covered / edges**:
  - Correctness of intermediate checkpointing (if any) and resume-from-checkpoint behavior.
  - Multi-run directory handling (collision, overwrite vs unique), permission errors.
  - Validation frequency other than once per epoch; early stopping.

---

### Coverage summary
- **Covered**:
  - Optimizer factory returns expected optimizer classes with a sensible default fallback.
  - Baseline training produces a consistent history/summary and expected artifacts.
  - Best-checkpoint saving toggled by `save_best` behaves as specified; model state matches best or last epoch accordingly.

- **Gaps to implement**:
  - Hyperparameter wiring for optimizers (momentum/betas/eps), parameter groups, and invalid input handling.
  - Device, mixed precision, gradient clipping, and scheduler step semantics.
  - Tie-breaking strategy for best selection; deterministic behavior with fixed seeds and shuffling.
  - Failure modes: I/O errors, partial writes, resume behavior, and run directory collisions.
  - Evaluation cadence configurability and early stopping/patience logic.


