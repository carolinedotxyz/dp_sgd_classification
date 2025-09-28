### Unit test documentation — tests/test_train_dp.py

This file documents what each test in `tests/test_train_dp.py` validates, its assumptions, and gaps/edges not yet covered.

---

#### Helpers and fakes
- **_make_tiny_loaders(...)**: Builds synthetic train/val/test `DataLoader`s with random images/labels using split-specific `torch.Generator` seeds. `shuffle=True`, default `batch_size=4`.
- **_clone_state_dict(...)**: Clones model weights to CPU for stable comparisons.
- **_state_dicts_allclose(...)**: Compares two state dicts with `atol=5e-2`.
- **FakeAccountant**: Provides `get_epsilon(delta)` returning a fixed epsilon.
- **FakePrivacyEngine**: Exposes `.make_private(...)` that returns inputs unchanged and holds a `FakeAccountant`.
- **FakeEvaluate**: Mimics `evaluate`, returning deterministic `(loss, acc)` per loader; snapshots model weights on each validation call.
- **_import_train_dp_with_fake_opacus()**: Injects a dummy `opacus.PrivacyEngine` before importing `src.train_dp` so tests do not depend on the real library.

---

#### test_train_dp_sgd_restores_best_and_reports_epsilon
- **Validates**: `train_dp_sgd(...)`
  - Restores model weights to the best validation snapshot across epochs.
  - Returns metrics dict including `test_acc` and `epsilon` (via the accountant).
  - Works with the injected fake `PrivacyEngine` (no-ops the privatization call).
- **Assumptions**: CPU device, `SimpleCNN`, synthetic loaders, `DPConfig` with `epochs=3`.
- **Not covered / edges**:
  - Real `opacus` integration (gradient clipping, noise addition) and its side effects.
  - Behavior on ties for best validation accuracy; determinism with seeded loaders.
  - Failure modes in accountant/engine wiring and incorrect config values.

#### test_train_dp_sgd_v2_scheduler_step_and_label_smoothing_and_best_acc
- **Validates**: `train_dp_sgd_v2(...)`
  - Supports label smoothing and a step LR scheduler (`step_size=1, gamma=0.5`).
  - Restores best validation snapshot and reports `best_val_acc`, `test_acc`, and `epsilon`.
- **Assumptions**: CPU device, `SimpleCNN`, synthetic loaders, `DPConfigV2` with `epochs=2`.
- **Not covered / edges**:
  - Alternate schedulers (cosine, plateau), warmup, or per-step vs per-epoch stepping.
  - Label smoothing effects on loss numerics and metric consistency across devices.
  - Interactions with real DP training (noise, clipping) and accuracy behavior.

---

### Coverage summary
- **Covered**:
  - DP training entrypoints restore the best validation snapshot and report `epsilon`.
  - V2 path exercises label smoothing and a step scheduler with expected metrics.
  - Isolation from real `opacus` via injection to keep tests deterministic and fast.

- **Gaps to implement**:
  - End-to-end with real `opacus`: verify gradients are clipped, noise is applied, and epsilon matches expectations for given `noise_multiplier`, `batch_size`, and epochs.
  - Tie-breaking policy for best model selection and deterministic behavior under fixed seeds.
  - Error handling for invalid configs (negative `noise_multiplier`, zero `max_grad_norm`, bad scheduler params).
  - Device coverage (CUDA/MPS), AMP/mixed precision, and correctness of device transfers.
  - Scheduler variants and step timing (per-batch vs per-epoch); resume-from-checkpoint behavior.


