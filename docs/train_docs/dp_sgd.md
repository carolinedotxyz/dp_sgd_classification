## DP‑SGD Training

See `main/train_dp.py` for the implementation of DP‑SGD with Opacus, including a configurable v2 loop with schedulers and light augmentations.


### Determinism and Performance Considerations (CUDA and Apple MPS)

For reproducibility and privacy accounting stability, the training loops set deterministic flags where supported:

- Disable cuDNN autotuner when on CUDA: `torch.backends.cudnn.benchmark = False` and `torch.backends.cudnn.deterministic = True`.
- Globally enable deterministic algorithms where available: `torch.use_deterministic_algorithms(True)` (wrapped in try/except to avoid errors on unsupported ops).

Notes by backend:
- CUDA: Determinism can reduce throughput (5–20% typical) due to restricted algorithm choices. Consider turning off strict determinism for exploratory runs and re‑enabling for final accounting‑critical runs.
- Apple MPS: cuDNN flags are ignored; the `torch.use_deterministic_algorithms` call remains. Expect smaller or no perf impact compared to CUDA.

DP‑SGD under Opacus already introduces overhead (per‑sample grads, clipping, noise). If you need extra speed during iteration:
- Keep dataset/augmentations minimal.
- Use smaller input sizes, lighter models.
- Temporarily relax strict determinism, but re‑enable for final measurements.


