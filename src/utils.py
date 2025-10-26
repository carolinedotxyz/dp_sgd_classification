"""Utility functions for device selection, seeding, and evaluation.

This module centralizes small helpers used across training scripts, including
device detection, reproducibility setup, accuracy calculation, and a simple
evaluation loop for classification models.
"""

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple


def get_device() -> torch.device:
    """Choose an available compute device with a sensible priority.

    Prefers Apple Metal Performance Shaders (MPS) when available, then CUDA,
    otherwise falls back to CPU.

    Returns:
        A ``torch.device`` instance.
    """
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: enable deterministic algorithms when training for strict reproducibility
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def make_worker_init_fn(base_seed: int):
    """Return a worker_init_fn that seeds Python and NumPy per worker deterministically.

    Args:
        base_seed: Base integer seed. Each worker will derive a unique seed.

    Returns:
        A callable suitable for DataLoader(worker_init_fn=...).
    """
    def _init_fn(worker_id: int):
        seed = (base_seed + worker_id * 9973) % (2**32 - 1)
        random.seed(seed)
        np.random.seed(seed)
    return _init_fn


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy from logits and integer class targets.

    Args:
        logits: Model outputs of shape ``(N, C)``.
        targets: Ground-truth labels of shape ``(N,)``.

    Returns:
        Accuracy in ``[0.0, 1.0]`` as a Python float.
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate a classification model and return loss and accuracy.

    The model is set to eval mode and gradients are disabled during iteration.

    Args:
        model: PyTorch model to evaluate.
        loader: Dataloader yielding ``(images, targets)`` batches.
        criterion: Loss function to compute scalar loss.
        device: Device to run evaluation on.

    Returns:
        Tuple ``(avg_loss, avg_accuracy)`` over the dataloader.
    """
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            bsz = images.size(0)
            total_loss += loss.item() * bsz
            total_acc += accuracy(logits, targets) * bsz
            total_n += bsz
    return total_loss / max(1, total_n), total_acc / max(1, total_n)


