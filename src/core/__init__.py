"""
Core ML components - dataset-agnostic, reusable components.

This module provides general-purpose machine learning components that are
not specific to any particular dataset. These can be reused across different
projects and datasets.
"""

from .model import SimpleCNN
from .data import (
    make_transforms,
    load_stats,
    build_dataloaders,
)
from .utils import (
    get_device,
    set_seed,
    make_worker_init_fn,
    accuracy,
    evaluate,
)

__all__ = [
    # Models
    "SimpleCNN",
    # Data
    "make_transforms",
    "load_stats",
    "build_dataloaders",
    # Utils
    "get_device",
    "set_seed",
    "make_worker_init_fn",
    "accuracy",
    "evaluate",
]

