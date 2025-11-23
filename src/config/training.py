"""Configuration dataclasses for baseline and DP-SGD training.

These dataclasses centralize commonly used hyperparameters for non-private
baselines and differentially private training variants, providing sensible
defaults suitable for small experiments while remaining explicit and
overrideable via code or CLI adapters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaselineConfig:
    """Non-private baseline training configuration.

    Attributes:
        epochs: Number of training epochs.
        lr: Base learning rate.
        weight_decay: L2 weight decay coefficient.
        optimizer: Optimizer name: "adam", "sgd", or "adamw".
        run_name: Human-friendly run identifier.
        output_dir: Optional output directory for artifacts; ``None`` uses a default.
        seed: Random seed for reproducibility; ``None`` leaves randomness unmanaged.
        save_best: If ``True``, persist the best validation checkpoint.
    """
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    run_name: str = "baseline"
    output_dir: Optional[str] = None
    seed: Optional[int] = 42
    save_best: bool = True


@dataclass
class DPConfig:
    """Core configuration for DP-SGD training.

    Attributes:
        max_grad_norm: Per-sample gradient clipping norm.
        noise_multiplier: Gaussian noise multiplier for DP-SGD.
        target_delta: Target delta for privacy accounting.
        epochs: Number of training epochs.
        lr: Base learning rate.
        weight_decay: L2 weight decay coefficient.
    """
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    target_delta: float = 1e-5
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class DPConfigV2(DPConfig):
    """Extended DP training configuration (builds on ``DPConfig``).

    Inherits all attributes from ``DPConfig`` and adds common training and
    augmentation options.

    Attributes (additions/overrides):
        weight_decay: L2 weight decay coefficient (overrides default in base).
        label_smoothing: Label smoothing factor in [0, 1].
        scheduler: Optional LR scheduler: "cosine", "step", or ``None``.
        step_size: Step size (epochs) for "step" scheduler.
        gamma: Decay factor for "step" scheduler.
        aug_flip: Enable random horizontal flip augmentation.
        aug_jitter: Enable color jitter augmentation (brightness, contrast, saturation).
        aug_rotation: Enable random rotation augmentation.
        rotation_degrees: Maximum rotation angle in degrees (only used if aug_rotation=True).
        seed: Random seed for reproducibility; ``None`` leaves randomness unmanaged.
        output_dir: Optional output directory for artifacts.
    """
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    scheduler: Optional[str] = None  # "cosine", "step", or None
    step_size: int = 2
    gamma: float = 0.5
    aug_flip: bool = True
    aug_jitter: bool = False
    aug_rotation: bool = False
    rotation_degrees: float = 10.0
    seed: Optional[int] = 42
    output_dir: Optional[str] = None


