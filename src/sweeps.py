"""Hyperparameter sweep utilities for baseline and DP-SGD training.

This module provides thin wrappers to iterate over configuration objects,
train models, and aggregate results into a ``pandas.DataFrame`` suitable for
analysis or saving to disk.
"""

from typing import List, Dict, Any, Optional
import os, json
import pandas as pd

from .config import BaselineConfig, DPConfig, DPConfigV2
from .train_baseline import train_baseline as _train_baseline
from .train_dp import train_dp_sgd as _train_dp, train_dp_sgd_v2 as _train_dp_v2


def run_baseline_sweep(model_ctor, train_loader, val_loader, test_loader, device, configs: List[BaselineConfig], output_dir: Optional[str] = None) -> pd.DataFrame:
    """Run a sweep of baseline (non-private) training configurations.

    Args:
        model_ctor: Zero-arg callable that returns a new model instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        test_loader: Test dataloader.
        device: Torch device to move models to (e.g., "cpu" or "cuda").
        configs: List of ``BaselineConfig`` instances to evaluate.
        output_dir: Optional directory to save artifacts (unused here; reserved).

    Returns:
        DataFrame where each row corresponds to a configuration and includes
        config values and training/validation/test metrics.
    """
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        model = model_ctor()
        model.to(device)
        _, summary, _ = _train_baseline(model, train_loader, val_loader, test_loader, device, cfg)
        rows.append({
            "run_name": cfg.run_name,
            "optimizer": cfg.optimizer,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            **summary,
        })
    return pd.DataFrame(rows)


def run_dp_sweep(model_ctor, train_loader, val_loader, test_loader, device, configs: List[DPConfig], output_dir: Optional[str] = None) -> pd.DataFrame:
    """Run a sweep of DP-SGD training configurations (core variant).

    Args:
        model_ctor: Zero-arg callable that returns a new model instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        test_loader: Test dataloader.
        device: Torch device to move models to.
        configs: List of ``DPConfig`` instances to evaluate.
        output_dir: Optional directory to save artifacts (unused here; reserved).

    Returns:
        DataFrame summarizing metrics per configuration.
    """
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        model = model_ctor()
        model.to(device)
        _, metrics = _train_dp(model, train_loader, val_loader, test_loader, device, cfg)
        rows.append({
            "run_name": f"nm{cfg.noise_multiplier}_clip{cfg.max_grad_norm}_ep{cfg.epochs}_lr{cfg.lr}",
            "max_grad_norm": cfg.max_grad_norm,
            "noise_multiplier": cfg.noise_multiplier,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            **metrics,
        })
    return pd.DataFrame(rows)


def run_dp_sweep_v2(model_ctor, train_loader, val_loader, test_loader, device, configs: List[DPConfigV2], output_dir: Optional[str] = None) -> pd.DataFrame:
    """Run a sweep of DP-SGD training configurations (extended variant).

    Includes additional options such as schedulers, label smoothing, and light
    data augmentation.

    Args:
        model_ctor: Zero-arg callable that returns a new model instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        test_loader: Test dataloader.
        device: Torch device to move models to.
        configs: List of ``DPConfigV2`` instances to evaluate.
        output_dir: Optional directory to save artifacts (unused here; reserved).

    Returns:
        DataFrame summarizing metrics per configuration.
    """
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        model = model_ctor()
        model.to(device)
        _, metrics = _train_dp_v2(model, train_loader, val_loader, test_loader, device, cfg)
        rows.append({
            "run_name": f"nm{cfg.noise_multiplier}_clip{cfg.max_grad_norm}_ep{cfg.epochs}_lr{cfg.lr}_wd{cfg.weight_decay}_sched{cfg.scheduler}_ls{cfg.label_smoothing}_flip{cfg.aug_flip}_jit{cfg.aug_jitter}",
            "max_grad_norm": cfg.max_grad_norm,
            "noise_multiplier": cfg.noise_multiplier,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "scheduler": cfg.scheduler,
            "label_smoothing": cfg.label_smoothing,
            "aug_flip": cfg.aug_flip,
            "aug_jitter": cfg.aug_jitter,
            **metrics,
        })
    return pd.DataFrame(rows)


