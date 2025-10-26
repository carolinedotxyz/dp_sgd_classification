"""Data loading utilities for the processed CelebA subset.

Provides:
- Transform creation with optional light augmentations
- Loading training statistics (mean/std) from a stats file
- Construction of train/val/test DataLoaders for an ImageFolder layout:
  <root>/{train,val,test}/<class>/image.jpg
"""

import os
import json
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


def make_transforms(mean, std, aug_flip: bool = False, aug_jitter: bool = False):
    """Create a torchvision transform pipeline.

    Args:
        mean: Per-channel mean (length-3 sequence of floats) for normalization.
        std: Per-channel std (length-3 sequence of floats) for normalization.
        aug_flip: If True, apply random horizontal flip with p=0.5.
        aug_jitter: If True, apply light color jitter on brightness/contrast.

    Returns:
        transforms.Compose combining optional augmentations, ToTensor, and Normalize.
    """
    tfms = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    if aug_flip:
        tfms.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    if aug_jitter:
        tfms.insert(0, transforms.ColorJitter(brightness=0.1, contrast=0.1))
    return transforms.Compose(tfms)


def load_stats(stats_path: str):
    """Load training mean/std statistics from a JSON file.

    The file is expected to contain keys "train_mean" and "train_std" as
    3-element sequences of floats.

    Args:
        stats_path: Path to stats.json.

    Returns:
        Tuple (train_mean, train_std).
    """
    with open(stats_path, "r") as f:
        stats = json.load(f)
    mean = stats["train_mean"]
    std = stats["train_std"]
    # Ensure stats are on [0,1] scale to match transforms.ToTensor()
    if not bool(stats.get("normalize_01", True)):
        mean = [m / 255.0 for m in mean]
        std = [s / 255.0 for s in std]
    return mean, std


def build_dataloaders(data_root: str, batch_size: int, num_workers: int,
                      mean, std, aug_flip: bool = False, aug_jitter: bool = False,
                      pin_memory: bool | None = None,
                      worker_init_fn=None,
                      generator: torch.Generator | None = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders for an ImageFolder-based dataset.

    Expects directory structure: <data_root>/{train,val,test}/<class_name>/*.jpg

    Args:
        data_root: Root directory of the processed subset.
        batch_size: Batch size for all splits.
        num_workers: DataLoader worker processes.
        mean: Per-channel mean used by normalization.
        std: Per-channel std used by normalization.
        aug_flip: Enable random horizontal flip on the training split only.
        aug_jitter: Enable light color jitter on the training split only.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_tfms = make_transforms(mean, std, aug_flip=aug_flip, aug_jitter=aug_jitter)
    test_tfms = make_transforms(mean, std, aug_flip=False, aug_jitter=False)

    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=test_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=test_tfms)

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_init_fn, generator=generator
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_init_fn, generator=generator
    )
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_init_fn, generator=generator
    )

    return train_loader, val_loader, test_loader


