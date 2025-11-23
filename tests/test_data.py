import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.core import make_transforms, load_stats, build_dataloaders


@pytest.fixture()
def tmp_imagefolder(tmp_path: Path):
    # Create ImageFolder structure with tiny RGB squares
    root = tmp_path / "dataset"
    (root / "train" / "class_a").mkdir(parents=True)
    (root / "train" / "class_b").mkdir(parents=True)
    (root / "val" / "class_a").mkdir(parents=True)
    (root / "val" / "class_b").mkdir(parents=True)
    (root / "test" / "class_a").mkdir(parents=True)
    (root / "test" / "class_b").mkdir(parents=True)

    def write_square(p: Path, color):
        arr = np.full((10, 10, 3), color, dtype=np.uint8)
        Image.fromarray(arr).convert("RGB").save(p)

    for i in range(5):
        write_square(root / "train" / "class_a" / f"a_{i}.jpg", (255, 0, 0))
        write_square(root / "train" / "class_b" / f"b_{i}.jpg", (0, 255, 0))
    for i in range(2):
        write_square(root / "val" / "class_a" / f"a_{i}.jpg", (0, 0, 255))
        write_square(root / "val" / "class_b" / f"b_{i}.jpg", (255, 255, 0))
    for i in range(2):
        write_square(root / "test" / "class_a" / f"a_{i}.jpg", (0, 255, 255))
        write_square(root / "test" / "class_b" / f"b_{i}.jpg", (255, 0, 255))

    return root


def test_make_transforms_no_aug():
    mean = [0.5, 0.4, 0.3]
    std = [0.2, 0.2, 0.2]
    tfm = make_transforms(mean, std, aug_flip=False, aug_jitter=False)
    assert hasattr(tfm, "__call__")


def test_make_transforms_with_aug():
    mean = [0.5, 0.4, 0.3]
    std = [0.2, 0.2, 0.2]
    tfm = make_transforms(mean, std, aug_flip=True, aug_jitter=True)
    assert hasattr(tfm, "__call__")


def test_load_stats(tmp_path: Path):
    stats_path = tmp_path / "stats.json"
    payload = {"train_mean": [0.1, 0.2, 0.3], "train_std": [0.9, 0.8, 0.7]}
    stats_path.write_text(json.dumps(payload))
    mean, std = load_stats(str(stats_path))
    assert list(mean) == payload["train_mean"]
    assert list(std) == payload["train_std"]


def test_build_dataloaders_shapes(tmp_imagefolder: Path):
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=str(tmp_imagefolder), batch_size=4, num_workers=0, mean=mean, std=std, aug_flip=False, aug_jitter=False
    )

    # basic length checks
    assert len(train_loader.dataset) == 10
    assert len(val_loader.dataset) == 4
    assert len(test_loader.dataset) == 4

    # fetch one batch to ensure tensors produced and normalized range is sensible
    images, targets = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    assert images.ndim == 4 and images.shape[1] == 3
    assert isinstance(targets, torch.Tensor)


def test_build_dataloaders_aug_flags(tmp_imagefolder: Path):
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    # With flip on; just ensure loader constructs without error
    train_loader, _, _ = build_dataloaders(
        data_root=str(tmp_imagefolder), batch_size=4, num_workers=0, mean=mean, std=std, aug_flip=True, aug_jitter=False
    )
    images, targets = next(iter(train_loader))
    assert images.shape[0] <= 4
