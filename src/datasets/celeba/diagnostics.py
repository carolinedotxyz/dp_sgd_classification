"""Diagnostics helpers for visual sampling and averaging in CelebA.

These utilities were extracted from the notebook to keep narrative cells clean.
"""

from __future__ import annotations

from typing import List, Tuple

import os
import random
import numpy as np
import pandas as pd

from .index import iter_subset_paths


def select_visual_paths(config) -> list[str]:
    visual_split = config.diag_visual_split
    classes = tuple(config.diag_visual_classes.split(","))
    paths = iter_subset_paths(str(config.subset_root), split=visual_split, classes=classes)
    return _sample_paths(paths, min(int(config.diag_visual_sample), len(paths)), config.random_seed)


def compute_average_original_and_cropped(paths: list[str], target_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    from PIL import Image
    from tqdm import tqdm
    acc_orig = np.zeros((target_size, target_size, 3), dtype=np.float64)
    acc_crop = np.zeros((target_size, target_size, 3), dtype=np.float64)
    count = 0
    for p in tqdm(paths, desc="Averaging images (original & cropped)"):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                imr = im.resize((target_size, target_size), Image.BICUBIC)
                arr_o = np.asarray(imr, dtype=np.float32) / 255.0
                imc = center_square_crop(im)
                imc = imc.resize((target_size, target_size), Image.BICUBIC)
                arr_c = np.asarray(imc, dtype=np.float32) / 255.0
                acc_orig += arr_o
                acc_crop += arr_c
                count += 1
        except (OSError, ValueError):
            continue
    if count > 0:
        avg_orig = (acc_orig / count).clip(0.0, 1.0)
        avg_crop = (acc_crop / count).clip(0.0, 1.0)
        return avg_orig, avg_crop, count
    return acc_orig, acc_crop, 0


# ---------------- Helpers migrated from scripts.celeba_subset_review ----------------

def _sample_paths(paths: List[str], k: int, seed: int) -> List[str]:
    if len(paths) <= k:
        return paths
    rng = random.Random(seed)
    paths = paths.copy()
    rng.shuffle(paths)
    return paths[:k]


def sample_paths(paths: List[str], k: int, seed: int) -> List[str]:
    """Public wrapper to sample up to k paths with a fixed seed."""
    return _sample_paths(paths, k, seed)


def collect_size_stats(paths: List[str]) -> pd.DataFrame:
    from PIL import Image
    from tqdm import tqdm
    import os
    records = []
    # Respect TQDM_DISABLE environment variable
    disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
    for p in tqdm(paths, desc="Reading image sizes", disable=disable_tqdm):
        try:
            with Image.open(p) as im:
                w, h = im.size
            records.append({"path": p, "width": w, "height": h, "aspect": (w / max(h, 1))})
        except Exception:
            continue
    return pd.DataFrame(records)


def describe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    percentiles = [0.0, 0.05, 0.5, 0.95, 1.0]
    return df[cols].describe(percentiles=percentiles).round(2)


def compute_channel_stats(paths: List[str], scale_01: bool) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    from PIL import Image
    from tqdm import tqdm
    import os
    total_sum = np.zeros(3, dtype=np.float64)
    total_sqsum = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    # Respect TQDM_DISABLE environment variable
    disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
    for p in tqdm(paths, desc="Accumulating channel stats", disable=disable_tqdm):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.float32)
                if scale_01:
                    arr = arr / 255.0
                total_sum += arr.sum(axis=(0, 1)).astype(np.float64)
                total_sqsum += np.square(arr).sum(axis=(0, 1)).astype(np.float64)
                total_pixels += arr.shape[0] * arr.shape[1]
        except Exception:
            continue
    if total_pixels == 0:
        return (float("nan"), float("nan"), float("nan")), (float("nan"), float("nan"), float("nan"))
    mean = total_sum / total_pixels
    var = np.maximum(total_sqsum / total_pixels - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return (float(mean[0]), float(mean[1]), float(mean[2])), (float(std[0]), float(std[1]), float(std[2]))


def center_square_crop(img):
    """Return a centered square crop from ``img`` (no resize)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def area_retained_after_center_square(w: int, h: int) -> float:
    side = min(int(w), int(h))
    denom = max(int(w) * int(h), 1)
    return float((side * side) / denom)


