#!/usr/bin/env python3
"""Reusable helpers to review CelebA subsets and compute quick stats.

This module centralizes logic that was previously embedded in notebooks,
so notebooks can import these utilities and stay concise.

Functions:
- find_index_csv
- discover_images_from_tree
- load_subset_index
- summarize_balance
- sample_paths
- collect_size_stats
- describe_numeric
- compute_channel_stats
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_index_csv(root: str) -> Optional[str]:
    for name in os.listdir(root):
        if name.startswith("subset_index_") and name.endswith(".csv"):
            return os.path.join(root, name)
    return None


def discover_images_from_tree(root: str) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in ("eyeglasses", "no_eyeglasses"):
            class_dir = os.path.join(split, class_name)
            abs_dir = os.path.join(root, class_dir)
            if not os.path.isdir(abs_dir):
                continue
            for fname in os.listdir(abs_dir):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                rows.append({
                    "image_id": fname,
                    "partition_name": split,
                    "class_name": class_name,
                    "source_path": os.path.join(root, class_dir, fname),
                })
    if not rows:
        raise FileNotFoundError(f"No images found under subset root: {root}")
    return pd.DataFrame(rows)


def load_subset_index(root: str) -> pd.DataFrame:
    csv_path = find_index_csv(root)
    if csv_path is not None:
        try:
            df = pd.read_csv(csv_path)
            # Normalize expected columns
            if "partition_name" not in df.columns and "split" in df.columns:
                df["partition_name"] = df["split"]
            if "class_name" not in df.columns and "label" in df.columns:
                # Fall back to label mapping if class_name missing
                df["class_name"] = df["label"].apply(lambda v: "eyeglasses" if int(v) == 1 else "no_eyeglasses")

            def reconstruct_src(row):
                split = str(row.get("partition_name", "train"))
                class_name = str(row.get("class_name", "eyeglasses"))
                image_id = str(row.get("image_id", ""))
                p = os.path.join(root, split, class_name, image_id)
                if os.path.isfile(p):
                    return p
                # fallback to dest/source columns if present
                for col in ("dest_path", "source_path"):
                    v = row.get(col)
                    if isinstance(v, str) and v:
                        return v if os.path.isabs(v) else os.path.join(root, v)
                return p

            df["source_path"] = df.apply(reconstruct_src, axis=1)
            # Keep only rows that point to files
            df = df[df["source_path"].apply(lambda p: os.path.isfile(str(p)))]
            if len(df) == 0:
                return discover_images_from_tree(root)
            keep = [c for c in ["image_id", "partition_name", "class_name", "source_path"] if c in df.columns]
            return df[keep].copy()
        except Exception:
            return discover_images_from_tree(root)
    else:
        return discover_images_from_tree(root)


def summarize_balance(df: pd.DataFrame) -> pd.DataFrame:
    ctab = df.groupby(["partition_name", "class_name"])['image_id'].count().unstack(fill_value=0)
    ctab["total"] = ctab.sum(axis=1)
    if "eyeglasses" in ctab.columns:
        ctab["pos_ratio"] = (ctab["eyeglasses"] / ctab["total"]).round(4)
    return ctab.sort_index()


def sample_paths(paths: List[str], k: int, seed: int) -> List[str]:
    if len(paths) <= k:
        return paths
    rng = random.Random(seed)
    paths = paths.copy()
    rng.shuffle(paths)
    return paths[:k]


def collect_size_stats(paths: List[str]) -> pd.DataFrame:
    records = []
    for p in tqdm(paths, desc="Reading image sizes"):
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
    total_sum = np.zeros(3, dtype=np.float64)
    total_sqsum = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    for p in tqdm(paths, desc="Accumulating channel stats"):
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


# ---------------- Center-crop utilities & path iteration ----------------

def center_crop_to_square(img: Image.Image) -> Image.Image:
    """Return a centered square crop from ``img`` (no resize)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def area_retained_after_center_square(w: int, h: int) -> float:
    """Fraction of original area retained by a center square crop."""
    side = min(int(w), int(h))
    denom = max(int(w) * int(h), 1)
    return float((side * side) / denom)


def iter_subset_paths(root: str, split: Optional[str] = None, classes: Optional[Tuple[str, str]] = ("eyeglasses", "no_eyeglasses")) -> List[str]:
    """Yield file paths from a materialized subset under ``root``.

    Prefers the index CSV via ``load_subset_index`` for robustness, and filters
    by split and class names when provided.
    """
    df = load_subset_index(root)
    if split is not None:
        df = df[df["partition_name"] == split]
    if classes is not None and "class_name" in df.columns:
        df = df[df["class_name"].isin(list(classes))]
    return df["source_path"].astype(str).tolist()


# ---------------- Quick summary printer (reusable) ----------------

def summarize_subset(root: str) -> None:
    """Print per-split/class counts and positive ratio from a subset index.

    Uses ``load_subset_index`` for robustness, falling back to walking the tree
    when the index CSV is missing.
    """
    from collections import Counter

    items_df = load_subset_index(root)
    counts = Counter(zip(items_df["partition_name"], items_df["class_name"]))
    splits = sorted({s for (s, _) in counts.keys()})
    classes = ("eyeglasses", "no_eyeglasses")

    print("Subset counts (files per split/class):")
    for sp in splits:
        row = {cls: counts.get((sp, cls), 0) for cls in classes}
        total = sum(row.values())
        pos_ratio = (row.get("eyeglasses", 0) / total) if total else float("nan")
        print(
            f"  {sp:>5}: eyeglasses={row['eyeglasses']:>5}  "
            f"no_eyeglasses={row['no_eyeglasses']:>5}  total={total:>5}  pos_ratio={pos_ratio:.4f}"
        )


