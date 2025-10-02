"""Processed index I/O and utilities for CelebA workflows.

Centralizes reading/writing and light munging around processed_index.csv so
that notebooks can focus on analysis and visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Optional, List
import os
import logging

import pandas as pd


logger = logging.getLogger(__name__)


def read_processed_index_csv(processed_root: Path) -> pd.DataFrame:
    """Read processed_index.csv under the given root.

    Args:
        processed_root: Root directory where processed_index.csv resides.
    Returns:
        DataFrame of the processed index.
    """
    idx_path = Path(processed_root) / "processed_index.csv"
    return pd.read_csv(idx_path)


def augment_processed_index_with_sizes(processed_root: Path) -> None:
    """Augment processed_index.csv with width/height of processed images.

    This reads processed_index.csv under processed_root, measures each image's
    dimensions, adds two columns: "width" and "height", and writes back.

    Skips unreadable files and records NaN for their size. Uses tqdm if present.
    """
    from PIL import Image as _PILImage
    try:
        from tqdm import tqdm as _tqdm2
    except Exception:
        _tqdm2 = None

    idx_path = Path(processed_root) / "processed_index.csv"
    if not idx_path.is_file():
        logger.warning("processed_index.csv not found at: %s", idx_path)
        return
    try:
        df = pd.read_csv(idx_path)
    except Exception as e:
        logger.warning("Failed reading processed_index.csv: %s", e)
        return

    dest_col = "dest_path" if "dest_path" in df.columns else ("abs_path" if "abs_path" in df.columns else None)
    if dest_col is None:
        logger.warning("processed_index.csv missing 'dest_path' (or 'abs_path'); cannot add size columns")
        return

    widths, heights = [], []
    paths = df[dest_col].astype(str).tolist()
    iterable = _tqdm2(paths, desc="Measuring processed sizes") if _tqdm2 else paths
    for rel_or_abs in iterable:
        ap = rel_or_abs if os.path.isabs(rel_or_abs) else str(Path(processed_root) / rel_or_abs)
        try:
            with _PILImage.open(ap) as _im:
                w, h = _im.size
        except Exception:
            w, h = (float("nan"), float("nan"))
        widths.append(w)
        heights.append(h)

    df["width"] = widths
    df["height"] = heights
    try:
        df.to_csv(idx_path, index=False)
        logger.info("Updated processed_index.csv with width/height columns")
    except Exception as e:
        logger.warning("Failed writing updated processed_index.csv: %s", e)


def load_processed_index_or_raise(out_root: Path) -> pd.DataFrame:
    """Load processed_index.csv or raise if missing."""
    index_csv = Path(out_root) / "processed_index.csv"
    if not index_csv.is_file():
        raise FileNotFoundError(f"processed_index.csv not found at: {index_csv}")
    return pd.read_csv(index_csv)


def ensure_partition_and_class_columns(dfp: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'partition_name' and 'class_name' exist, inferring when possible."""
    if "partition_name" not in dfp.columns:
        if "split" in dfp.columns:
            dfp["partition_name"] = dfp["split"]
        else:
            raise ValueError("processed_index.csv missing 'partition_name' (or 'split').")
    if "class_name" not in dfp.columns:
        if "label" in dfp.columns:
            dfp["class_name"] = dfp["label"].apply(lambda v: "eyeglasses" if int(v) == 1 else "no_eyeglasses")
        else:
            raise ValueError("processed_index.csv missing 'class_name' and 'label' to infer class.")
    return dfp


def summarize_processed_index_df(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Summarize counts by split and class from a processed index DataFrame."""
    by_split = df.groupby("partition_name").size().to_dict()
    by_class = df.groupby("class_name").size().to_dict() if "class_name" in df.columns else {}
    return by_split, by_class


# ---------------- Subset index discovery (migrated from scripts) ----------------

def find_index_csv(root: str) -> Optional[str]:
    for name in os.listdir(root):
        if name.startswith("subset_index_") and name.endswith(".csv"):
            return os.path.join(root, name)
    return None


def load_subset_index(root: str) -> pd.DataFrame:
    """Load subset index from CSV or discover by walking the directory tree.

    Returns a DataFrame with columns: image_id, partition_name, class_name, source_path.
    """
    csv_path = find_index_csv(root)
    if csv_path is not None:
        try:
            df = pd.read_csv(csv_path)
            if "partition_name" not in df.columns and "split" in df.columns:
                df["partition_name"] = df["split"]
            if "class_name" not in df.columns and "label" in df.columns:
                df["class_name"] = df["label"].apply(lambda v: "eyeglasses" if int(v) == 1 else "no_eyeglasses")

            def _reconstruct_src(row):
                split = str(row.get("partition_name", "train"))
                class_name = str(row.get("class_name", "eyeglasses"))
                image_id = str(row.get("image_id", ""))
                p = os.path.join(root, split, class_name, image_id)
                if os.path.isfile(p):
                    return p
                for col in ("dest_path", "source_path"):
                    v = row.get(col)
                    if isinstance(v, str) and v:
                        return v if os.path.isabs(v) else os.path.join(root, v)
                return p

            df["source_path"] = df.apply(_reconstruct_src, axis=1)
            df = df[df["source_path"].apply(lambda p: os.path.isfile(str(p)))]
            if len(df) == 0:
                return _discover_images_from_tree(root)
            keep = [c for c in ["image_id", "partition_name", "class_name", "source_path"] if c in df.columns]
            return df[keep].copy()
        except Exception:
            return _discover_images_from_tree(root)
    else:
        return _discover_images_from_tree(root)


def _discover_images_from_tree(root: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
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
                if not str(fname).lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
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


def iter_subset_paths(root: str, split: Optional[str] = None, classes: Optional[Tuple[str, str]] = ("eyeglasses", "no_eyeglasses")) -> List[str]:
    """Yield file paths from a materialized subset under root via index or tree."""
    df = load_subset_index(root)
    if split is not None:
        df = df[df["partition_name"] == split]
    if classes is not None and "class_name" in df.columns:
        df = df[df["class_name"].isin(list(classes))]
    return df["source_path"].astype(str).tolist()
