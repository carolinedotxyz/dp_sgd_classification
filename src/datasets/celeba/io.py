"""Archive and workflow I/O helpers for CelebA workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import os
import pandas as pd


def validate_archive_dir(archive_dir: str) -> Tuple[str, str, str, str]:
    """Validate that required CelebA CSVs exist in the given directory.

    Returns absolute file paths for attributes, partitions, bboxes, and landmarks.
    Raises FileNotFoundError if required CSVs are missing.
    """
    attrs = os.path.join(archive_dir, "list_attr_celeba.csv")
    parts = os.path.join(archive_dir, "list_eval_partition.csv")
    bboxes = os.path.join(archive_dir, "list_bbox_celeba.csv")
    landmarks = os.path.join(archive_dir, "list_landmarks_align_celeba.csv")
    missing = [p for p in (attrs, parts) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Required CSVs not found: {', '.join(missing)}. Archive dir: {archive_dir}"
        )
    return attrs, parts, bboxes, landmarks


def load_archive_paths(archive_dir: Path) -> Tuple[str, str, str, str]:
    """Return key CSV paths from a CelebA archive directory."""
    return validate_archive_dir(str(archive_dir))


def write_archive_outputs(summary_all, archive_dir: Path) -> None:
    """Write summary CSV to archive directory and ensure plot dir exists."""
    out_dir = (archive_dir / "balance_plots"); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = (archive_dir / "celeba_balance_summary.csv")
    summary_all.drop(columns=["display_name","overall_pos_pct"], errors="ignore").to_csv(out_csv, index=False)
    from ...notebooks.display import _relpath
    print(f"\n- Saved: {_relpath(out_csv)}   •   plots → {_relpath(out_dir)}")


def load_archive_data(attrs_csv: str, parts_csv: str) -> pd.DataFrame:
    """Load attributes and partition CSVs and return a merged DataFrame.

    Adds a ``partition_name`` column mapped from the numeric partition id.
    """
    attrs_df = pd.read_csv(attrs_csv)
    parts_df = pd.read_csv(parts_csv)
    if "image_id" not in attrs_df.columns or "image_id" not in parts_df.columns:
        raise ValueError("Both CSVs must include 'image_id' column.")
    # Coerce attribute columns to nullable Int64 and drop rows with missing attributes
    attr_cols = [c for c in attrs_df.columns if c != "image_id"]
    for c in attr_cols:
        attrs_df[c] = pd.to_numeric(attrs_df[c], errors="coerce").astype("Int64")
    # Drop rows that have any missing attribute values (matches prior CLI behavior)
    if attr_cols:
        missing_mask = attrs_df[attr_cols].isna().any(axis=1)
        if missing_mask.any():
            attrs_df = attrs_df.loc[~missing_mask].copy()
    merged = attrs_df.merge(parts_df, on="image_id", how="inner")
    PARTITION_NAME_BY_ID = {0: "train", 1: "val", 2: "test"}
    merged["partition_name"] = merged["partition"].map(PARTITION_NAME_BY_ID)
    return merged


