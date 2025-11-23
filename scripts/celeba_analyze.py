#!/usr/bin/env python3
"""Utilities to inspect CelebA metadata and summarize attribute balance.

This script validates the expected CelebA CSV files inside an archive
directory, inspects their basic schema and sizes, loads and merges the
attributes with the official train/val/test partition, computes per-attribute
class balance overall and by split, and writes a summary CSV. Optionally, it
renders simple bar plots of the positive fractions.

Typical usage from the command line:
  python scripts/celeba_analyze.py --archive-dir <path> [--attributes ...] [--plots]

Outputs:
- Summary CSV: ``<archive-dir>/celeba_balance_summary.csv`` (or ``--output-csv``)
- Optional plots in ``<archive-dir>/balance_plots`` when ``--plots`` is set
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd


# Constants
PARTITION_NAME_BY_ID: Dict[int, str] = {0: "train", 1: "val", 2: "test"}
REQUIRED_CSV_FILES = ["list_attr_celeba.csv", "list_eval_partition.csv"]
OPTIONAL_CSV_FILES = ["list_bbox_celeba.csv", "list_landmarks_align_celeba.csv"]
METADATA_COLUMNS = ("image_id", "partition", "partition_name")
ATTRIBUTE_VALUE_POSITIVE = 1
ATTRIBUTE_VALUE_NEGATIVE = -1
DEFAULT_ARCHIVE_DIR = "./educational_dp_sgd_image_classifer/data/celeba/archive"
DEFAULT_SUMMARY_CSV = "celeba_balance_summary.csv"
DEFAULT_PLOTS_DIR = "balance_plots"
PLOT_DPI = 200
PLOT_COLOR_PRIMARY = "#4C78A8"
PLOT_COLOR_TRAIN = "#4C78A8"
PLOT_COLOR_VAL = "#F58518"
PLOT_COLOR_TEST = "#54A24B"
INSPECTION_PEEK_ROWS = 5
SCHEMA_PREVIEW_COLS = 8


def validate_archive_dir(archive_dir: str) -> Tuple[str, str, str, str]:
    """Validate that required CelebA CSVs exist in the given directory.

    Args:
        archive_dir: Path to the CelebA archive directory containing CSV files.

    Returns:
        A tuple of absolute file paths: (attrs_csv, parts_csv, bboxes_csv, landmarks_csv).

    Raises:
        FileNotFoundError: If required CSVs (attributes, partitions) are missing.
    """
    attrs = os.path.join(archive_dir, REQUIRED_CSV_FILES[0])
    parts = os.path.join(archive_dir, REQUIRED_CSV_FILES[1])
    bboxes = os.path.join(archive_dir, OPTIONAL_CSV_FILES[0])
    landmarks = os.path.join(archive_dir, OPTIONAL_CSV_FILES[1])
    
    missing: List[str] = [p for p in [attrs, parts] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Required CSVs not found: {', '.join(missing)}. Archive dir: {archive_dir}"
        )
    return attrs, parts, bboxes, landmarks


def inspect_csvs(attrs_csv: str, parts_csv: str, bboxes_csv: str, landmarks_csv: str) -> None:
    """Print a quick inspection of expected CSVs and their schemas.

    Args:
        attrs_csv: Path to ``list_attr_celeba.csv``.
        parts_csv: Path to ``list_eval_partition.csv``.
        bboxes_csv: Path to ``list_bbox_celeba.csv``.
        landmarks_csv: Path to ``list_landmarks_align_celeba.csv``.
    """
    print("CelebA CSVs found:")
    for p in [attrs_csv, parts_csv, bboxes_csv, landmarks_csv]:
        status = "exists" if os.path.isfile(p) else "missing"
        print(f"- {os.path.basename(p)}: {status}")

    # Peek into headers and sizes
    attrs_head = pd.read_csv(attrs_csv, nrows=INSPECTION_PEEK_ROWS)
    parts_head = pd.read_csv(parts_csv, nrows=INSPECTION_PEEK_ROWS)
    print("\nSchemas:")
    preview_cols = list(attrs_head.columns)[:SCHEMA_PREVIEW_COLS]
    print(f"- {os.path.basename(attrs_csv)} -> columns: {preview_cols}... (total {len(attrs_head.columns)})")
    print(f"- {os.path.basename(parts_csv)} -> columns: {list(parts_head.columns)}")

    # Sizes (fast by reading only index)
    attrs_rows = sum(1 for _ in open(attrs_csv, "r", encoding="utf-8")) - 1
    parts_rows = sum(1 for _ in open(parts_csv, "r", encoding="utf-8")) - 1
    print("\nRow counts:")
    print(f"- {os.path.basename(attrs_csv)} -> {attrs_rows:,} rows")
    print(f"- {os.path.basename(parts_csv)} -> {parts_rows:,} rows")


def load_data(attrs_csv: str, parts_csv: str) -> pd.DataFrame:
    """Load attributes and partition CSVs and return a merged DataFrame.

    The result includes all original columns plus ``partition_name`` mapped
    from the numeric partition id. Attribute columns are coerced to nullable
    Int64 with values in {-1, 1}. Rows with missing attribute values are dropped.

    Args:
        attrs_csv: Path to the attributes CSV (must include ``image_id``).
        parts_csv: Path to the partition CSV (must include ``image_id`` and ``partition``).

    Returns:
        A merged DataFrame keyed by ``image_id`` with attribute columns and
        partition metadata.

    Raises:
        ValueError: If required columns are missing.
    """
    attrs_df = pd.read_csv(attrs_csv)
    parts_df = pd.read_csv(parts_csv)

    # Normalize column names
    if "image_id" not in attrs_df.columns or "image_id" not in parts_df.columns:
        raise ValueError("Both CSVs must include 'image_id' column.")

    # Merge on image_id
    merged = attrs_df.merge(parts_df, on="image_id", how="inner")

    # Ensure attribute values are integers -1/1
    attr_cols = [c for c in merged.columns if c not in METADATA_COLUMNS[:2]]
    for c in attr_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").astype("Int64")

    # Drop rows with any missing attribute values (rare)
    missing_mask = merged[attr_cols].isna().any(axis=1)
    if missing_mask.any():
        merged = merged.loc[~missing_mask].copy()

    # Partition names
    merged["partition_name"] = merged["partition"].map(PARTITION_NAME_BY_ID)
    return merged


def compute_balance(df: pd.DataFrame, attributes: List[str]) -> pd.DataFrame:
    """Compute per-attribute class balance overall and by split.

    If ``attributes`` is empty, all attribute columns in ``df`` are used.

    Args:
        df: Merged DataFrame containing attribute columns and ``partition_name``.
        attributes: Optional list of attribute column names to include.

    Returns:
        A DataFrame with one row per attribute containing counts and positive/negative
        fractions overall and for each of train/val/test splits.

    Raises:
        ValueError: If any requested attribute is not present in ``df``.
    """
    # Validate attributes
    available = [c for c in df.columns if c not in METADATA_COLUMNS]
    if not attributes:
        attributes = available
    else:
        invalid = [a for a in attributes if a not in available]
        if invalid:
            raise ValueError(f"Attributes not found: {invalid}")

    rows = []
    for attr in attributes:
        series = df[attr]
        total = int(series.notna().sum())
        pos = int((series == ATTRIBUTE_VALUE_POSITIVE).sum())
        neg = int((series == ATTRIBUTE_VALUE_NEGATIVE).sum())
        row: Dict[str, object] = {
            "attribute": attr,
            "total": total,
            "pos": pos,
            "neg": neg,
            "pos_pct": (pos / total) if total else 0.0,
            "neg_pct": (neg / total) if total else 0.0,
        }
        # By split
        for pid, pname in PARTITION_NAME_BY_ID.items():
            sub = df[df["partition_name"] == pname]
            stotal = int(sub.shape[0])
            npos = int((sub[attr] == ATTRIBUTE_VALUE_POSITIVE).sum())
            nneg = int((sub[attr] == ATTRIBUTE_VALUE_NEGATIVE).sum())
            row[f"{pname}_total"] = stotal
            row[f"{pname}_pos"] = npos
            row[f"{pname}_neg"] = nneg
            row[f"{pname}_pos_pct"] = (npos / stotal) if stotal else 0.0
            row[f"{pname}_neg_pct"] = (nneg / stotal) if stotal else 0.0
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by=["attribute"]).reset_index(drop=True)


def save_plots(df_summary: pd.DataFrame, out_dir: str) -> None:
    """Render bar plots of positive fractions from a balance summary.

    Args:
        df_summary: Output of ``compute_balance``.
        out_dir: Directory to write plot images to.

    Notes:
        If Matplotlib is not available, plotting is skipped with a message.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"Matplotlib not available, skipping plots: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)
    
    # Overall positive percentage plot (sorted)
    df_overall = df_summary[["attribute", "pos_pct"]].sort_values("pos_pct")
    plt.figure(figsize=(10, max(4, len(df_overall) * 0.25)))
    plt.barh(df_overall["attribute"], df_overall["pos_pct"], color=PLOT_COLOR_PRIMARY)
    plt.xlabel("Positive fraction")
    plt.ylabel("Attribute")
    plt.title("CelebA attribute balance (overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "celeba_balance_overall.png"), dpi=PLOT_DPI)
    plt.close()

    # By split stacked bar per attribute (pos pct)
    splits = list(PARTITION_NAME_BY_ID.values())
    split_colors = [PLOT_COLOR_TRAIN, PLOT_COLOR_VAL, PLOT_COLOR_TEST]
    for attr, row in df_summary.set_index("attribute").iterrows():
        vals = [row.get(f"{s}_pos_pct", 0.0) for s in splits]
        plt.figure(figsize=(5, 3))
        plt.bar(splits, vals, color=split_colors)
        plt.ylim(0, 1)
        plt.ylabel("Positive fraction")
        plt.title(f"{attr}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"attr_{attr}_by_split.png"), dpi=PLOT_DPI)
        plt.close()


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of CLI arguments (excluding the program name).

    Returns:
        The populated ``argparse.Namespace`` of parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Inspect CelebA CSVs and compute attribute balance summary."
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=DEFAULT_ARCHIVE_DIR,
        help="Path to CelebA archive directory containing CSVs.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help=f"Path to write the balance summary CSV. Defaults to <archive-dir>/{DEFAULT_SUMMARY_CSV}",
    )
    parser.add_argument(
        "--attributes",
        nargs="*",
        help="Optional list of attribute names to include (default: all).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=f"If set, render plots to <archive-dir>/{DEFAULT_PLOTS_DIR}.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    """Entry point for command-line execution.

    Args:
        argv: List of CLI arguments (excluding the program name).

    Returns:
        Process exit code where 0 indicates success.
    """
    args = parse_args(argv)
    attrs_csv, parts_csv, bboxes_csv, landmarks_csv = validate_archive_dir(args.archive_dir)
    inspect_csvs(attrs_csv, parts_csv, bboxes_csv, landmarks_csv)

    print("\nLoading and merging data...")
    df = load_data(attrs_csv, parts_csv)
    print(f"Merged rows: {df.shape[0]:,}; attributes: {df.shape[1] - 3}")

    attributes = args.attributes or []
    print("Computing balance summary...")
    summary = compute_balance(df, attributes)

    out_csv = (
        args.output_csv
        if args.output_csv
        else os.path.join(args.archive_dir, DEFAULT_SUMMARY_CSV)
    )
    summary.to_csv(out_csv, index=False)
    print(f"Wrote summary CSV -> {out_csv}")

    if args.plots:
        plots_dir = os.path.join(args.archive_dir, DEFAULT_PLOTS_DIR)
        print(f"Rendering plots to {plots_dir} ...")
        save_plots(summary, plots_dir)
        print("Plots saved.")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
