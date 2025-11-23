#!/usr/bin/env python3
"""Build a balanced CelebA subset for a single binary attribute.

This script discovers the required CelebA metadata and image directories, merges
attributes with official train/val/test partitions, optionally enriches with
landmarks and bounding boxes, and materializes a balanced subset per split by
either symlinking or copying files. It also writes an index CSV containing
the selections and any requested metadata.

Typical usage:
  python scripts/celeba_build_subset.py --archive-dir <path> --attribute Eyeglasses \
    --output-dir <out> [--max-per-class 500] [--link-mode symlink|copy] \
    [--use-landmarks --min-iod 20 --max-iod 120] [--include-attrs ...]

Outputs:
- Subset directory tree under ``--output-dir`` with class subfolders per split
- Index CSV at ``<output-dir>/subset_index_<attribute>.csv``
"""

import argparse
import os
import sys
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd


# Constants
PARTITION_NAME_BY_ID: Dict[int, str] = {0: "train", 1: "val", 2: "test"}
PARTITION_NAMES = list(PARTITION_NAME_BY_ID.values())

# File names
CSV_ATTRS = "list_attr_celeba.csv"
CSV_PARTS = "list_eval_partition.csv"
CSV_BBOXES = "list_bbox_celeba.csv"
CSV_LANDMARKS = "list_landmarks_align_celeba.csv"
IMAGES_DIR_NESTED = "img_align_celeba/img_align_celeba"
IMAGES_DIR_FLAT = "img_align_celeba"

# Attribute values
ATTR_VALUE_POSITIVE = 1
ATTR_VALUE_NEGATIVE = -1
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0

# Class names
CLASS_POSITIVE = "eyeglasses"
CLASS_NEGATIVE = "no_eyeglasses"

# Default values
DEFAULT_ARCHIVE_DIR = "./data/celeba/archive"
DEFAULT_OUTPUT_DIR = "./data/celeba/subsets/eyeglasses_balanced"
DEFAULT_ATTRIBUTE = "Eyeglasses"
DEFAULT_SEED = 1337
DEFAULT_IOD_BINS = 4

# Seed offsets for different random operations
SEED_OFFSET_NEGATIVE_POOL = 1
SEED_OFFSET_BACKFILL_POS = 137
SEED_OFFSET_BACKFILL_NEG = 138

# Bounding box required columns
BBOX_COLUMNS = ["x_1", "y_1", "width", "height"]

# Landmark columns
LANDMARK_EYE_COLUMNS = ["lefteye_x", "lefteye_y", "righteye_x", "righteye_y"]
LANDMARK_ALL_COLUMNS = [
    "lefteye_x", "lefteye_y", "righteye_x", "righteye_y",
    "nose_x", "nose_y", "leftmouth_x", "leftmouth_y", "rightmouth_x", "rightmouth_y",
]

# Metadata columns to exclude from attributes
METADATA_COLUMNS = ("image_id", "partition", "partition_name")

# Link modes
LINK_MODE_SYMLINK = "symlink"
LINK_MODE_COPY = "copy"


@dataclass
class Selection:
    """A single selected image for inclusion in the subset.

    Attributes:
        image_id: Filename as listed in CelebA (e.g., ``000001.jpg``).
        label: Binary class label (1=positive attribute, 0=negative).
        partition_name: Split name: ``train``, ``val``, or ``test``.
        source_path: Absolute path to the source image.
        dest_path: Absolute path where the file will be created.
    """
    image_id: str
    label: int  # 0=no eyeglasses, 1=eyeglasses
    partition_name: str
    source_path: str
    dest_path: str


def validate_archive_dir(archive_dir: str) -> Tuple[str, str, str]:
    """Validate required files and locate the images root directory.

    Args:
        archive_dir: Path to a CelebA archive directory.

    Returns:
        Tuple of paths ``(attrs_csv, parts_csv, images_root)``.

    Raises:
        FileNotFoundError: If required CSVs or the images directory are missing.
    """
    attrs = os.path.join(archive_dir, CSV_ATTRS)
    parts = os.path.join(archive_dir, CSV_PARTS)
    images_root_candidates: List[str] = [
        os.path.join(archive_dir, IMAGES_DIR_NESTED),
        os.path.join(archive_dir, IMAGES_DIR_FLAT),
    ]
    
    if not os.path.isfile(attrs) or not os.path.isfile(parts):
        raise FileNotFoundError(
            f"Required CSVs not found in {archive_dir}. "
            f"Expected files: {CSV_ATTRS}, {CSV_PARTS}"
        )
    
    images_root = ""
    for cand in images_root_candidates:
        if os.path.isdir(cand):
            images_root = cand
            break
    
    if not images_root:
        raise FileNotFoundError(
            f"Could not find images directory. Looked for '{IMAGES_DIR_NESTED}' "
            f"and '{IMAGES_DIR_FLAT}' under archive dir."
        )
    
    return attrs, parts, images_root


def load_and_merge(attrs_csv: str, parts_csv: str) -> pd.DataFrame:
    """Load attributes and partitions and return a merged DataFrame.

    Adds a ``partition_name`` column mapped from the numeric partition id.

    Args:
        attrs_csv: Path to ``list_attr_celeba.csv`` (must include ``image_id``).
        parts_csv: Path to ``list_eval_partition.csv`` (must include ``image_id``).

    Returns:
        Merged DataFrame keyed by ``image_id``.

    Raises:
        ValueError: If required columns are missing.
    """
    attrs_df = pd.read_csv(attrs_csv)
    parts_df = pd.read_csv(parts_csv)
    
    if "image_id" not in attrs_df.columns or "image_id" not in parts_df.columns:
        raise ValueError("Both CSVs must include 'image_id'.")
    
    merged = attrs_df.merge(parts_df, on="image_id", how="inner")
    merged["partition_name"] = merged["partition"].map(PARTITION_NAME_BY_ID)
    return merged


def load_landmarks(archive_dir: str) -> pd.DataFrame:
    """Load aligned landmark coordinates CSV.

    Args:
        archive_dir: Path to a CelebA archive directory containing the CSV.

    Returns:
        DataFrame of raw landmark coordinates.

    Raises:
        FileNotFoundError: If the landmarks CSV is missing.
    """
    path = os.path.join(archive_dir, CSV_LANDMARKS)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Landmarks CSV not found at {path}")
    return pd.read_csv(path)


def load_bboxes(archive_dir: str) -> pd.DataFrame:
    """Load bounding boxes CSV and validate expected columns.

    Args:
        archive_dir: Path to a CelebA archive directory containing the CSV.

    Returns:
        DataFrame with required bbox columns present.

    Raises:
        FileNotFoundError: If the bbox CSV is missing.
        ValueError: If required columns are missing.
    """
    path = os.path.join(archive_dir, CSV_BBOXES)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"BBox CSV not found at {path}")
    
    df = pd.read_csv(path)
    for c in BBOX_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"BBox column missing: {c}")
    return df


def to_lookup(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, object]]:
    """Build an ``image_id``-keyed lookup for selected columns.

    Args:
        df: DataFrame containing an ``image_id`` column.
        cols: Columns to include in the lookup value dict.

    Returns:
        Mapping ``image_id -> {col: value, ...}``.
    """
    return df.set_index("image_id")[cols].to_dict("index")


def compute_interocular_distance(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """Compute inter-ocular distance from landmark coordinates.

    Args:
        landmarks_df: DataFrame with eye center columns.

    Returns:
        DataFrame with columns ``image_id`` and ``interocular``.

    Raises:
        ValueError: If required landmark columns are missing.
    """
    for c in LANDMARK_EYE_COLUMNS:
        if c not in landmarks_df.columns:
            raise ValueError(f"Landmarks column missing: {c}")
    
    ld = landmarks_df.copy()
    for c in LANDMARK_EYE_COLUMNS:
        ld[c] = pd.to_numeric(ld[c], errors="coerce")
    
    # Euclidean distance between eye centers
    dx = ld["lefteye_x"] - ld["righteye_x"]
    dy = ld["lefteye_y"] - ld["righteye_y"]
    ld["interocular"] = (dx ** 2 + dy ** 2) ** 0.5
    
    return ld[["image_id", "interocular"]]


def merge_and_filter_by_landmarks(
    merged_df: pd.DataFrame,
    archive_dir: str,
    min_iod: Optional[float],
    max_iod: Optional[float],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Merge inter-ocular distance and filter rows by range.

    Args:
        merged_df: Merged attributes/partition DataFrame.
        archive_dir: Path that contains the landmarks CSV.
        min_iod: Minimum allowed inter-ocular distance; pass ``None`` to skip.
        max_iod: Maximum allowed inter-ocular distance; pass ``None`` to skip.

    Returns:
        A tuple ``(filtered_df, iod_lookup)`` where ``iod_lookup`` maps ``image_id``
        to inter-ocular distance.
    """
    landmarks = load_landmarks(archive_dir)
    iod_df = compute_interocular_distance(landmarks)
    with_iod = merged_df.merge(iod_df, on="image_id", how="inner")
    
    if min_iod is not None:
        with_iod = with_iod[with_iod["interocular"] >= float(min_iod)]
    if max_iod is not None:
        with_iod = with_iod[with_iod["interocular"] <= float(max_iod)]
    
    # Build a fast lookup for index CSV enrichment
    lookup = with_iod.set_index("image_id")["interocular"].to_dict()
    return with_iod, lookup


def get_shuffled_pools(
    df: pd.DataFrame,
    attribute: str,
    seed: int,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create shuffled positive/negative pools per split for a given attribute.

    Args:
        df: DataFrame including ``partition_name`` and the attribute column.
        attribute: Attribute column name to binarize (1 -> positive, else negative).
        seed: Random seed used for shuffling.

    Returns:
        Mapping ``split -> (positive_pool_df, negative_pool_df)``.

    Raises:
        ValueError: If the attribute column is missing.
    """
    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in CSV columns.")
    
    # Normalize labels to 0/1
    df = df[["image_id", "partition_name", attribute]].copy()
    df[attribute] = pd.to_numeric(df[attribute], errors="coerce").fillna(ATTR_VALUE_NEGATIVE).astype(int)
    df["label"] = (df[attribute] == ATTR_VALUE_POSITIVE).astype(int)

    pools: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for pname in PARTITION_NAMES:
        sub = df[df["partition_name"] == pname]
        pos_pool = sub[sub["label"] == LABEL_POSITIVE].sample(
            frac=1.0, random_state=seed
        ).reset_index(drop=True)
        neg_pool = sub[sub["label"] == LABEL_NEGATIVE].sample(
            frac=1.0, random_state=seed + SEED_OFFSET_NEGATIVE_POOL
        ).reset_index(drop=True)
        pools[pname] = (pos_pool, neg_pool)
    
    return pools


def ensure_dir(path: str) -> None:
    """Create directory ``path`` if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def materialize_selection(
    selections: List[Selection],
    link_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Create files for the chosen selections using the requested mode.

    Args:
        selections: Items to materialize on disk.
        link_mode: ``"symlink"`` to symlink or ``"copy"`` to copy files.
        overwrite: Whether to overwrite existing paths.
        dry_run: If ``True``, only print actions without changing the filesystem.

    Raises:
        ValueError: If ``link_mode`` is invalid.
    """
    for sel in selections:
        if dry_run:
            print(f"DRY-RUN would create: {sel.dest_path} -> {sel.source_path}")
            continue
        
        ensure_dir(os.path.dirname(sel.dest_path))
        
        if os.path.lexists(sel.dest_path):
            if overwrite:
                if os.path.islink(sel.dest_path) or os.path.isfile(sel.dest_path):
                    os.remove(sel.dest_path)
                else:
                    shutil.rmtree(sel.dest_path)
            else:
                # Skip existing
                continue
        
        if link_mode == LINK_MODE_SYMLINK:
            os.symlink(sel.source_path, sel.dest_path)
        elif link_mode == LINK_MODE_COPY:
            shutil.copy2(sel.source_path, sel.dest_path)
        else:
            raise ValueError(f"link_mode must be '{LINK_MODE_SYMLINK}' or '{LINK_MODE_COPY}'")


def discretize_interocular(
    df: pd.DataFrame,
    num_bins: int,
    seed: int,
) -> pd.Series:
    """Discretize inter-ocular distance into bins (quantiles when possible).

    Args:
        df: DataFrame including an ``interocular`` column.
        num_bins: Desired number of bins.
        seed: Random seed (unused; kept for API symmetry).

    Returns:
        Series of string bin labels aligned to ``df.index``.
    """
    series = pd.to_numeric(df["interocular"], errors="coerce")
    valid = series.dropna()
    
    if valid.empty:
        return pd.Series(["nan"] * len(df), index=df.index)
    
    # Quantile bins
    try:
        bins = pd.qcut(
            valid,
            q=min(num_bins, max(1, valid.nunique())),
            duplicates="drop"
        )
    except Exception:
        bins = pd.cut(valid, bins=num_bins)
    
    out = pd.Series(["nan"] * len(df), index=df.index, dtype=object)
    out.loc[valid.index] = bins.astype(str)
    return out


def build_strata_key(df: pd.DataFrame, strat_cols: List[str]) -> pd.Series:
    """Combine columns into a single string key for stratified grouping.

    Args:
        df: Input DataFrame.
        strat_cols: Column names to combine.

    Returns:
        Series of pipe-delimited keys, e.g., ``"Female|Young|bin_2"``.
    """
    return df[strat_cols].astype(str).agg("|".join, axis=1)


def allocate_per_stratum(
    min_counts: Dict[str, int],
    desired: int,
) -> Dict[str, int]:
    """Allocate per-stratum sample counts proportionally with greedy rounding.

    Args:
        min_counts: Minimum available items per stratum (balanced across classes).
        desired: Total desired items per class to allocate across strata.

    Returns:
        Mapping of ``stratum -> allocated_count``.
    """
    if not min_counts:
        return {}
    
    total_min = sum(min_counts.values())
    if total_min == 0:
        return {k: 0 for k in min_counts.keys()}
    
    # Proportional allocation with rounding and greedy fill
    alloc = {k: int(min_counts[k] * desired / total_min) for k in min_counts}
    remaining = desired - sum(alloc.values())
    
    # Greedy distribute leftover to strata with most slack
    slack = {k: min_counts[k] - alloc[k] for k in min_counts}
    for k in sorted(slack.keys(), key=lambda x: slack[x], reverse=True):
        if remaining <= 0:
            break
        if slack[k] > 0:
            alloc[k] += 1
            remaining -= 1
    
    return alloc


def sample_stratified_split(
    sub_df: pd.DataFrame,
    label_col: str,
    strat_cols: List[str],
    desired_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Sample a balanced split using stratification across given columns.

    If no strata overlap between classes, falls back to simple balanced sampling.

    Args:
        sub_df: Candidate rows for a single split.
        label_col: Binary label column name (0/1).
        strat_cols: Columns used to define strata.
        desired_per_class: Target count per class to sample.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of sampled rows, shuffled.
    """
    # Compute per-stratum min counts and sample that many from each class
    work = sub_df.copy()
    work["__stratum__"] = build_strata_key(work, strat_cols)
    pos = work[work[label_col] == LABEL_POSITIVE]
    neg = work[work[label_col] == LABEL_NEGATIVE]
    
    # Counts per stratum
    pos_counts = pos["__stratum__"].value_counts().to_dict()
    neg_counts = neg["__stratum__"].value_counts().to_dict()
    common = set(pos_counts.keys()) & set(neg_counts.keys())
    min_counts = {k: min(pos_counts[k], neg_counts[k]) for k in common}
    
    if not min_counts:
        # Fallback to simple balanced sampling if no overlap
        limit = min(len(pos), len(neg), desired_per_class)
        return pd.concat([
            pos.sample(n=limit, random_state=seed, replace=False),
            neg.sample(n=limit, random_state=seed + SEED_OFFSET_NEGATIVE_POOL, replace=False),
        ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    alloc = allocate_per_stratum(min_counts, desired_per_class)
    selected = []
    
    for k, n in alloc.items():
        if n <= 0:
            continue
        
        pos_k = pos[pos["__stratum__"] == k]
        neg_k = neg[neg["__stratum__"] == k]
        m = min(n, len(pos_k), len(neg_k))
        
        if m <= 0:
            continue
        
        selected.append(pos_k.sample(n=m, random_state=seed, replace=False))
        selected.append(neg_k.sample(n=m, random_state=seed + SEED_OFFSET_NEGATIVE_POOL, replace=False))
    
    return pd.concat(selected, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _build_row_metadata(
    image_id: str,
    label: int,
    partition_name: str,
    source_path: str,
    dest_path: str,
    link_mode: str,
    output_dir: str,
    interocular: Optional[float],
    attrs_to_include: Optional[List[str]],
    attrs_lookup: Optional[Dict[str, Dict[str, object]]],
    bbox_lookup: Optional[Dict[str, Dict[str, object]]],
    landmarks_raw_lookup: Optional[Dict[str, Dict[str, object]]],
) -> Dict[str, object]:
    """Build metadata row for index CSV.

    Args:
        image_id: Image identifier.
        label: Binary label (0 or 1).
        partition_name: Split name.
        source_path: Absolute source path.
        dest_path: Absolute destination path.
        link_mode: Link mode (symlink or copy).
        output_dir: Output directory root.
        interocular: Optional inter-ocular distance.
        attrs_to_include: Optional list of attribute names to include.
        attrs_lookup: Optional attribute lookup dictionary.
        bbox_lookup: Optional bounding box lookup dictionary.
        landmarks_raw_lookup: Optional landmarks lookup dictionary.

    Returns:
        Dictionary of metadata fields for the index CSV row.
    """
    class_name = CLASS_POSITIVE if label == LABEL_POSITIVE else CLASS_NEGATIVE
    
    # Record paths: for copy mode, store paths relative to dataset root
    if link_mode == LINK_MODE_COPY:
        rec_source = os.path.relpath(source_path, output_dir)
        rec_dest = os.path.relpath(dest_path, output_dir)
    else:
        rec_source = source_path
        rec_dest = dest_path
    
    row: Dict[str, object] = {
        "image_id": image_id,
        "label": label,
        "class_name": class_name,
        "partition_name": partition_name,
        "source_path": rec_source,
        "dest_path": rec_dest,
    }
    
    if interocular is not None:
        row["interocular"] = interocular
    
    # Optional enrichments
    if attrs_to_include and attrs_lookup is not None:
        attr_data = attrs_lookup.get(image_id)
        if attr_data:
            for k in attrs_to_include:
                if k in attr_data:
                    row[k] = attr_data[k]
    
    if bbox_lookup is not None:
        bbox_data = bbox_lookup.get(image_id)
        if bbox_data:
            row.update(bbox_data)
    
    if landmarks_raw_lookup is not None:
        landmark_data = landmarks_raw_lookup.get(image_id)
        if landmark_data:
            row.update(landmark_data)
    
    return row


def _process_split(
    df: pd.DataFrame,
    partition_name: str,
    images_root: str,
    output_dir: str,
    attribute: str,
    link_mode: str,
    seed: int,
    max_per_class: Optional[int],
    dry_run: bool,
    strict_missing: bool,
    fill_missing: bool,
    stratify_by: Optional[List[str]],
    strat_cols: Optional[List[str]],
    landmarks_lookup: Optional[Dict[str, float]],
    attrs_to_include: Optional[List[str]],
    attrs_lookup: Optional[Dict[str, Dict[str, object]]],
    bbox_lookup: Optional[Dict[str, Dict[str, object]]],
    landmarks_raw_lookup: Optional[Dict[str, Dict[str, object]]],
) -> Tuple[List[Selection], List[Dict[str, object]]]:
    """Process a single split to create selections and metadata.

    Args:
        df: DataFrame for the split.
        partition_name: Name of the split (train/val/test).
        images_root: Root directory containing source images.
        output_dir: Output directory for the subset.
        attribute: Attribute name being balanced.
        link_mode: Link mode (symlink or copy).
        seed: Random seed.
        max_per_class: Maximum samples per class for this split.
        dry_run: If True, don't actually create files.
        strict_missing: If True, raise on missing images.
        fill_missing: If True, backfill missing images.
        stratify_by: Optional list of columns to stratify by.
        strat_cols: Processed stratification columns.
        landmarks_lookup: Optional inter-ocular distance lookup.
        attrs_to_include: Optional list of attributes to include.
        attrs_lookup: Optional attribute lookup.
        bbox_lookup: Optional bounding box lookup.
        landmarks_raw_lookup: Optional landmarks lookup.

    Returns:
        Tuple of (selections, metadata_rows).
    """
    sub = df[df["partition_name"] == partition_name]
    
    # Calculate desired per class
    pos_count = int((sub["label"] == LABEL_POSITIVE).sum())
    neg_count = int((sub["label"] == LABEL_NEGATIVE).sum())
    desired = min(pos_count, neg_count)
    
    if max_per_class is not None:
        desired = min(desired, max_per_class)
    
    print(f"Split {partition_name}: target per-class cap {desired}")
    
    # Choose rows either stratified or simple
    if stratify_by and desired > 0 and strat_cols:
        chosen = sample_stratified_split(
            sub_df=sub,
            label_col="label",
            strat_cols=strat_cols,
            desired_per_class=desired,
            seed=seed,
        )
    else:
        pos = sub[sub["label"] == LABEL_POSITIVE]
        neg = sub[sub["label"] == LABEL_NEGATIVE]
        limit = min(len(pos), len(neg), desired)
        chosen = pd.concat([
            pos.sample(n=limit, random_state=seed, replace=False),
            neg.sample(n=limit, random_state=seed + SEED_OFFSET_NEGATIVE_POOL, replace=False),
        ]).sample(frac=1.0, random_state=seed)
    
    selections: List[Selection] = []
    rows_meta: List[Dict[str, object]] = []
    kept_pos = kept_neg = 0
    selected_ids = set()
    
    # Process chosen rows
    for _, row in chosen.iterrows():
        image_id = str(row["image_id"])
        label_val = int(row["label"])
        source_path = os.path.join(images_root, image_id)
        
        # Check if file exists
        if not dry_run and not os.path.isfile(source_path):
            if strict_missing:
                raise FileNotFoundError(f"Image not found: {source_path}")
            print(f"Skipping missing image: {source_path}")
            continue
        
        class_name = CLASS_POSITIVE if label_val == LABEL_POSITIVE else CLASS_NEGATIVE
        dest_path = os.path.join(output_dir, partition_name, class_name, image_id)
        
        interocular_val = landmarks_lookup.get(image_id) if landmarks_lookup else None
        
        # Build metadata row
        meta_row = _build_row_metadata(
            image_id=image_id,
            label=label_val,
            partition_name=partition_name,
            source_path=source_path,
            dest_path=dest_path,
            link_mode=link_mode,
            output_dir=output_dir,
            interocular=interocular_val,
            attrs_to_include=attrs_to_include,
            attrs_lookup=attrs_lookup,
            bbox_lookup=bbox_lookup,
            landmarks_raw_lookup=landmarks_raw_lookup,
        )
        
        selections.append(Selection(
            image_id=image_id,
            label=label_val,
            partition_name=partition_name,
            source_path=source_path,
            dest_path=dest_path,
        ))
        rows_meta.append(meta_row)
        selected_ids.add(image_id)
        
        if label_val == LABEL_POSITIVE:
            kept_pos += 1
        else:
            kept_neg += 1
    
    # Backfill if needed
    if fill_missing and (kept_pos < desired or kept_neg < desired):
        def backfill_for_label(target_label: int, need: int) -> int:
            if need <= 0:
                return 0
            
            seed_offset = SEED_OFFSET_BACKFILL_POS if target_label == LABEL_POSITIVE else SEED_OFFSET_BACKFILL_NEG
            candidates = (
                sub[(sub["label"] == target_label) & (~sub["image_id"].isin(selected_ids))]
                .sample(frac=1.0, random_state=seed + seed_offset)
                .reset_index(drop=True)
            )
            
            added = 0
            for _, crow in candidates.iterrows():
                c_image_id = str(crow["image_id"])
                c_source_path = os.path.join(images_root, c_image_id)
                
                if not dry_run and not os.path.isfile(c_source_path):
                    if strict_missing:
                        raise FileNotFoundError(f"Image not found: {c_source_path}")
                    continue
                
                c_class_name = CLASS_POSITIVE if target_label == LABEL_POSITIVE else CLASS_NEGATIVE
                c_dest_path = os.path.join(output_dir, partition_name, c_class_name, c_image_id)
                c_interocular = landmarks_lookup.get(c_image_id) if landmarks_lookup else None
                
                c_meta_row = _build_row_metadata(
                    image_id=c_image_id,
                    label=target_label,
                    partition_name=partition_name,
                    source_path=c_source_path,
                    dest_path=c_dest_path,
                    link_mode=link_mode,
                    output_dir=output_dir,
                    interocular=c_interocular,
                    attrs_to_include=attrs_to_include,
                    attrs_lookup=attrs_lookup,
                    bbox_lookup=bbox_lookup,
                    landmarks_raw_lookup=landmarks_raw_lookup,
                )
                
                selections.append(Selection(
                    image_id=c_image_id,
                    label=target_label,
                    partition_name=partition_name,
                    source_path=c_source_path,
                    dest_path=c_dest_path,
                ))
                rows_meta.append(c_meta_row)
                selected_ids.add(c_image_id)
                added += 1
                
                if added >= need:
                    break
            
            return added
        
        add_pos = backfill_for_label(LABEL_POSITIVE, desired - kept_pos)
        kept_pos += add_pos
        add_neg = backfill_for_label(LABEL_NEGATIVE, desired - kept_neg)
        kept_neg += add_neg
    
    print(f"  Kept {kept_pos} pos, {kept_neg} neg (total {kept_pos + kept_neg})")
    if kept_pos < desired or kept_neg < desired:
        print(
            f"  Warning: could not reach desired cap for split '{partition_name}'. "
            f"Available pos={kept_pos}, neg={kept_neg}."
        )
    
    return selections, rows_meta


def build_subset(
    merged: pd.DataFrame,
    images_root: str,
    output_dir: str,
    attribute: str,
    link_mode: str,
    seed: int,
    max_per_class_by_split: Dict[str, Optional[int]],
    overwrite: bool,
    dry_run: bool,
    strict_missing: bool,
    fill_missing: bool,
    landmarks_lookup: Optional[Dict[str, float]],
    attrs_to_include: Optional[List[str]],
    attrs_lookup: Optional[Dict[str, Dict[str, object]]],
    bbox_lookup: Optional[Dict[str, Dict[str, object]]],
    landmarks_raw_lookup: Optional[Dict[str, Dict[str, object]]],
    stratify_by: Optional[List[str]],
    iod_bins: Optional[int],
) -> None:
    """Create a balanced per-split subset and write an index CSV.

    Per split, selects up to the specified per-class cap, balanced between
    positive and negative classes. Optionally stratifies by additional columns
    to better match distributions across classes.

    Args:
        merged: Merged attributes/partitions DataFrame (optionally filtered).
        images_root: Directory containing CelebA images.
        output_dir: Destination directory for the subset.
        attribute: Attribute to balance on (e.g., ``Eyeglasses``).
        link_mode: ``"symlink"`` to symlink or ``"copy"`` to copy files.
        seed: Random seed for reproducibility.
        max_per_class_by_split: Optional per-split caps for each class.
        overwrite: Overwrite existing files when materializing.
        dry_run: If ``True``, do not write files; print planned actions only.
        strict_missing: If ``True``, raise on missing images, else skip.
        fill_missing: If ``True``, backfill missing files with new candidates.
        landmarks_lookup: Optional mapping ``image_id -> interocular``.
        attrs_to_include: Optional list of attribute columns to include in index CSV.
        attrs_lookup: Optional mapping ``image_id -> {attr: value}``.
        bbox_lookup: Optional mapping ``image_id -> {x_1,y_1,width,height}``.
        landmarks_raw_lookup: Optional mapping of raw landmark coordinates.
        stratify_by: Optional list of columns to stratify on; supports ``"interocular"``.
        iod_bins: Number of bins when discretizing inter-ocular distance.
    """
    # Ensure label column exists
    df = merged.copy()
    df[attribute] = pd.to_numeric(df[attribute], errors="coerce").fillna(ATTR_VALUE_NEGATIVE).astype(int)
    df["label"] = (df[attribute] == ATTR_VALUE_POSITIVE).astype(int)

    # If stratifying on interocular, ensure we have a binned column
    strat_cols: Optional[List[str]] = None
    if stratify_by:
        strat_cols = list(stratify_by)
        if "interocular" in strat_cols:
            if "interocular" not in df.columns:
                raise ValueError(
                    "Stratify by interocular requires --use-landmarks to add interocular column."
                )
            bins = discretize_interocular(df, iod_bins or DEFAULT_IOD_BINS, seed)
            df["iod_bin"] = bins
            strat_cols = [c if c != "interocular" else "iod_bin" for c in strat_cols]

    # Process each split
    all_selections: List[Selection] = []
    all_rows_meta: List[Dict[str, object]] = []
    
    for pname in PARTITION_NAMES:
        selections, rows_meta = _process_split(
            df=df,
            partition_name=pname,
            images_root=images_root,
            output_dir=output_dir,
            attribute=attribute,
            link_mode=link_mode,
            seed=seed,
            max_per_class=max_per_class_by_split.get(pname),
            dry_run=dry_run,
            strict_missing=strict_missing,
            fill_missing=fill_missing,
            stratify_by=stratify_by,
            strat_cols=strat_cols,
            landmarks_lookup=landmarks_lookup,
            attrs_to_include=attrs_to_include,
            attrs_lookup=attrs_lookup,
            bbox_lookup=bbox_lookup,
            landmarks_raw_lookup=landmarks_raw_lookup,
        )
        all_selections.extend(selections)
        all_rows_meta.extend(rows_meta)

    # Write index CSV
    ensure_dir(output_dir)
    index_csv = os.path.join(output_dir, f"subset_index_{attribute.lower()}.csv")
    index_df = pd.DataFrame(all_rows_meta)
    
    if dry_run:
        print(f"DRY-RUN would write index CSV: {index_csv} with {len(index_df)} rows")
    else:
        index_df.to_csv(index_csv, index=False)
        print(f"Wrote index CSV: {index_csv}")

    # Materialize files
    materialize_selection(
        all_selections,
        link_mode=link_mode,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    
    if not dry_run:
        print(f"Created files under: {output_dir}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments for subset building.

    Args:
        argv: List of CLI arguments (excluding program name).

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Create a balanced subset for a binary CelebA attribute (e.g., Eyeglasses)."
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=DEFAULT_ARCHIVE_DIR,
        help="Path to CelebA archive directory containing CSVs and images folder.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Path to images root. Defaults to autodetected under archive dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to create the balanced subset into.",
    )
    parser.add_argument(
        "--attribute",
        type=str,
        default=DEFAULT_ATTRIBUTE,
        help=f"Attribute to balance on (default: {DEFAULT_ATTRIBUTE}).",
    )
    parser.add_argument(
        "--link-mode",
        choices=[LINK_MODE_SYMLINK, LINK_MODE_COPY],
        default=LINK_MODE_SYMLINK,
        help="Create symlinks (default) or copy image files into subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class per split. Defaults to the minority class size.",
    )
    parser.add_argument(
        "--max-per-class-train",
        type=int,
        default=None,
        help="Optional cap per class for train split only.",
    )
    parser.add_argument(
        "--max-per-class-val",
        type=int,
        default=None,
        help="Optional cap per class for val split only.",
    )
    parser.add_argument(
        "--max-per-class-test",
        type=int,
        default=None,
        help="Optional cap per class for test split only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not create files; only print planned actions and counts.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Error on missing images instead of skipping (default: skip).",
    )
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="Do not backfill missing files with new candidates (default: backfill).",
    )
    parser.add_argument(
        "--use-landmarks",
        action="store_true",
        help="Merge landmarks and compute inter-ocular distance for filtering and index CSV.",
    )
    parser.add_argument(
        "--min-iod",
        type=float,
        default=None,
        help="Minimum inter-ocular distance filter (pixels).",
    )
    parser.add_argument(
        "--max-iod",
        type=float,
        default=None,
        help="Maximum inter-ocular distance filter (pixels).",
    )
    parser.add_argument(
        "--include-attrs",
        nargs="*",
        default=[],
        help="Attribute column names from list_attr_celeba.csv to include in index CSV.",
    )
    parser.add_argument(
        "--include-all-attrs",
        action="store_true",
        help="Include all 40 attribute columns in index CSV.",
    )
    parser.add_argument(
        "--include-bbox",
        action="store_true",
        help="Include bounding box columns x_1,y_1,width,height in index CSV.",
    )
    parser.add_argument(
        "--include-landmarks-raw",
        action="store_true",
        help="Include raw landmark coordinates in index CSV.",
    )
    parser.add_argument(
        "--stratify-by",
        nargs="*",
        default=[],
        help="Attributes to stratify by so both classes match distributions. "
             "Use attribute names and/or 'interocular' (requires --use-landmarks).",
    )
    parser.add_argument(
        "--iod-bins",
        type=int,
        default=DEFAULT_IOD_BINS,
        help="Number of quantile bins to discretize inter-ocular distance when stratifying by it.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    """Program entry point.

    Args:
        argv: List of CLI arguments (excluding program name).

    Returns:
        Process exit code where 0 indicates success.
    """
    args = parse_args(argv)
    attrs_csv, parts_csv, auto_images_root = validate_archive_dir(args.archive_dir)
    images_root = args.images_root or auto_images_root
    merged = load_and_merge(attrs_csv, parts_csv)

    print(f"Using attribute: {args.attribute}")
    print(f"Images root: {images_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Link mode: {args.link_mode}; dry-run={args.dry_run}; overwrite={args.overwrite}")

    cap_by_split: Dict[str, Optional[int]] = {
        "train": args.max_per_class_train if args.max_per_class_train is not None else args.max_per_class,
        "val": args.max_per_class_val if args.max_per_class_val is not None else args.max_per_class,
        "test": args.max_per_class_test if args.max_per_class_test is not None else args.max_per_class,
    }

    landmarks_lookup: Optional[Dict[str, float]] = None
    filtered_df = merged
    landmarks_df: Optional[pd.DataFrame] = None
    
    if args.use_landmarks or args.include_landmarks_raw:
        # Load once
        landmarks_df = load_landmarks(args.archive_dir)
        if args.use_landmarks:
            filtered_df, landmarks_lookup = merge_and_filter_by_landmarks(
                merged_df=merged,
                archive_dir=args.archive_dir,
                min_iod=args.min_iod,
                max_iod=args.max_iod,
            )
            print(f"After landmarks filter: {filtered_df.shape[0]} rows")

    # Build optional lookups
    attr_cols_all = [
        c for c in filtered_df.columns
        if c not in METADATA_COLUMNS
    ]
    attrs_to_include: Optional[List[str]] = None
    attrs_lookup: Optional[Dict[str, Dict[str, object]]] = None
    
    if args.include_all_attrs:
        attrs_to_include = attr_cols_all
        attrs_lookup = to_lookup(
            filtered_df[["image_id", *attrs_to_include]],
            attrs_to_include,
        )
    elif args.include_attrs:
        invalid = [a for a in args.include_attrs if a not in filtered_df.columns]
        if invalid:
            raise ValueError(f"Unknown attribute columns requested: {invalid}")
        attrs_to_include = list(args.include_attrs)
        attrs_lookup = to_lookup(
            filtered_df[["image_id", *attrs_to_include]],
            attrs_to_include,
        )

    bbox_lookup: Optional[Dict[str, Dict[str, object]]] = None
    if args.include_bbox:
        bdf = load_bboxes(args.archive_dir)
        bbox_lookup = to_lookup(
            bdf[["image_id", *BBOX_COLUMNS]],
            BBOX_COLUMNS,
        )

    landmarks_raw_lookup: Optional[Dict[str, Dict[str, object]]] = None
    if args.include_landmarks_raw:
        ldf = landmarks_df if landmarks_df is not None else load_landmarks(args.archive_dir)
        landmarks_raw_lookup = to_lookup(
            ldf[["image_id", *LANDMARK_ALL_COLUMNS]],
            LANDMARK_ALL_COLUMNS,
        )

    # Stratification columns
    stratify_by = list(args.stratify_by) if args.stratify_by else []
    iod_bins = args.iod_bins

    build_subset(
        merged=filtered_df,
        images_root=images_root,
        output_dir=args.output_dir,
        attribute=args.attribute,
        link_mode=args.link_mode,
        seed=args.seed,
        max_per_class_by_split=cap_by_split,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        strict_missing=args.strict_missing,
        fill_missing=(not args.no_fill_missing),
        landmarks_lookup=landmarks_lookup,
        attrs_to_include=attrs_to_include,
        attrs_lookup=attrs_lookup,
        bbox_lookup=bbox_lookup,
        landmarks_raw_lookup=landmarks_raw_lookup,
        stratify_by=stratify_by,
        iod_bins=iod_bins,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
