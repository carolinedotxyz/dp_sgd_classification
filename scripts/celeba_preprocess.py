#!/usr/bin/env python3
"""Preprocess a CelebA subset: crop/resize images and compute simple stats.

This script reads a previously created balanced CelebA subset (e.g., produced by
``celeba_build_subset.py``), optionally discovers metadata from the subset index,
processes images with an optional center square crop and resizing, and writes the
results to a new processed directory mirroring the split/class structure.

Optionally, it computes per-channel mean and standard deviation over the TRAIN
split and writes ``stats.json`` alongside a processing summary report.

Typical usage:
  python scripts/celeba_preprocess.py --subset-root <subset_dir> \
    --out-root <out_dir> --size 64 [--center-crop] [--normalize-01] \
    [--compute-stats] [--stats-out <path>]

Outputs:
- Processed images under ``<out-root>/<split>/<class_name>/``
- ``<out-root>/processed_index.csv`` with relative source/dest paths
- Optional ``stats.json`` and ``processing_summary.json`` under ``<out-root>/stats``
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib
import csv

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


# Constants
DEFAULT_IMAGE_SIZE = 64
DEFAULT_IMAGE_QUALITY = 95
PIXEL_MAX_VALUE = 255.0
STD_EPSILON = 1e-8
JSON_INDENT = 2
CHUNK_SIZE_MB = 1024 * 1024  # 1MB chunks for hashing

# Index CSV patterns
INDEX_CSV_PREFIX = "subset_index_"
INDEX_CSV_SUFFIX = ".csv"

# Default values
DEFAULT_SPLIT = "train"
DEFAULT_CLASS = "eyeglasses"

# Image file extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Partition and class names
PARTITION_NAMES = ("train", "val", "test")
CLASS_NAMES = ("eyeglasses", "no_eyeglasses")


@dataclass
class ImageRecord:
    """Metadata for a single image before/after processing.

    Attributes:
        image_id: File name (e.g., ``000001.jpg``).
        split: Split name: ``train``, ``val``, or ``test``.
        class_name: Class folder name (e.g., ``eyeglasses``).
        src_path: Source image path.
        dst_path: Destination image path (filled after processing).
        label: Optional binary label if available from subset index.
        interocular: Optional inter-ocular distance if available.
    """
    image_id: str
    split: str
    class_name: str
    src_path: str
    dst_path: str
    label: Optional[int] = None
    interocular: Optional[float] = None


def discover_subset(root: str) -> List[ImageRecord]:
    """Discover images and carry over metadata from a subset index if present.

    Prefers a ``subset_index_*.csv`` file to populate split, class, label, and
    inter-ocular distance; falls back to walking the directory tree.

    Args:
        root: Subset root containing ``train/val/test``.

    Returns:
        A list of ``ImageRecord`` entries with source paths populated.

    Raises:
        FileNotFoundError: If no images are found under ``root``.
    """
    # Attempt to locate an index CSV first
    index_csv: Optional[str] = None
    for name in os.listdir(root):
        if name.startswith(INDEX_CSV_PREFIX) and name.endswith(INDEX_CSV_SUFFIX):
            index_csv = os.path.join(root, name)
            break

    records: List[ImageRecord] = []
    if index_csv and os.path.isfile(index_csv):
        try:
            import pandas as pd  # local import to keep dependency optional if not used
            df = pd.read_csv(index_csv)
            # Normalize expected columns
            has_label = "label" in df.columns
            has_split = "partition_name" in df.columns or "split" in df.columns
            has_class = "class_name" in df.columns
            for _, r in df.iterrows():
                image_id = str(r.get("image_id", "")).strip()
                if not image_id:
                    continue
                split = str(r.get("partition_name", r.get("split", ""))) or DEFAULT_SPLIT
                class_name = str(r.get("class_name", DEFAULT_CLASS))
                # Reconstruct src_path from directory structure to be robust
                src_path = os.path.join(root, split, class_name, image_id)
                if not os.path.isfile(src_path):
                    # Fallback to dest_path from CSV if present
                    dp = r.get("dest_path")
                    if isinstance(dp, str) and dp:
                        src_path = dp if os.path.isabs(dp) else os.path.join(root, dp)
                label_val: Optional[int] = int(r["label"]) if has_label and not pd.isna(r["label"]) else None
                interocular_val: Optional[float] = None
                if "interocular" in df.columns and not pd.isna(r.get("interocular")):
                    try:
                        interocular_val = float(r.get("interocular"))
                    except Exception:
                        interocular_val = None
                records.append(
                    ImageRecord(
                        image_id=image_id,
                        split=split,
                        class_name=class_name,
                        src_path=src_path,
                        dst_path="",
                        label=label_val,
                        interocular=interocular_val,
                    )
                )
        except Exception:
            # Fall back to directory walk if CSV cannot be read
            records = []

    if not records:
        for split in PARTITION_NAMES:
            for class_name in CLASS_NAMES:
                d = os.path.join(root, split, class_name)
                if not os.path.isdir(d):
                    continue
                for fname in os.listdir(d):
                    if not fname.lower().endswith(IMAGE_EXTENSIONS):
                        continue
                    src_path = os.path.join(d, fname)
                    records.append(
                        ImageRecord(
                            image_id=fname,
                            split=split,
                            class_name=class_name,
                            src_path=src_path,
                            dst_path="",
                        )
                    )
    if not records:
        raise FileNotFoundError(f"No images found under subset root: {root}")
    return records


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """Return the centered square crop of an image.

    Args:
        img: Input PIL image.

    Returns:
        A new PIL image cropped to a square centered in the original image.
    """
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def resize_image(img: Image.Image, size: int) -> Image.Image:
    """Resize an image to ``size x size`` using high-quality downsampling.

    Args:
        img: Input PIL image.
        size: Target side length in pixels.

    Returns:
        The resized PIL image.
    """
    return img.resize((size, size), Image.Resampling.LANCZOS)


def ensure_dir(path: str) -> None:
    """Create directory ``path`` if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def process_image(
    rec: ImageRecord,
    out_root: str,
    size: int,
    do_center_crop: bool,
    normalize_01: bool,
    mean: Optional[Tuple[float, float, float]],
    std: Optional[Tuple[float, float, float]],
    return_array: bool = True,
) -> Tuple[ImageRecord, Optional[np.ndarray]]:
    """Process a single image and optionally return an array for stats.

    Applies an optional center crop and resize, writes the processed image to the
    output tree, and returns the updated record and an optional pixel array
    suitable for mean/std computation.

    Args:
        rec: The input image record.
        out_root: Root directory for processed images.
        size: Output side length in pixels.
        do_center_crop: If ``True``, apply a centered square crop before resize.
        normalize_01: If ``True``, scale pixel values to [0, 1] in the returned array.
        mean: Unused placeholder for future normalization on save.
        std: Unused placeholder for future normalization on save.

    Returns:
        Tuple ``(record_out, per_image_array_or_none)`` where the array is ``HxWxC``
        float32 and present only if ``normalize_01`` or stats are requested.
    """
    with Image.open(rec.src_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if do_center_crop:
            img = center_crop_to_square(img)
        img = resize_image(img, size)

        dst_dir = os.path.join(out_root, rec.split, rec.class_name)
        ensure_dir(dst_dir)
        dst_path = os.path.join(dst_dir, rec.image_id)
        rec_out = ImageRecord(
            image_id=rec.image_id,
            split=rec.split,
            class_name=rec.class_name,
            src_path=rec.src_path,
            dst_path=dst_path,
            label=rec.label,
            interocular=rec.interocular,
        )

        # Save processed image to disk first
        img.save(dst_path, quality=DEFAULT_IMAGE_QUALITY)

        # Return pixel stats array only if requested (to avoid overhead)
        np_img: Optional[np.ndarray] = None
        if return_array:
            np_img = np.asarray(img, dtype=np.float32)
            if normalize_01:
                np_img = np_img / PIXEL_MAX_VALUE
        return rec_out, np_img


def compute_mean_std(per_image_arrays: List[np.ndarray]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute per-channel mean and std from a list of image arrays.

    Expects arrays shaped ``HxWxC`` and scaled to [0, 1] if appropriate.

    Args:
        per_image_arrays: List of per-image arrays (``HxWxC`` float32).

    Returns:
        Tuple of ``(mean_rgb, std_rgb)`` where each is a 3-tuple of floats.
    """
    stacked = np.stack(per_image_arrays, axis=0)  # N x H x W x C
    mean = stacked.mean(axis=(0, 1, 2))
    std = stacked.std(axis=(0, 1, 2)) + STD_EPSILON
    return (float(mean[0]), float(mean[1]), float(mean[2])), (float(std[0]), float(std[1]), float(std[2]))


def write_index(records: List[ImageRecord], out_root: str) -> None:
    """Write an index CSV mirroring subset columns with relative paths.

    Args:
        records: Processed image records to index.
        out_root: Root directory of processed images; paths are written relative to it.
    """
    out_csv = os.path.join(out_root, "processed_index.csv")
    header_cols = [
        "image_id",
        "label",
        "class_name",
        "partition_name",
        "source_path",
        "dest_path",
        "interocular",
    ]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header_cols) + "\n")
        for r in records:
            # Relative paths w.r.t processed root
            rel_src = os.path.relpath(r.src_path, out_root)
            rel_dst = os.path.relpath(r.dst_path, out_root)
            row = [
                r.image_id,
                "" if r.label is None else str(int(r.label)),
                r.class_name,
                r.split,
                rel_src,
                rel_dst,
                "" if r.interocular is None else f"{float(r.interocular):.6f}",
            ]
            f.write(",".join(row) + "\n")


def write_stats(stats_path: str, stats: Dict[str, object]) -> None:
    """Write a JSON stats file, creating parent directories if needed.

    Args:
        stats_path: Destination file path.
        stats: Serializable dictionary of statistics.
    """
    ensure_dir(os.path.dirname(stats_path))
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=JSON_INDENT)


def summarize(records: List[ImageRecord]) -> Dict[str, object]:
    """Summarize processed records by split and class.

    Args:
        records: Collection of processed records.

    Returns:
        A dictionary with total counts and nested split/class counts.
    """
    by_split: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    by_split_class: Dict[str, Dict[str, int]] = {}
    for r in records:
        by_split[r.split] = by_split.get(r.split, 0) + 1
        by_class[r.class_name] = by_class.get(r.class_name, 0) + 1
        if r.split not in by_split_class:
            by_split_class[r.split] = {}
        by_split_class[r.split][r.class_name] = by_split_class[r.split].get(r.class_name, 0) + 1
    return {
        "num_images": len(records),
        "counts_by_split": by_split,
        "counts_by_class": by_class,
        "counts_by_split_class": by_split_class,
    }


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of CLI arguments (excluding the program name).

    Returns:
        Parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(description="Preprocess CelebA subset: resize/crop/normalize and compute stats.")
    p.add_argument("--subset-root", type=str, required=True, help="Path to balanced subset root (with train/val/test).")
    p.add_argument("--out-root", type=str, default=None, help="Output root for processed images (defaults to <subset-root>_processed).")
    p.add_argument("--size", type=int, default=DEFAULT_IMAGE_SIZE, help="Output image side length (square).")
    p.add_argument("--center-crop", action="store_true", help="Apply center square crop before resizing.")
    p.add_argument("--normalize-01", action="store_true", help="Scale pixels to [0,1] when computing mean/std.")
    p.add_argument("--compute-stats", action="store_true", help="Compute mean/std on TRAIN split and save stats.json.")
    p.add_argument("--stats-out", type=str, default=None, help="Optional path to stats.json (defaults under out-root).")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    """Program entry point.

    Args:
        argv: List of CLI arguments (excluding the program name).

    Returns:
        Process exit code where 0 indicates success.
    """
    args = parse_args(argv)
    subset_root = os.path.abspath(args.subset_root)
    out_root = os.path.abspath(args.out_root or (subset_root.rstrip("/") + "_processed"))

    records = discover_subset(subset_root)

    processed_records: List[ImageRecord] = []
    train_arrays: List[np.ndarray] = []
    skipped: int = 0
    skipped_rows: List[Dict[str, str]] = []
    manifest_rows: List[Dict[str, object]] = []

    for rec in tqdm(records, desc="Processing images"):
        try:
            rec_out, arr = process_image(
                rec,
                out_root=out_root,
                size=args.size,
                do_center_crop=bool(args.center_crop),
                normalize_01=bool(args.normalize_01),
                mean=None,
                std=None,
                return_array=bool(args.compute_stats and rec.split == "train"),
            )
            processed_records.append(rec_out)
            # Hash and record manifest entry
            try:
                h = hashlib.sha256()
                with open(rec_out.dst_path, "rb") as _f:
                    for chunk in iter(lambda: _f.read(CHUNK_SIZE_MB), b""):
                        h.update(chunk)
                size_bytes = os.path.getsize(rec_out.dst_path)
                manifest_rows.append({
                    "image_id": rec_out.image_id,
                    "partition_name": rec_out.split,
                    "class_name": rec_out.class_name,
                    "source_path": os.path.relpath(rec_out.src_path, out_root) if not os.path.isabs(rec_out.src_path) else rec_out.src_path,
                    "dest_path": os.path.relpath(rec_out.dst_path, out_root),
                    "sha256": h.hexdigest(),
                    "size_bytes": int(size_bytes),
                })
            except Exception:
                pass
            if args.compute_stats and rec.split == "train" and arr is not None:
                train_arrays.append(arr)
        except Exception as e:
            skipped += 1
            skipped_rows.append({
                "image_id": rec.image_id,
                "split": rec.split,
                "class_name": rec.class_name,
                "src_path": rec.src_path,
                "reason": str(e),
            })

    write_index(processed_records, out_root)

    if args.compute_stats and train_arrays:
        mean, std = compute_mean_std(train_arrays)
        stats = {
            "size": args.size,
            "normalize_01": bool(args.normalize_01),
            "train_mean": mean,
            "train_std": std,
        }
        stats_path = args.stats_out or os.path.join(out_root, "stats", "stats.json")
        write_stats(stats_path, stats)

    # Always write a processing summary
    summary = summarize(processed_records)
    summary["skipped"] = skipped
    summary_path = os.path.join(out_root, "stats", "processing_summary.json")
    write_stats(summary_path, summary)

    # If any failures, write a CSV log for inspection
    if skipped_rows:
        ensure_dir(os.path.join(out_root, "stats"))
        skipped_csv = os.path.join(out_root, "stats", "skipped_failed_images.csv")
        try:
            import csv as _csv
            with open(skipped_csv, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=["image_id", "split", "class_name", "src_path", "reason"])
                writer.writeheader()
                writer.writerows(skipped_rows)
        except Exception:
            pass

    # Write manifest CSV with hashes and sizes
    if manifest_rows:
        stats_dir = os.path.join(out_root, "stats")
        ensure_dir(stats_dir)
        mf = os.path.join(stats_dir, "manifest.csv")
        try:
            with open(mf, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image_id","partition_name","class_name","source_path","dest_path","sha256","size_bytes"
                ])
                writer.writeheader()
                writer.writerows(manifest_rows)
        except Exception:
            pass

    # Consolidated data ledger JSON
    try:
        summary_rel = os.path.join("stats", "processing_summary.json")
        ledger = {
            "size": int(args.size),
            "normalize_01": bool(args.normalize_01),
            "stats_path": os.path.join("stats", "stats.json"),
            "manifest_path": os.path.join("stats", "manifest.csv"),
            "processing_summary_path": summary_rel,
            "counts": summary,
            "randomness_policy": {
                "worker_seeding": "make_worker_init_fn(base_seed)",
                "global_seed": "cfg.random_seed or environment",
                "note": "Per-sample transform randomness should derive from seeded workers"
            },
            "transform_policy": {
                "center_crop": bool(args.center_crop),
                "resize": {"side": int(args.size), "method": "LANCZOS"},
            },
        }
        write_stats(os.path.join(out_root, "stats", "data_ledger.json"), ledger)
    except Exception:
        pass

    # Concise console report
    print(f"Processed images written to: {out_root}")
    print(f"Total processed: {summary['num_images']} (skipped: {skipped})")
    print(f"By split: {summary['counts_by_split']}")
    print(f"By class: {summary['counts_by_class']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


