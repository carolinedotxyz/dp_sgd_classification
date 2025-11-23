#!/usr/bin/env python3
"""Analyze face centering for a CelebA subset using landmark metadata.

This script loads a previously built CelebA subset (e.g., from
``celeba_build_subset.py``), locates images and corresponding facial landmarks,
computes each face's center relative to the image center, and writes per-image
metrics. It can also render summary plots and a grid of the most off-center
examples.

Typical usage:
  python scripts/celeba_centering.py --subset-root <subset_dir> \
    --archive-dir <celeba_archive> [--out-dir <out>] [--k-outliers 32]

Outputs:
- Metrics CSV at ``<out-dir>/centering_metrics.csv``
- Plots and an outliers grid image under ``<out-dir>``
"""

import argparse
import os
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# Constants
CSV_LANDMARKS = "list_landmarks_align_celeba.csv"
DEFAULT_ARCHIVE_DIR = "./data/celeba/archive"
DEFAULT_OUT_DIR = "centering_analysis"
DEFAULT_K_OUTLIERS = 32
DEFAULT_GRID_COLS = 8

# Landmark columns
LANDMARK_REQUIRED_COLS = [
    "image_id", "lefteye_x", "lefteye_y", "righteye_x", "righteye_y",
    "nose_x", "nose_y"
]
LANDMARK_COLS_FOR_LOOKUP = [
    "lefteye_x", "lefteye_y", "righteye_x", "righteye_y", "nose_x", "nose_y"
]

# Face center calculation weights
EYE_WEIGHT = 0.75
NOSE_WEIGHT = 0.25

# Image file extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Plot settings
PLOT_DPI = 200
PLOT_COLOR_PRIMARY = "#4C78A8"
HIST_BINS = 40
HEXBIN_GRIDSIZE = 40

# Default CelebA image size (aligned)
DEFAULT_CELEBA_WIDTH = 178
DEFAULT_CELEBA_HEIGHT = 218

# Outlier grid settings
OUTLIER_MARKER_RADIUS = 4
OUTLIER_TEXT_HEIGHT = 18
OUTLIER_TEXT_PADDING = 4
OUTLIER_MARKER_COLOR = (255, 50, 50)
OUTLIER_BG_COLOR = (0, 0, 0, 128)
OUTLIER_TEXT_COLOR = (255, 255, 255)
GRID_BG_COLOR = (30, 30, 30)
GRID_SAVE_QUALITY = 95

# Partition names
PARTITION_NAMES = ("train", "val", "test")
CLASS_NAMES = ("eyeglasses", "no_eyeglasses")
DEFAULT_SPLIT = "train"
DEFAULT_CLASS = "eyeglasses"

# Index CSV patterns
INDEX_CSV_PREFIX = "subset_index_"
INDEX_CSV_SUFFIX = ".csv"


@dataclass
class Item:
    """A single sample in the subset.

    Attributes:
        image_id: File name (e.g., ``000001.jpg``).
        split: Split name: ``train``, ``val``, or ``test``.
        class_name: Class folder name (e.g., ``eyeglasses``).
        path: Absolute or subset-root-relative image path.
    """
    image_id: str
    split: str
    class_name: str
    path: str


def ensure_dir(path: str) -> None:
    """Create directory ``path`` if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_subset_items(subset_root: str) -> List[Item]:
    """Load subset items from an index CSV or by walking class folders.

    Prefers a ``subset_index_*.csv`` file if present; otherwise, falls back to
    discovering images under ``train/val/test`` and class subfolders.

    Args:
        subset_root: Root directory of the subset.

    Returns:
        List of ``Item`` entries with split, class, and paths populated.

    Raises:
        FileNotFoundError: If no images can be found under ``subset_root``.
    """
    idx_csv = None
    for name in os.listdir(subset_root):
        if name.startswith(INDEX_CSV_PREFIX) and name.endswith(INDEX_CSV_SUFFIX):
            idx_csv = os.path.join(subset_root, name)
            break
    items: List[Item] = []
    if idx_csv and os.path.isfile(idx_csv):
        df = pd.read_csv(idx_csv)
        # Prefer dest_path if present, else reconstruct
        for _, r in df.iterrows():
            img_id = str(r["image_id"]) if "image_id" in r else os.path.basename(str(r["dest_path"]))
            split = str(r.get("partition_name", r.get("split", DEFAULT_SPLIT)))
            class_name = str(r.get("class_name", DEFAULT_CLASS))
            dest_path = r.get("dest_path")
            if isinstance(dest_path, str) and dest_path:
                # dest_path may be relative to subset_root
                if not os.path.isabs(dest_path):
                    path = os.path.join(subset_root, dest_path)
                else:
                    path = dest_path
            else:
                path = os.path.join(subset_root, split, class_name, img_id)
            if os.path.isfile(path):
                items.append(Item(image_id=img_id, split=split, class_name=class_name, path=path))
        if items:
            return items
    # Fallback: walk directories
    for split in PARTITION_NAMES:
        for cls in CLASS_NAMES:
            d = os.path.join(subset_root, split, cls)
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                items.append(Item(image_id=fname, split=split, class_name=cls, path=os.path.join(d, fname)))
    if not items:
        raise FileNotFoundError(f"No images found under {subset_root}")
    return items


def load_landmarks(archive_dir: str) -> pd.DataFrame:
    """Load aligned landmarks CSV and validate required columns.

    Args:
        archive_dir: Path to CelebA archive containing ``list_landmarks_align_celeba.csv``.

    Returns:
        DataFrame with numeric eye and nose coordinates.

    Raises:
        FileNotFoundError: If the CSV is missing.
        ValueError: If required columns are missing.
    """
    path = os.path.join(archive_dir, CSV_LANDMARKS)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Landmarks CSV not found: {path}")
    df = pd.read_csv(path)
    for c in LANDMARK_REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column in landmarks CSV: {c}")
    for c in LANDMARK_REQUIRED_COLS[1:]:  # Skip image_id
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def face_center_from_landmarks(row: pd.Series) -> Tuple[float, float]:
    """Estimate face center using eye midpoint nudged toward the nose.

    Args:
        row: Series containing eye and nose coordinates.

    Returns:
        Tuple ``(cx, cy)`` representing the estimated face center in pixels.
    """
    lx, ly = float(row["lefteye_x"]), float(row["lefteye_y"])
    rx, ry = float(row["righteye_x"]), float(row["righteye_y"])
    nx, ny = float(row["nose_x"]), float(row["nose_y"])
    # Use midpoint of eyes, lightly nudged towards nose
    ex, ey = (lx + rx) * 0.5, (ly + ry) * 0.5
    cx = EYE_WEIGHT * ex + NOSE_WEIGHT * nx
    cy = EYE_WEIGHT * ey + NOSE_WEIGHT * ny
    return cx, cy


def compute_offsets(items: List[Item], lm_df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized center offsets and distances for a set of images.

    For each image with available landmarks, computes ``(dx, dy)`` normalized to
    [-0.5, 0.5] where (0, 0) is the image center, along with the Euclidean
    distance ``radius``.

    Args:
        items: List of subset items with paths.
        lm_df: DataFrame of landmarks for all images.

    Returns:
        DataFrame with per-image metrics and metadata.
    """
    lm_lookup = lm_df.set_index("image_id")[LANDMARK_COLS_FOR_LOOKUP].to_dict("index")

    rows: List[Dict[str, object]] = []
    missing = 0
    for it in items:
        lm = lm_lookup.get(it.image_id)
        if lm is None:
            missing += 1
            continue
        try:
            with Image.open(it.path) as img:
                w, h = img.size
        except Exception:
            continue
        cx, cy = face_center_from_landmarks(pd.Series(lm))
        dx = (cx / max(1.0, (w - 1))) - 0.5
        dy = (cy / max(1.0, (h - 1))) - 0.5
        r = math.sqrt(dx * dx + dy * dy)
        rows.append(
            {
                "image_id": it.image_id,
                "split": it.split,
                "class_name": it.class_name,
                "width": w,
                "height": h,
                "cx": cx,
                "cy": cy,
                "dx": dx,
                "dy": dy,
                "radius": r,
                "path": it.path,
            }
        )
    if missing:
        print(f"Warning: {missing} images missing landmarks; skipped.")
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Render summary plots for centering metrics.

    Outputs a histogram of distances from center and a hexbin plot of
    ``dx`` vs. ``dy``. If Matplotlib is unavailable, plotting is skipped.

    Args:
        df: Metrics DataFrame from ``compute_offsets``.
        out_dir: Directory to write plot images.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"Matplotlib not available, skipping plots: {e}")
        return
    ensure_dir(out_dir)

    # Histogram of radius
    plt.figure(figsize=(6, 4))
    plt.hist(df["radius"], bins=HIST_BINS, color=PLOT_COLOR_PRIMARY)
    plt.xlabel("distance from center (normalized)")
    plt.ylabel("count")
    plt.title("Face center distance distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "center_distance_hist.png"), dpi=PLOT_DPI)
    plt.close()

    # 2D scatter/hex of dx, dy
    plt.figure(figsize=(5, 5))
    hb = plt.hexbin(df["dx"], df["dy"], gridsize=HEXBIN_GRIDSIZE, cmap="viridis")
    plt.axvline(0.0, color="white", linewidth=1, alpha=0.6)
    plt.axhline(0.0, color="white", linewidth=1, alpha=0.6)
    plt.xlabel("dx (centered)")
    plt.ylabel("dy (centered)")
    plt.title("Face center offset (dx, dy)")
    cb = plt.colorbar(hb)
    cb.set_label("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "center_offset_hex.png"), dpi=PLOT_DPI)
    plt.close()


def draw_outliers_grid(df: pd.DataFrame, out_png: str, k: int = 32, cols: int = 8) -> None:
    """Draw a grid of the top-K most off-center faces and save to PNG.

    Args:
        df: Metrics DataFrame including ``radius`` and ``path`` columns.
        out_png: Output PNG file path.
        k: Number of most off-center images to include.
        cols: Number of grid columns.
    """
    ensure_dir(os.path.dirname(out_png))
    sel = df.sort_values("radius", ascending=False).head(k)
    if sel.empty:
        return
    # Assume aligned CelebA (178x218). Fallback from first image size
    try:
        with Image.open(sel.iloc[0]["path"]) as im0:
            tile_w, tile_h = im0.size
    except Exception:
        tile_w, tile_h = DEFAULT_CELEBA_WIDTH, DEFAULT_CELEBA_HEIGHT
    rows = int(math.ceil(len(sel) / cols))
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), GRID_BG_COLOR)

    for i, (_, row) in enumerate(sel.iterrows()):
        try:
            with Image.open(row["path"]) as im:
                im = im.convert("RGB")
                im = im.resize((tile_w, tile_h))
                draw = ImageDraw.Draw(im)
                cx, cy = float(row["cx"]), float(row["cy"])
                # scale cx,cy if resized from original size
                # We assume landmarks refer to original size; after resizing to (tile_w,tile_h), scale by factors
                # But we used the same image size as tile; so cx,cy are already in that space for typical aligned images
                r = OUTLIER_MARKER_RADIUS
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=OUTLIER_MARKER_COLOR)
                # annotate radius
                txt = f"{row['split'][:1]}/{row['class_name'][:2]} r={row['radius']:.3f}"
                draw.rectangle((0, 0, tile_w, OUTLIER_TEXT_HEIGHT), fill=OUTLIER_BG_COLOR)
                try:
                    draw.text((OUTLIER_TEXT_PADDING, 2), txt, fill=OUTLIER_TEXT_COLOR)
                except Exception:
                    pass
                r_idx = i // cols
                c_idx = i % cols
                grid.paste(im, (c_idx * tile_w, r_idx * tile_h))
        except Exception:
            continue
    grid.save(out_png, quality=GRID_SAVE_QUALITY)


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of CLI arguments (excluding the program name).

    Returns:
        Parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(description="Analyze face centering for a CelebA subset using landmarks.")
    p.add_argument("--subset-root", type=str, required=True, help="Path to subset root (train/val/test).")
    p.add_argument("--archive-dir", type=str, default=DEFAULT_ARCHIVE_DIR, help="Path to CelebA archive (for landmarks CSV).")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for analysis artifacts.")
    p.add_argument("--k-outliers", type=int, default=DEFAULT_K_OUTLIERS, help="Top-K most off-center examples to render.")
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
    out_dir = os.path.abspath(args.out_dir or os.path.join(subset_root, DEFAULT_OUT_DIR))

    items = load_subset_items(subset_root)
    lm_df = load_landmarks(args.archive_dir)
    df = compute_offsets(items, lm_df)

    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "centering_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(df)} rows)")

    save_plots(df, out_dir)
    draw_outliers_grid(
        df,
        os.path.join(out_dir, "outliers_grid.png"),
        k=int(args.k_outliers),
        cols=DEFAULT_GRID_COLS,
    )
    print(f"Artifacts in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

