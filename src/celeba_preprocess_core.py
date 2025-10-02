"""Core preprocessing for CelebA subsets (resize/crop and stats).

Programmatic API that mirrors scripts/celeba_preprocess.py so notebooks and
other modules can call directly without a CLI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class ImageRecord:
    image_id: str
    split: str
    class_name: str
    src_path: str
    dst_path: str
    label: Optional[int] = None
    interocular: Optional[float] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def resize_image(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BICUBIC)


def write_index(records: List[ImageRecord], out_root: str) -> None:
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


def compute_mean_std(per_image_arrays: List[np.ndarray]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    stacked = np.stack(per_image_arrays, axis=0)
    mean = stacked.mean(axis=(0, 1, 2))
    std = stacked.std(axis=(0, 1, 2)) + 1e-8
    return (float(mean[0]), float(mean[1]), float(mean[2])), (float(std[0]), float(std[1]), float(std[2]))


def write_json(path: str, data: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def preprocess_subset(
    subset_root: str,
    out_root: str,
    size: int,
    center_crop: bool,
    normalize_01: bool,
    compute_stats: bool,
) -> Dict[str, object]:
    # Discover items using src.celeba_index
    from src.celeba_index import load_subset_index

    df = load_subset_index(subset_root)
    records: List[ImageRecord] = []
    arrays: List[np.ndarray] = []

    for _, r in df.iterrows():
        image_id = str(r["image_id"])  # type: ignore[index]
        split = str(r["partition_name"])  # type: ignore[index]
        class_name = str(r.get("class_name", "eyeglasses"))
        src_path = str(r["source_path"])  # type: ignore[index]
        dst_dir = os.path.join(out_root, split, class_name)
        ensure_dir(dst_dir)
        dst_path = os.path.join(dst_dir, image_id)

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                if center_crop:
                    img = center_square_crop(img)
                img = resize_image(img, size)
                img.save(dst_path, quality=95)
                if compute_stats:
                    np_img = np.asarray(img, dtype=np.float32)
                    if normalize_01:
                        np_img = np_img / 255.0
                    arrays.append(np_img)
        except Exception:
            continue

        records.append(
            ImageRecord(
                image_id=image_id,
                split=split,
                class_name=class_name,
                src_path=src_path,
                dst_path=dst_path,
            )
        )

    write_index(records, out_root)

    stats: Dict[str, object] = {}
    if compute_stats and arrays:
        mean, std = compute_mean_std(arrays)
        stats = {"size": size, "normalize_01": bool(normalize_01), "train_mean": mean, "train_std": std}
        write_json(os.path.join(out_root, "stats", "stats.json"), stats)

    summary = {
        "num_images": len(records),
    }
    write_json(os.path.join(out_root, "stats", "processing_summary.json"), summary)
    return {"summary": summary, "stats": stats}


