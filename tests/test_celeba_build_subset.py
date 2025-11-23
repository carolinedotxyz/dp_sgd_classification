import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.datasets.celeba.builder import build_subset


def _write_img(p: Path, color):
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((8, 8, 3), color, dtype=np.uint8)
    Image.fromarray(arr).convert("RGB").save(p)


@pytest.fixture()
def tiny_celeba_archive(tmp_path: Path):
    # Layout:
    # - list_attr_celeba.csv with Eyeglasses ∈ {1,-1}
    # - list_eval_partition.csv with partition ∈ {0,1,2}
    # - list_landmarks_align_celeba.csv for interocular calculation
    # - img_align_celeba/img_align_celeba/<images>
    archive = tmp_path / "archive"
    img_root = archive / "img_align_celeba" / "img_align_celeba"
    img_root.mkdir(parents=True, exist_ok=True)

    rows = []
    attr_rows = []
    lmk_rows = []

    # build 18 images: 12 train (6 pos/6 neg), 3 val (2 pos/1 neg), 3 test (1 pos/2 neg)
    # Create slight coordinate differences for interocular distance stratification
    def add_item(idx: int, part_id: int, label: int):
        image_id = f"img_{idx:03d}.jpg"
        color = (255, 0, 0) if label == 1 else (0, 255, 0)
        _write_img(img_root / image_id, color)
        rows.append({"image_id": image_id, "partition": part_id})
        attr_rows.append({"image_id": image_id, "Eyeglasses": label})
        lmk_rows.append({
            "image_id": image_id,
            "lefteye_x": 10 + idx % 3,
            "lefteye_y": 10,
            "righteye_x": 20 + (idx % 5),
            "righteye_y": 10,
        })

    # train: 12 samples
    for i in range(6):
        add_item(idx=i, part_id=0, label=1)
    for i in range(6, 12):
        add_item(idx=i, part_id=0, label=-1)
    # val: 3 samples (2 pos, 1 neg)
    add_item(idx=100, part_id=1, label=1)
    add_item(idx=101, part_id=1, label=1)
    add_item(idx=102, part_id=1, label=-1)
    # test: 3 samples (1 pos, 2 neg)
    add_item(idx=200, part_id=2, label=1)
    add_item(idx=201, part_id=2, label=-1)
    add_item(idx=202, part_id=2, label=-1)

    # Write CSVs
    attrs = pd.DataFrame(attr_rows)
    # ensure only needed columns + potential extra attr to ensure passthrough is robust
    attrs.to_csv(archive / "list_attr_celeba.csv", index=False)
    parts = pd.DataFrame(rows)
    parts.to_csv(archive / "list_eval_partition.csv", index=False)
    lmks = pd.DataFrame(lmk_rows)
    # add required columns for compute_interocular_distance
    lmks["lefteye_x"] = lmks["lefteye_x"].astype(float)
    lmks["lefteye_y"] = 10.0
    lmks["righteye_y"] = 10.0
    lmks.to_csv(archive / "list_landmarks_align_celeba.csv", index=False)

    return archive


def test_build_subset_balanced_and_deterministic(tmp_path: Path, tiny_celeba_archive: Path):
    # Prepare merged df similar to load_and_merge output
    attrs = pd.read_csv(tiny_celeba_archive / "list_attr_celeba.csv")
    parts = pd.read_csv(tiny_celeba_archive / "list_eval_partition.csv")
    merged = attrs.merge(parts, on="image_id", how="inner")
    merged["partition_name"] = merged["partition"].map({0: "train", 1: "val", 2: "test"})

    output_dir = tmp_path / "subset"

    # cap per class per split
    max_per_class_by_split = {"train": 5, "val": 1, "test": 1}

    # First run
    build_subset(
        merged=merged,
        images_root=str(tiny_celeba_archive / "img_align_celeba" / "img_align_celeba"),
        output_dir=str(output_dir),
        attribute="Eyeglasses",
        link_mode="symlink",
        seed=123,
        max_per_class_by_split=max_per_class_by_split,
        overwrite=True,
        dry_run=False,
        strict_missing=True,
        fill_missing=True,
        landmarks_lookup=None,
        attrs_to_include=None,
        attrs_lookup=None,
        bbox_lookup=None,
        landmarks_raw_lookup=None,
        stratify_by=[],
        iod_bins=None,
    )

    index_csv = output_dir / "subset_index_eyeglasses.csv"
    assert index_csv.exists()
    idx1 = pd.read_csv(index_csv)

    # Balanced per split
    for split, cap in [("train", 5), ("val", 1), ("test", 1)]:
        sub = idx1[idx1["partition_name"] == split]
        # Expect exactly 2*cap items
        assert len(sub) == 2 * cap
        # 50/50 class balance
        assert sub["label"].sum() == cap

    # Deterministic with same seed
    # Clean and rerun to compare
    os.remove(index_csv)
    build_subset(
        merged=merged,
        images_root=str(tiny_celeba_archive / "img_align_celeba" / "img_align_celeba"),
        output_dir=str(output_dir),
        attribute="Eyeglasses",
        link_mode="symlink",
        seed=123,
        max_per_class_by_split=max_per_class_by_split,
        overwrite=True,
        dry_run=False,
        strict_missing=True,
        fill_missing=True,
        landmarks_lookup=None,
        attrs_to_include=None,
        attrs_lookup=None,
        bbox_lookup=None,
        landmarks_raw_lookup=None,
        stratify_by=[],
        iod_bins=None,
    )
    idx2 = pd.read_csv(index_csv)
    pd.testing.assert_frame_equal(idx1.sort_values(["image_id", "partition_name"]).reset_index(drop=True),
                                  idx2.sort_values(["image_id", "partition_name"]).reset_index(drop=True))


def test_build_subset_stratified_by_interocular(tmp_path: Path, tiny_celeba_archive: Path):
    attrs = pd.read_csv(tiny_celeba_archive / "list_attr_celeba.csv")
    parts = pd.read_csv(tiny_celeba_archive / "list_eval_partition.csv")
    merged = attrs.merge(parts, on="image_id", how="inner")
    merged["partition_name"] = merged["partition"].map({0: "train", 1: "val", 2: "test"})

    # attach interocular by simulating what merge_and_filter_by_landmarks would do
    # create a simple spread of interocular distances
    lmks = pd.read_csv(tiny_celeba_archive / "list_landmarks_align_celeba.csv")
    # add missing required columns that compute_interocular_distance expects
    lmks["lefteye_y"] = 10.0
    lmks["righteye_y"] = 10.0
    lmks["interocular"] = (lmks["righteye_x"] - lmks["lefteye_x"]).abs()
    merged = merged.merge(lmks[["image_id", "interocular"]], on="image_id", how="left")

    output_dir = tmp_path / "subset_strat"
    max_per_class_by_split = {"train": 4, "val": 1, "test": 1}

    build_subset(
        merged=merged,
        images_root=str(tiny_celeba_archive / "img_align_celeba" / "img_align_celeba"),
        output_dir=str(output_dir),
        attribute="Eyeglasses",
        link_mode="symlink",
        seed=321,
        max_per_class_by_split=max_per_class_by_split,
        overwrite=True,
        dry_run=False,
        strict_missing=True,
        fill_missing=True,
        landmarks_lookup=None,
        attrs_to_include=None,
        attrs_lookup=None,
        bbox_lookup=None,
        landmarks_raw_lookup=None,
        stratify_by=["interocular"],
        iod_bins=3,
    )

    idx = pd.read_csv(output_dir / "subset_index_eyeglasses.csv")

    # Check balance per split
    for split, cap in [("train", 4), ("val", 1), ("test", 1)]:
        sub = idx[idx["partition_name"] == split]
        assert len(sub) == 2 * cap
        assert sub["label"].sum() == cap


