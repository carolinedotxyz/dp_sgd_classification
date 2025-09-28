import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.celeba_preprocess import (
    ImageRecord,
    discover_subset,
    center_crop_to_square,
    resize_image,
    process_image,
    compute_mean_std,
    write_index,
    write_stats,
    summarize,
    parse_args,
    main,
)


def _write_rgb(path: Path, color=(200, 100, 50), size=(20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    Image.fromarray(arr).save(path)


@pytest.fixture()
def tiny_subset(tmp_path: Path) -> Path:
    root = tmp_path / "subset"
    # Directory structure
    for split in ("train", "val", "test"):
        for cls in ("eyeglasses", "no_eyeglasses"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)

    _write_rgb(root / "train" / "eyeglasses" / "a.jpg")
    _write_rgb(root / "train" / "no_eyeglasses" / "b.jpg")
    _write_rgb(root / "val" / "eyeglasses" / "c.jpg")

    # Also create a subset index to test fast path and metadata passthrough
    idx = pd.DataFrame([
        {"image_id": "a.jpg", "partition_name": "train", "class_name": "eyeglasses", "label": 1, "interocular": 12.5, "dest_path": "train/eyeglasses/a.jpg"},
        {"image_id": "b.jpg", "partition_name": "train", "class_name": "no_eyeglasses", "label": -1, "dest_path": "train/no_eyeglasses/b.jpg"},
        {"image_id": "c.jpg", "partition_name": "val", "class_name": "eyeglasses", "label": 1, "interocular": 10.0, "dest_path": "val/eyeglasses/c.jpg"},
    ])
    idx.to_csv(root / "subset_index_eyeglasses.csv", index=False)
    return root


def test_discover_subset_reads_index_and_passthrough(tiny_subset: Path):
    recs = discover_subset(str(tiny_subset))
    assert len(recs) == 3
    a = next(r for r in recs if r.image_id == "a.jpg")
    assert a.split == "train" and a.class_name == "eyeglasses"
    assert a.label == 1
    assert a.interocular == pytest.approx(12.5)
    # src_path resolved under root
    assert a.src_path.endswith("train/eyeglasses/a.jpg")


def test_discover_subset_walk_fallback(tmp_path: Path):
    root = tmp_path / "subset"
    _write_rgb(root / "test" / "eyeglasses" / "x.jpg")
    recs = discover_subset(str(root))
    assert len(recs) == 1
    assert recs[0].image_id == "x.jpg"


def test_center_crop_and_resize(tmp_path: Path):
    p = tmp_path / "img.jpg"
    _write_rgb(p, size=(30, 20))  # width > height
    with Image.open(p) as im:
        cropped = center_crop_to_square(im)
        assert cropped.size[0] == cropped.size[1] == 20
        resized = resize_image(cropped, 16)
        assert resized.size == (16, 16)


def test_process_image_and_np_array(tmp_path: Path):
    src = tmp_path / "subset" / "train" / "eyeglasses" / "a.jpg"
    _write_rgb(src, size=(20, 20))
    rec = ImageRecord(
        image_id="a.jpg",
        split="train",
        class_name="eyeglasses",
        src_path=str(src),
        dst_path="",
        label=1,
        interocular=10.0,
    )
    out_root = tmp_path / "processed"
    rec_out, arr = process_image(
        rec,
        out_root=str(out_root),
        size=16,
        do_center_crop=True,
        normalize_01=True,
        mean=None,
        std=None,
    )
    assert rec_out.dst_path.endswith("train/eyeglasses/a.jpg")
    assert Path(rec_out.dst_path).exists()
    assert arr is not None and arr.shape == (16, 16, 3)
    # normalized values in [0,1]
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def test_compute_mean_std_values():
    a = np.ones((2, 2, 3), dtype=np.float32)
    b = np.zeros((2, 2, 3), dtype=np.float32)
    mean, std = compute_mean_std([a, b])
    assert mean == (0.5, 0.5, 0.5)
    assert all(s > 0 for s in std)


def test_write_index_and_stats(tmp_path: Path):
    out_root = tmp_path / "processed"
    recs = [
        ImageRecord("a.jpg", "train", "eyeglasses", src_path="/abs/src/a.jpg", dst_path=str(out_root / "train/eyeglasses/a.jpg"), label=1, interocular=12.3456789),
        ImageRecord("b.jpg", "val", "no_eyeglasses", src_path="/abs/src/b.jpg", dst_path=str(out_root / "val/no_eyeglasses/b.jpg"), label=None, interocular=None),
    ]
    # Ensure out_root exists for index writing
    (out_root).mkdir(parents=True, exist_ok=True)
    write_index(recs, str(out_root))
    csv_path = out_root / "processed_index.csv"
    assert csv_path.exists()
    txt = csv_path.read_text()
    assert "image_id,label,class_name,partition_name,source_path,dest_path,interocular" in txt.splitlines()[0]
    # interocular formatted to 6 decimals when present
    assert any("12.345679" in line for line in txt.splitlines()[1:])

    stats = {"a": 1, "b": [1, 2, 3]}
    stats_path = out_root / "stats" / "stats.json"
    write_stats(str(stats_path), stats)
    assert stats_path.exists()
    loaded = json.loads(stats_path.read_text())
    assert loaded == stats


def test_summarize_counts():
    recs = [
        ImageRecord("a.jpg", "train", "eyeglasses", "src", "dst"),
        ImageRecord("b.jpg", "train", "no_eyeglasses", "src", "dst"),
        ImageRecord("c.jpg", "val", "eyeglasses", "src", "dst"),
    ]
    s = summarize(recs)
    assert s["num_images"] == 3
    assert s["counts_by_split"]["train"] == 2
    assert s["counts_by_class"]["eyeglasses"] == 2


def test_parse_args_values(tmp_path: Path):
    argv = [
        "--subset-root", str(tmp_path / "subset"),
        "--out-root", str(tmp_path / "out"),
        "--size", "32",
        "--center-crop",
        "--normalize-01",
        "--compute-stats",
        "--stats-out", str(tmp_path / "stats.json"),
    ]
    args = parse_args(argv)
    assert args.subset_root == str(tmp_path / "subset")
    assert args.out_root == str(tmp_path / "out")
    assert args.size == 32
    assert args.center_crop is True
    assert args.normalize_01 is True
    assert args.compute_stats is True
    assert args.stats_out == str(tmp_path / "stats.json")


def test_main_integration_with_stats(tmp_path: Path):
    # Create subset and index
    subset = tmp_path / "subset"
    (subset / "train" / "eyeglasses").mkdir(parents=True, exist_ok=True)
    (subset / "val" / "eyeglasses").mkdir(parents=True, exist_ok=True)
    _write_rgb(subset / "train" / "eyeglasses" / "a.jpg", size=(24, 24))
    _write_rgb(subset / "val" / "eyeglasses" / "b.jpg", size=(24, 24))
    idx = pd.DataFrame([
        {"image_id": "a.jpg", "partition_name": "train", "class_name": "eyeglasses", "label": 1, "dest_path": "train/eyeglasses/a.jpg"},
        {"image_id": "b.jpg", "partition_name": "val", "class_name": "eyeglasses", "label": 1, "dest_path": "val/eyeglasses/b.jpg"},
    ])
    idx.to_csv(subset / "subset_index_eyeglasses.csv", index=False)

    out_root = tmp_path / "processed"
    stats_out = tmp_path / "stats.json"
    rc = main([
        "--subset-root", str(subset),
        "--out-root", str(out_root),
        "--size", "16",
        "--center-crop",
        "--normalize-01",
        "--compute-stats",
        "--stats-out", str(stats_out),
    ])
    assert rc == 0
    # Index written
    assert (out_root / "processed_index.csv").exists()
    # Stats written
    assert stats_out.exists()
    st = json.loads(stats_out.read_text())
    assert st["size"] == 16 and st["normalize_01"] is True
    # Processing summary written
    summary_path = out_root / "stats" / "processing_summary.json"
    assert summary_path.exists()
    sm = json.loads(summary_path.read_text())
    assert sm["num_images"] == 2


