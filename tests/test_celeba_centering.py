import builtins
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.celeba_centering import (
    Item,
    ensure_dir,
    load_subset_items,
    load_landmarks,
    face_center_from_landmarks,
    compute_offsets,
    save_plots,
    draw_outliers_grid,
    parse_args,
    main,
)


def _write_rgb(path: Path, color=(200, 100, 50), size=(178, 218)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    Image.fromarray(arr).save(path)


@pytest.fixture()
def tiny_subset(tmp_path: Path) -> Path:
    root = tmp_path / "subset"
    # Image layout compatible with loader fallback
    for split in ("train", "val", "test"):
        for cls in ("eyeglasses", "no_eyeglasses"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)
    _write_rgb(root / "train" / "eyeglasses" / "img_001.jpg")
    _write_rgb(root / "train" / "no_eyeglasses" / "img_002.jpg")
    _write_rgb(root / "val" / "eyeglasses" / "img_003.jpg")

    # Also create an index CSV to exercise fast path
    idx = pd.DataFrame(
        [
            {"image_id": "img_001.jpg", "partition_name": "train", "class_name": "eyeglasses", "dest_path": "train/eyeglasses/img_001.jpg"},
            {"image_id": "img_002.jpg", "partition_name": "train", "class_name": "no_eyeglasses", "dest_path": "train/no_eyeglasses/img_002.jpg"},
            {"image_id": "img_003.jpg", "partition_name": "val", "class_name": "eyeglasses", "dest_path": "val/eyeglasses/img_003.jpg"},
        ]
    )
    idx.to_csv(root / "subset_index_eyeglasses.csv", index=False)
    return root


@pytest.fixture()
def tiny_landmarks(tmp_path: Path) -> Path:
    arch = tmp_path / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    # Minimal required columns
    df = pd.DataFrame(
        [
            {"image_id": "img_001.jpg", "lefteye_x": 60.0, "lefteye_y": 80.0, "righteye_x": 120.0, "righteye_y": 80.0, "nose_x": 90.0, "nose_y": 120.0},
            {"image_id": "img_002.jpg", "lefteye_x": 40.0, "lefteye_y": 60.0, "righteye_x": 140.0, "righteye_y": 60.0, "nose_x": 90.0, "nose_y": 110.0},
            # Leave img_003 missing to exercise missing-landmark path
        ]
    )
    df.to_csv(arch / "list_landmarks_align_celeba.csv", index=False)
    return arch


def test_load_subset_items_reads_index_first(tiny_subset: Path):
    items = load_subset_items(str(tiny_subset))
    assert len(items) == 3
    assert {it.image_id for it in items} == {"img_001.jpg", "img_002.jpg", "img_003.jpg"}
    assert {it.split for it in items} == {"train", "val"}


def test_load_subset_items_fallback_walk(tmp_path: Path):
    root = tmp_path / "subset"
    _write_rgb(root / "test" / "eyeglasses" / "x.jpg")
    items = load_subset_items(str(root))
    assert len(items) == 1
    assert items[0].class_name == "eyeglasses"


def test_load_landmarks_validates_columns(tmp_path: Path):
    arch = tmp_path / "arch"
    arch.mkdir()
    with pytest.raises(FileNotFoundError):
        load_landmarks(str(arch))

    # Create with missing column
    df = pd.DataFrame([{"image_id": "a", "lefteye_x": 0, "lefteye_y": 0, "righteye_x": 0, "righteye_y": 0, "nose_x": 0}])
    (arch / "list_landmarks_align_celeba.csv").write_text(df.to_csv(index=False))
    with pytest.raises(ValueError):
        load_landmarks(str(arch))


def test_face_center_from_landmarks():
    row = pd.Series({"lefteye_x": 0.0, "lefteye_y": 0.0, "righteye_x": 2.0, "righteye_y": 0.0, "nose_x": 2.0, "nose_y": 2.0})
    cx, cy = face_center_from_landmarks(row)
    # Eyes midpoint at (1,0); nudged 25% toward nose (2,2) -> x=1.25, y=0.5
    assert pytest.approx(cx, rel=1e-6) == 1.25
    assert pytest.approx(cy, rel=1e-6) == 0.5


def test_compute_offsets_happy_path(tiny_subset: Path, tiny_landmarks: Path):
    items = load_subset_items(str(tiny_subset))
    lm = load_landmarks(str(tiny_landmarks))
    df = compute_offsets(items, lm)
    # img_003 is missing landmarks; expect 2 rows
    assert len(df) == 2
    assert set(df["image_id"]) == {"img_001.jpg", "img_002.jpg"}
    assert {"dx", "dy", "radius", "cx", "cy"}.issubset(df.columns)


def test_save_plots_import_failure(monkeypatch, tmp_path: Path):
    # Force matplotlib import to fail
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("No MPL in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    df = pd.DataFrame([
        {"dx": 0.0, "dy": 0.0, "radius": 0.1},
        {"dx": 0.1, "dy": -0.1, "radius": 0.2},
    ])
    out_dir = tmp_path / "plots"
    save_plots(df, str(out_dir))
    assert not out_dir.exists()


def test_save_plots_with_stub(monkeypatch, tmp_path: Path):
    import types

    plt_stub = types.ModuleType("matplotlib.pyplot")
    plt_stub.save_count = 0

    def figure(*args, **kwargs):
        return None

    def hist(*args, **kwargs):
        return None

    def hexbin(*args, **kwargs):
        class HB: pass
        return HB()

    def axvline(*args, **kwargs):
        return None

    def axhline(*args, **kwargs):
        return None

    def xlabel(*args, **kwargs):
        return None

    def ylabel(*args, **kwargs):
        return None

    def title(*args, **kwargs):
        return None

    def colorbar(hb):
        class CB:
            def set_label(self, *args, **kwargs):
                return None
        return CB()

    def tight_layout(*args, **kwargs):
        return None

    def savefig(path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
        plt_stub.save_count += 1

    def close(*args, **kwargs):
        return None

    plt_stub.figure = figure
    plt_stub.hist = hist
    plt_stub.hexbin = hexbin
    plt_stub.axvline = axvline
    plt_stub.axhline = axhline
    plt_stub.xlabel = xlabel
    plt_stub.ylabel = ylabel
    plt_stub.title = title
    plt_stub.colorbar = colorbar
    plt_stub.tight_layout = tight_layout
    plt_stub.savefig = savefig
    plt_stub.close = close

    matplotlib_stub = types.ModuleType("matplotlib")
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_stub)

    df = pd.DataFrame([
        {"dx": 0.0, "dy": 0.0, "radius": 0.1},
        {"dx": 0.1, "dy": -0.1, "radius": 0.2},
    ])
    out_dir = tmp_path / "plots"
    save_plots(df, str(out_dir))
    # Expect two saves: histogram + hexbin
    assert plt_stub.save_count == 2
    assert out_dir.exists()


def test_draw_outliers_grid(tmp_path: Path):
    # Build a small df with valid paths
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    _write_rgb(img1)
    _write_rgb(img2)
    df = pd.DataFrame([
        {"image_id": "a.jpg", "split": "train", "class_name": "eyeglasses", "cx": 20.0, "cy": 30.0, "radius": 0.9, "path": str(img1)},
        {"image_id": "b.jpg", "split": "val", "class_name": "no_eyeglasses", "cx": 30.0, "cy": 40.0, "radius": 0.8, "path": str(img2)},
    ])
    out_png = tmp_path / "grid" / "grid.png"
    draw_outliers_grid(df, str(out_png), k=2, cols=2)
    assert out_png.exists()


def test_parse_args_values(tmp_path: Path):
    argv = [
        "--subset-root", str(tmp_path / "subset"),
        "--archive-dir", str(tmp_path / "archive"),
        "--out-dir", str(tmp_path / "out"),
        "--k-outliers", "16",
    ]
    args = parse_args(argv)
    assert args.subset_root == str(tmp_path / "subset")
    assert args.archive_dir == str(tmp_path / "archive")
    assert args.out_dir == str(tmp_path / "out")
    assert args.k_outliers == 16


def test_main_integration(tmp_path: Path):
    # Create subset with three images and index
    subset = tmp_path / "subset"
    (subset / "train" / "eyeglasses").mkdir(parents=True, exist_ok=True)
    (subset / "train" / "no_eyeglasses").mkdir(parents=True, exist_ok=True)
    (subset / "val" / "eyeglasses").mkdir(parents=True, exist_ok=True)
    _write_rgb(subset / "train" / "eyeglasses" / "img_001.jpg")
    _write_rgb(subset / "train" / "no_eyeglasses" / "img_002.jpg")
    _write_rgb(subset / "val" / "eyeglasses" / "img_003.jpg")
    idx = pd.DataFrame([
        {"image_id": "img_001.jpg", "partition_name": "train", "class_name": "eyeglasses", "dest_path": "train/eyeglasses/img_001.jpg"},
        {"image_id": "img_002.jpg", "partition_name": "train", "class_name": "no_eyeglasses", "dest_path": "train/no_eyeglasses/img_002.jpg"},
        {"image_id": "img_003.jpg", "partition_name": "val", "class_name": "eyeglasses", "dest_path": "val/eyeglasses/img_003.jpg"},
    ])
    idx.to_csv(subset / "subset_index_eyeglasses.csv", index=False)

    # Create landmarks archive
    arch = tmp_path / "archive"
    arch.mkdir()
    lm = pd.DataFrame([
        {"image_id": "img_001.jpg", "lefteye_x": 60.0, "lefteye_y": 80.0, "righteye_x": 120.0, "righteye_y": 80.0, "nose_x": 90.0, "nose_y": 120.0},
        {"image_id": "img_002.jpg", "lefteye_x": 40.0, "lefteye_y": 60.0, "righteye_x": 140.0, "righteye_y": 60.0, "nose_x": 90.0, "nose_y": 110.0},
        {"image_id": "img_003.jpg", "lefteye_x": 40.0, "lefteye_y": 60.0, "righteye_x": 140.0, "righteye_y": 60.0, "nose_x": 90.0, "nose_y": 110.0},
    ])
    (arch / "list_landmarks_align_celeba.csv").write_text(lm.to_csv(index=False))

    out_dir = tmp_path / "out"
    # Inject a matplotlib stub so plots are generated during integration
    import types
    plt_stub = types.ModuleType("matplotlib.pyplot")
    def savefig(path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
    plt_stub.figure = lambda *a, **k: None
    plt_stub.hist = lambda *a, **k: None
    plt_stub.hexbin = lambda *a, **k: type("HB", (), {})()
    plt_stub.axvline = lambda *a, **k: None
    plt_stub.axhline = lambda *a, **k: None
    plt_stub.xlabel = lambda *a, **k: None
    plt_stub.ylabel = lambda *a, **k: None
    plt_stub.title = lambda *a, **k: None
    plt_stub.colorbar = lambda *a, **k: type("CB", (), {"set_label": lambda *a, **k: None})()
    plt_stub.tight_layout = lambda *a, **k: None
    plt_stub.savefig = savefig
    plt_stub.close = lambda *a, **k: None
    matplotlib_stub = types.ModuleType("matplotlib")
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = plt_stub

    rc = main([
        "--subset-root", str(subset),
        "--archive-dir", str(arch),
        "--out-dir", str(out_dir),
        "--k-outliers", "4",
    ])
    assert rc == 0
    # CSV written
    csv_path = out_dir / "centering_metrics.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    # Artifacts
    assert (out_dir / "center_distance_hist.png").exists()
    assert (out_dir / "center_offset_hex.png").exists()
    assert (out_dir / "outliers_grid.png").exists()


