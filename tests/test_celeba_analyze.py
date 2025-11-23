import builtins
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.celeba.io import validate_archive_dir, load_archive_data
from src.datasets.celeba.analysis import compute_balance_from_df as compute_balance
from scripts.celeba_analyze import (
    save_plots,
    parse_args,
    main,
)


@pytest.fixture()
def tiny_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    # Minimal CSVs expected by the analyzer
    attrs = pd.DataFrame(
        [
            {"image_id": "img_001.jpg", "Eyeglasses": 1},
            {"image_id": "img_002.jpg", "Eyeglasses": -1},
            {"image_id": "img_003.jpg", "Eyeglasses": 1},
        ]
    )
    parts = pd.DataFrame(
        [
            {"image_id": "img_001.jpg", "partition": 0},
            {"image_id": "img_002.jpg", "partition": 1},
            {"image_id": "img_003.jpg", "partition": 2},
        ]
    )
    attrs.to_csv(archive / "list_attr_celeba.csv", index=False)
    parts.to_csv(archive / "list_eval_partition.csv", index=False)
    # Note: bbox/landmarks are optional for validation, but paths are inspected
    return archive


def test_validate_archive_dir_missing(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    # Only attrs exists
    pd.DataFrame([{"image_id": "a", "A": 1}]).to_csv(archive / "list_attr_celeba.csv", index=False)

    with pytest.raises(FileNotFoundError):
        validate_archive_dir(str(archive))

    # Add parts to satisfy validation
    pd.DataFrame([{"image_id": "a", "partition": 0}]).to_csv(
        archive / "list_eval_partition.csv", index=False
    )
    attrs, parts, bboxes, landmarks = validate_archive_dir(str(archive))
    assert attrs.endswith("list_attr_celeba.csv")
    assert parts.endswith("list_eval_partition.csv")
    assert bboxes.endswith("list_bbox_celeba.csv")
    assert landmarks.endswith("list_landmarks_align_celeba.csv")


def test_load_data_merge_and_types(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    # Third row has a non-numeric attribute that should be dropped
    attrs = pd.DataFrame(
        [
            {"image_id": "img_1.jpg", "A": 1},
            {"image_id": "img_2.jpg", "A": -1},
            {"image_id": "img_3.jpg", "A": "x"},
        ]
    )
    parts = pd.DataFrame(
        [
            {"image_id": "img_1.jpg", "partition": 0},
            {"image_id": "img_2.jpg", "partition": 1},
            {"image_id": "img_3.jpg", "partition": 2},
        ]
    )
    attrs_csv = archive / "list_attr_celeba.csv"
    parts_csv = archive / "list_eval_partition.csv"
    attrs.to_csv(attrs_csv, index=False)
    parts.to_csv(parts_csv, index=False)

    df = load_archive_data(str(attrs_csv), str(parts_csv))
    # Row with non-numeric A is dropped
    assert df.shape[0] == 2
    # Partition name added
    assert set(df["partition_name"]) == {"train", "val"}
    # Nullable integer dtype preserved
    assert str(df["A"].dtype) == "Int64"


def test_compute_balance_counts():
    # Build a simple merged df
    merged = pd.DataFrame(
        [
            {"image_id": "i1", "A": 1, "partition": 0, "partition_name": "train"},
            {"image_id": "i2", "A": -1, "partition": 0, "partition_name": "train"},
            {"image_id": "i3", "A": 1, "partition": 1, "partition_name": "val"},
            {"image_id": "i4", "A": -1, "partition": 2, "partition_name": "test"},
        ]
    )

    summary = compute_balance(merged, ["A"])  # single attribute
    assert list(summary["attribute"]) == ["A"]
    row = summary.iloc[0]
    assert row["total"] == 4
    assert row["pos"] == 2 and row["neg"] == 2
    assert pytest.approx(row["pos_pct"], rel=1e-6) == 0.5
    assert row["train_pos"] == 1 and row["train_neg"] == 1
    assert row["val_pos"] == 1 and row["val_neg"] == 0
    assert row["test_pos"] == 0 and row["test_neg"] == 1


def test_compute_balance_invalid_attribute():
    df = pd.DataFrame(
        [
            {"image_id": "i1", "A": 1, "partition": 0, "partition_name": "train"},
        ]
    )
    with pytest.raises(ValueError):
        compute_balance(df, ["DoesNotExist"])  # invalid attribute name


def test_save_plots_import_failure(monkeypatch, tmp_path: Path):
    # Force matplotlib import to fail
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("No MPL in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    df_summary = pd.DataFrame(
        [
            {"attribute": "A", "pos_pct": 0.5},
            {"attribute": "B", "pos_pct": 0.25},
        ]
    )
    out_dir = tmp_path / "plots"
    save_plots(df_summary, str(out_dir))
    # Directory should not be created when matplotlib is unavailable
    assert not out_dir.exists()


def test_save_plots_with_stub(monkeypatch, tmp_path: Path):
    # Provide a minimal ModuleType stub for matplotlib.pyplot
    import types

    plt_stub = types.ModuleType("matplotlib.pyplot")
    plt_stub.save_count = 0

    def figure(*args, **kwargs):
        return None

    def barh(*args, **kwargs):
        return None

    def bar(*args, **kwargs):
        return None

    def xlabel(*args, **kwargs):
        return None

    def ylabel(*args, **kwargs):
        return None

    def title(*args, **kwargs):
        return None

    def tight_layout(*args, **kwargs):
        return None

    def ylim(*args, **kwargs):
        return None

    def savefig(path, *args, **kwargs):
        # simulate writing a file and count
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"")
        plt_stub.save_count += 1

    def close(*args, **kwargs):
        return None

    plt_stub.figure = figure
    plt_stub.barh = barh
    plt_stub.bar = bar
    plt_stub.xlabel = xlabel
    plt_stub.ylabel = ylabel
    plt_stub.title = title
    plt_stub.tight_layout = tight_layout
    plt_stub.ylim = ylim
    plt_stub.savefig = savefig
    plt_stub.close = close

    matplotlib_stub = types.ModuleType("matplotlib")
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_stub)

    df_summary = pd.DataFrame(
        [
            {"attribute": "A", "pos_pct": 0.5, "train_pos_pct": 0.5, "val_pos_pct": 0.4, "test_pos_pct": 0.3},
            {"attribute": "B", "pos_pct": 0.25, "train_pos_pct": 0.2, "val_pos_pct": 0.3, "test_pos_pct": 0.1},
        ]
    )
    out_dir = tmp_path / "plots"
    save_plots(df_summary, str(out_dir))

    # One overall + one per-attribute (2) => 3 saves
    assert plt_stub.save_count == 3
    assert out_dir.exists()


def test_parse_args_values(tmp_path: Path):
    argv = [
        "--archive-dir",
        str(tmp_path / "archive"),
        "--output-csv",
        str(tmp_path / "out.csv"),
        "--attributes",
        "A",
        "B",
        "--plots",
    ]
    args = parse_args(argv)
    assert args.archive_dir == str(tmp_path / "archive")
    assert args.output_csv == str(tmp_path / "out.csv")
    assert args.attributes == ["A", "B"]
    assert args.plots is True


def test_main_integration_creates_summary_csv(tiny_archive: Path, tmp_path: Path):
    out_csv = tmp_path / "summary.csv"
    rc = main([
        "--archive-dir",
        str(tiny_archive),
        "--output-csv",
        str(out_csv),
    ])
    assert rc == 0
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "attribute" in df.columns
    # expect at least the Eyeglasses row present
    assert (df["attribute"] == "Eyeglasses").any()


