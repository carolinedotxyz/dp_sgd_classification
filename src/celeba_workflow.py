"""Workflow orchestration utilities for CelebA subset build and preprocessing.

This module centralizes small orchestration helpers that were previously
embedded in the notebook script, so they can be reused from both scripts
and notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Notebook-friendly workflow configuration for CelebA Eyeglasses pipeline."""
    archive_dir: Path
    images_root: Path
    output_dir: Path
    subset_root: Path
    out_root: Path
    # Core knobs
    attribute: str = "Eyeglasses"
    max_train: int = 8000
    max_val: int = 1000
    max_test: int = 1000
    link_mode: str = "copy"
    random_seed: int = 42
    # Stage toggles
    run_review: bool = True
    run_build: bool = True
    run_preprocess: bool = True
    run_analyze: bool = True
    # Display/diagnostics
    plot: bool = True
    dry_run: bool = False
    overwrite: bool = True
    visual_split: str = "train"
    visual_sample: int = 256
    plot_top_n_attrs: int = 20
    size_hist_bins: int = 30
    area_hist_bins: int = 40
    proc_pixel_hist_bins: int = 50
    diag_visual_split: str = "train"
    diag_visual_classes: str = "eyeglasses,no_eyeglasses"
    diag_visual_sample: int = 256
    diag_target_size: int = 64
    size_sample_max: int = 5000
    channel_stats_split: str = "train"
    channel_sample_max_images: int = 1500
    channel_stats_scale_01: bool = True
    pixel_hist_sample: int = 128
    pixel_hist_scale_01: bool = False
    # Optional visuals
    plot_balance_bars: bool = False
    # Preprocess
    preprocess_size: int = 64
    preprocess_center_crop: bool = True
    preprocess_normalize_01: bool = True
    preprocess_compute_stats: bool = True

def build_subset(config) -> None:
    """Build the balanced Eyeglasses subset using the builder script's main().

    Args:
        config: Workflow configuration including paths, caps, and flags.

    Side effects:
        - Writes subset directory tree under config.output_dir
        - Writes subset_index_*.csv in config.output_dir
        - Logs progress; may print a brief summary
    """
    # Use src builder API
    from src.celeba_io import load_archive_paths, load_archive_data
    from src.celeba_builder import build_subset as build_subset_core

    # Build merged DF and call src builder directly
    attrs_csv, parts_csv, _, _ = load_archive_paths(config.archive_dir)
    merged = load_archive_data(attrs_csv, parts_csv)
    cap_by_split: Dict[str, int | None] = {
        "train": int(config.max_train) if config.max_train is not None else None,
        "val": int(config.max_val) if config.max_val is not None else None,
        "test": int(config.max_test) if config.max_test is not None else None,
    }
    build_subset_core(
        merged=merged,
        images_root=str(config.images_root),
        output_dir=str(config.output_dir),
        attribute=str(config.attribute),
        link_mode=str(config.link_mode),
        seed=int(config.random_seed),
        max_per_class_by_split=cap_by_split,
        overwrite=bool(config.overwrite),
        dry_run=bool(config.dry_run),
        strict_missing=False,
        fill_missing=True,
        landmarks_lookup=None,
        attrs_to_include=None,
        attrs_lookup=None,
        bbox_lookup=None,
        landmarks_raw_lookup=None,
        stratify_by=[],
        iod_bins=None,
    )
    if (not getattr(config, "dry_run", False)) and Path(config.output_dir).is_dir():
        try:
            # Prefer src utilities if available
            from src.celeba_index import read_processed_index_csv  # type: ignore
        except Exception:
            pass
        # Lightweight inline summary from src (counts by split/class)
        try:
            from collections import Counter as _Counter
            from src.celeba_index import load_subset_index as _load_subset_index
            items_df = _load_subset_index(str(config.output_dir))
            counts = _Counter(zip(items_df["partition_name"], items_df["class_name"]))
            logger.info("Subset counts (files per split/class): %s", dict(counts))
        except Exception:
            logger.info("Subset created in %s", config.output_dir)
        logger.info("Subset created in %s", config.output_dir)
    else:
        if getattr(config, "dry_run", False):
            logger.info("Dry run complete. Review achievable counts in the log.")
        else:
            logger.warning("Command did not complete successfully; see log for details.")


def build_subset_with_skip_summary(config, suppress_per_file_logs: bool = True, skip_csv_name: str = "skipped_missing_images.csv") -> None:
    """Run subset build while summarizing missing-image skips and writing a CSV.

    Captures stdout/stderr from the builder CLI, filters out per-file skip spam
    in the cell output (optional), and writes a CSV listing skipped images.

    Args:
        config: Workflow configuration.
        suppress_per_file_logs: If True, hides individual "Skipping missing image" lines.
        skip_csv_name: File name for the CSV written under config.output_dir.
    """
    try:
        from scripts.celeba_build_subset import main as build_subset_cli_main
    except Exception:
        logger.exception("Failed to import build subset CLI main")
        raise

    import io, contextlib, os, csv
    from typing import List, Tuple

    argv: List[str] = [
        "--archive-dir", str(config.archive_dir),
        "--images-root", str(config.images_root),
        "--output-dir", str(config.output_dir),
        "--attribute", str(config.attribute),
        "--max-per-class-train", str(config.max_train),
        "--max-per-class-val", str(config.max_val),
        "--max-per-class-test", str(config.max_test),
        "--link-mode", str(config.link_mode),
        "--seed", str(config.random_seed),
    ]
    if getattr(config, "overwrite", False):
        argv.append("--overwrite")
    if getattr(config, "dry_run", False):
        argv.append("--dry-run")

    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    logger.info("Running build-subset (captured): %s", " ".join(["celeba_build_subset"] + argv))
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        rc = build_subset_cli_main(argv)
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()

    # Parse skipped lines
    skipped: List[Tuple[str, str]] = []  # (image_id, source_path)
    for line in out.splitlines():
        if "Skipping missing image:" in line:
            try:
                path = line.split(":", 1)[1].strip()
                image_id = os.path.basename(path)
                skipped.append((image_id, path))
            except Exception:
                continue

    # Optionally print filtered output
    if suppress_per_file_logs:
        filtered_lines = [ln for ln in out.splitlines() if "Skipping missing image:" not in ln]
        if filtered_lines:
            print("\n".join(filtered_lines))
    else:
        if out:
            print(out, end="")
    if err:
        print(err, end="")

    # Write CSV of skipped images
    if skipped and not getattr(config, "dry_run", False):
        try:
            csv_path = Path(config.output_dir) / skip_csv_name
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "source_path"]) 
                for image_id, path in skipped:
                    w.writerow([image_id, path])
            print(f"Skipped missing images: {len(skipped)} (written to {csv_path})")
        except Exception as e:
            logger.warning("Failed to write skipped images CSV: %s", e)
    elif skipped:
        # In dry-run, still report count
        print(f"Skipped missing images (dry-run): {len(skipped)}")

    if rc != 0:
        raise RuntimeError(f"celeba_build_subset failed with exit code {rc}")


def build_preprocess_argv(config) -> list[str]:
    argv: list[str] = [
        "--subset-root",
        str(config.subset_root),
        "--out-root",
        str(config.out_root),
        "--size",
        str(config.preprocess_size),
    ]
    if getattr(config, "preprocess_center_crop", False):
        argv.append("--center-crop")
    if getattr(config, "preprocess_normalize_01", False):
        argv.append("--normalize-01")
    if getattr(config, "preprocess_compute_stats", False):
        argv.append("--compute-stats")
    return argv


def run_preprocess_cli(argv: list[str]) -> int:
    from src.celeba_preprocess_core import preprocess_subset
    # Map argv-like flags to direct call
    subset_root = None
    out_root = None
    size = 64
    center_crop = False
    normalize_01 = False
    compute_stats = False
    it = iter(argv)
    for a in it:
        if a == "--subset-root":
            subset_root = next(it)
        elif a == "--out-root":
            out_root = next(it)
        elif a == "--size":
            size = int(next(it))
        elif a == "--center-crop":
            center_crop = True
        elif a == "--normalize-01":
            normalize_01 = True
        elif a == "--compute-stats":
            compute_stats = True
    if subset_root is None:
        raise ValueError("--subset-root is required")
    if out_root is None:
        from pathlib import Path as _P
        out_root = str(_P(subset_root).with_name(_P(subset_root).name + "_processed"))
    preprocess_subset(
        subset_root=subset_root,
        out_root=out_root,
        size=size,
        center_crop=center_crop,
        normalize_01=normalize_01,
        compute_stats=compute_stats,
    )
    return 0


def log_processed_summary(out_root: Path, by_split: dict[str, int], by_class: dict[str, int]) -> None:
    total = sum(by_split.values()) if by_split else 0
    logger.info("Summary:")
    logger.info("- Processed root: %s", out_root)
    logger.info("- Total processed: %s | By split: %s | By class: %s", total, by_split, by_class)



def preprocess_images(config) -> None:
    """Run optional diagnostics and preprocess images using the CLI main.

    Args:
        config: Workflow configuration with subset_root, out_root, and preprocess flags.

    Side effects:
        - Displays diagnostics when enabled
        - Writes processed images, processed_index.csv, and optional stats.json
    """
    # Average-image diagnostics and crop overlays (optional)
    if getattr(config, "plot", False):
        try:
            from src.celeba_diagnostics import select_visual_paths, compute_average_original_and_cropped
            from src.celeba_plots import plot_average_and_diff, plot_center_crop_overlays
        except Exception:
            logger.exception("Diagnostics imports failed; skipping optional visuals")
        else:
            target_size = getattr(config, "diag_target_size", 64)
            paths = select_visual_paths(config)
            if paths:
                avg_orig, avg_crop, count = compute_average_original_and_cropped(paths, target_size)
                if count > 0:
                    plot_average_and_diff(avg_orig, avg_crop, count)
                plot_center_crop_overlays(paths, getattr(config, "random_seed", 42))

    # Run preprocessing script
    argv = build_preprocess_argv(config)
    rc = run_preprocess_cli(argv)
    if rc != 0:
        logger.error("Preprocess failed with exit code %s", rc)
        return

    # Summarize processed index if available
    try:
            from src.celeba_index import (
            augment_processed_index_with_sizes,
            read_processed_index_csv,
            summarize_processed_index_df,
        )
        augment_processed_index_with_sizes(config.out_root)
        _idx = read_processed_index_csv(config.out_root)
        _by_split, _by_class = summarize_processed_index_df(_idx)
        log_processed_summary(config.out_root, _by_split, _by_class)
    except Exception as e:
        logger.warning("Could not summarize processed index: %s", e)


def preview_center_crop_diagnostics(config) -> None:
    """Render optional center-crop diagnostics without running preprocessing."""
    if not getattr(config, "plot", False):
        return
    try:
        from src.celeba_diagnostics import select_visual_paths, compute_average_original_and_cropped
        from src.celeba_plots import plot_average_and_diff, plot_center_crop_overlays
    except Exception:
        logger.exception("Diagnostics imports failed; skipping optional visuals")
        return
    target_size = getattr(config, "diag_target_size", 64)
    paths = select_visual_paths(config)
    if paths:
        avg_orig, avg_crop, count = compute_average_original_and_cropped(paths, target_size)
        if count > 0:
            plot_average_and_diff(avg_orig, avg_crop, count)
        plot_center_crop_overlays(paths, getattr(config, "random_seed", 42))


def preprocess_images_only(config) -> None:
    """Run the preprocessing CLI and summarize processed index (no diagnostics)."""
    argv = build_preprocess_argv(config)
    rc = run_preprocess_cli(argv)
    if rc != 0:
        logger.error("Preprocess failed with exit code %s", rc)
        return
    try:
        from src.celeba_index import (
            augment_processed_index_with_sizes,
            read_processed_index_csv,
            summarize_processed_index_df,
        )
        augment_processed_index_with_sizes(config.out_root)
        _idx = read_processed_index_csv(config.out_root)
        _by_split, _by_class = summarize_processed_index_df(_idx)
        log_processed_summary(config.out_root, _by_split, _by_class)
    except Exception as e:
        logger.warning("Could not summarize processed index: %s", e)


def review_archive(config) -> None:
    """Review the CelebA archive: validate files, show splits and attribute balance.

    Side effects:
        - Displays compact validation and tables/plots
        - Writes balance plots and CSV under archive_dir
    """
    # Local imports to avoid unnecessary top-level dependencies
    from typing import List
    from IPython.display import display, HTML  # type: ignore
    import pandas as pd  # type: ignore
    from src.celeba_io import load_archive_paths, load_archive_data
    from src.nb_display import (
        _h,
        style_focus_table,
        render_validation_badges,
    )
    from src.nb_utils import pick_column, to_percent_series
    from src.celeba_analysis import (
        compute_attribute_summary,
        build_focus_table,
    )
    from src.celeba_plots import (
        plot_split_sizes,
        plot_attribute_overall,
    )
    from src.celeba_analysis import compute_balance_from_df as compute_balance

    attrs_csv, parts_csv, bboxes_csv, landmarks_csv = load_archive_paths(config.archive_dir)
    render_validation_badges(attrs_csv, parts_csv, bboxes_csv, landmarks_csv)
    # Load archive data using src helper
    df = load_archive_data(attrs_csv, parts_csv)
    num_rows = df.shape[0]
    all_attr_cols: List[str] = [c for c in df.columns if c not in ("image_id", "partition", "partition_name")]
    meta_html = f"""
    <div style="margin-top:6px;color:#374151">\n  Rows: <b>{num_rows:,}</b> &nbsp; • &nbsp; Attributes: <b>{len(all_attr_cols)}</b>\n</div>"""
    display(HTML(meta_html))
    _h("Official splits", "4", "From list_eval_partition.csv")
    split_counts = (
        df["partition_name"].value_counts().rename_axis("split").reset_index(name="count").sort_values("split")
    )
    tbl = split_counts.copy(); tbl["count"] = tbl["count"].map(lambda x: f"{x:,}")
    display(tbl.style.hide(axis="index"))
    plot_split_sizes(split_counts)
    _h("Attribute balance", "4", "Positive fraction per attribute (overall)")
    summary_all = compute_attribute_summary(df)
    plot_attribute_overall(summary_all, config.plot_top_n_attrs)
    _h("Focused attributes", "4", "Positive rate by split — sorted, with drift deltas")
    from src.nb_display import PREFERRED_FOCUS
    focus_attrs = [a for a in PREFERRED_FOCUS if a in all_attr_cols] or all_attr_cols[:6]
    summary_focus = compute_balance(df, focus_attrs).copy()
    col_overall = pick_column(["pos_frac","pos_pct"], summary_focus.columns)
    col_train   = pick_column(["train_pos_frac","train_pos_pct"], summary_focus.columns)
    col_val     = pick_column(["val_pos_frac","val_pos_pct"], summary_focus.columns)
    col_test    = pick_column(["test_pos_frac","test_pos_pct"], summary_focus.columns)
    ov  = to_percent_series(summary_focus, col_overall); trn = to_percent_series(summary_focus, col_train)
    val = to_percent_series(summary_focus, col_val);    tst = to_percent_series(summary_focus, col_test)
    focus_tbl = build_focus_table(summary_focus)
    focus_tbl = focus_tbl.sort_values("Overall %", ascending=False).reset_index(drop=True)
    display(style_focus_table(focus_tbl))
    from src.celeba_io import write_archive_outputs as _write_archive_outputs
    _write_archive_outputs(summary_all, config.archive_dir)
    logger.info("Review archive section completed.")


def compute_channel_stats_for_paths(paths: list[str], scale_01: bool):
    from src.celeba_diagnostics import compute_channel_stats
    return compute_channel_stats(paths, scale_01=scale_01)


def analyze_processed(config) -> None:
    """Compare processed vs original subset; sizes, balance, and channel stats."""
    # Local imports to keep dependencies optional
    import os, json
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from IPython.display import display, HTML  # type: ignore
    from src.nb_display import _h
    from src.celeba_plots import (
        plot_channel_bars,
        plot_processed_pixel_hists,
        render_size_block,
    )
    from src.celeba_index import (
        read_processed_index_csv,
        augment_processed_index_with_sizes,
        summarize_processed_index_df,
        load_processed_index_or_raise,
        ensure_partition_and_class_columns,
    )
    from src.celeba_diagnostics import sample_paths
    from src.celeba_plots import plot_grouped_bars_two_series
    from src.celeba_analysis import build_balance_comparison

    _h("Datasets")
    display(HTML(
        f"<div style='color:#374151'>Original root: <code>{config.subset_root}</code></div>"
        f"<div style='color:#374151'>Processed root: <code>{config.out_root}</code></div>"
    ))
    from src.celeba_index import load_subset_index
    dfo = load_subset_index(str(config.subset_root))
    display(HTML(f"<div style='color:#374151'>Original images: <b>{len(dfo):,}</b></div>"))
    dfp = ensure_partition_and_class_columns(load_processed_index_or_raise(config.out_root))
    dfp["abs_path"] = [v if os.path.isabs(v) else str(config.out_root / v) for v in dfp["dest_path"].astype(str)]
    display(HTML(f"<div style='color:#374151'>Processed images: <b>{len(dfp):,}</b></div>"))
    _h("Class balance by split", "Positive class = 'eyeglasses'")
    bal_before, bal_after, splits, cmp_balance = build_balance_comparison(dfo, dfp)
    display(cmp_balance)
    if config.plot and getattr(config, "plot_balance_bars", False) and "pos_%" in bal_before.columns and "pos_%" in bal_after.columns:
        yb = [float(bal_before.loc[s]["pos_%"]) if s in bal_before.index else float("nan") for s in splits]
        ya = [float(bal_after.loc[s]["pos_%"])  if s in bal_after.index  else float("nan") for s in splits]
        plot_grouped_bars_two_series(
            categories=splits,
            series_a=yb,
            series_b=ya,
            label_a="before",
            label_b="after",
            ylabel="positive rate (%)",
            title="Positive class rate by split",
            ylim=(0, 100),
            figsize=(6.8, 3.0),
        )
    paths_before = dfo["source_path"].astype(str).tolist()
    paths_after  = dfp["abs_path"].astype(str).tolist()
    size_paths_b = sample_paths(paths_before, int(config.size_sample_max), config.random_seed)
    size_paths_a = sample_paths(paths_after,  int(config.size_sample_max), config.random_seed)
    from src.celeba_diagnostics import collect_size_stats
    sizes_b = collect_size_stats(size_paths_b)
    sizes_a = collect_size_stats(size_paths_a)
    render_size_block("Before (original)", sizes_b)
    render_size_block("After (processed)", sizes_a)
    _h("Channel stats (TRAIN only)", f"Sample up to {config.channel_sample_max_images} images; scale={'[0,1]' if config.channel_stats_scale_01 else '[0,255]'}")
    train_b = dfo[dfo["partition_name"] == config.channel_stats_split]["source_path"].astype(str).tolist()
    train_a = dfp[dfp["partition_name"] == config.channel_stats_split]["abs_path"].astype(str).tolist()
    train_b = sample_paths(train_b, int(config.channel_sample_max_images), config.random_seed)
    train_a = sample_paths(train_a, int(config.channel_sample_max_images), config.random_seed)
    mean_b, std_b = compute_channel_stats_for_paths(train_b, scale_01=bool(config.channel_stats_scale_01))
    mean_a, std_a = compute_channel_stats_for_paths(train_a, scale_01=bool(config.channel_stats_scale_01))
    m_b = tuple(round(float(x), 6) for x in mean_b); s_b = tuple(round(float(x), 6) for x in std_b)
    m_a = tuple(round(float(x), 6) for x in mean_a); s_a = tuple(round(float(x), 6) for x in std_a)
    chan_cmp = pd.DataFrame({
        "": ["R","G","B"],
        "mean (before)": m_b,
        "mean (after)":  m_a,
        "Δ mean": [round(m_a[i]-m_b[i], 6) for i in range(3)],
        "std (before)": s_b,
        "std (after)":  s_a,
        "Δ std":  [round(s_a[i]-s_b[i], 6) for i in range(3)],
    })
    display(chan_cmp.set_index(""))
    if config.plot:
        plot_channel_bars(m_b, m_a, s_b, s_a)
    stats_path = config.out_root / "stats" / "stats.json"
    from src.nb_display import compare_saved_stats_and_display
    compare_saved_stats_and_display(stats_path, m_a, s_a)
    # Show both original and processed pixel distributions in the same section
    # Use [0,1] for processed when preprocessing used normalize_01
    orig_scale01 = bool(getattr(config, "pixel_hist_scale_01", False))
    proc_scale01 = bool(getattr(config, "pixel_hist_scale_01", False) or getattr(config, "preprocess_normalize_01", False))
    if config.plot and len(train_b) > 0:
        from src.celeba_plots import plot_original_pixel_hists
        plot_original_pixel_hists(
            train_b,
            sample_n=int(getattr(config, "pixel_hist_sample", 128)),
            scale_01=orig_scale01,
            seed=config.random_seed,
        )
    if config.plot and len(train_a) > 0:
        plot_processed_pixel_hists(
            train_a,
            bins=int(getattr(config, "proc_pixel_hist_bins", 50)),
            scale_01=proc_scale01,
            seed=config.random_seed,
        )


def analyze_original_subset(config) -> None:
    """Analyze the original subset: class balance, size/aspect, and pixel hists."""
    from IPython.display import display, HTML  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from src.nb_display import _h, _has_variance
    from src.celeba_index import load_subset_index
    from src.celeba_diagnostics import sample_paths, collect_size_stats, area_retained_after_center_square, describe_numeric
    from src.celeba_plots import plot_original_pixel_hists
    from src.celeba_analysis import build_balance_display_table as _build_bal_table

    df = load_subset_index(str(config.subset_root))
    _h("Subset overview")
    display(HTML(
        f'<div style="color:#374151">Root: <code>{config.subset_root}</code>'
        f' &nbsp; • &nbsp; Total images: <b>{len(df):,}</b></div>'
    ))
    _h("Counts by split and class", "Includes positive ratios and deltas vs 50%")
    # Compute balance table directly from df
    t = df.groupby(["partition_name", "class_name"])['image_id'].count().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    if "eyeglasses" in t.columns:
        t["pos_ratio"] = (t["eyeglasses"] / t["total"]).round(4)
    balance = t.sort_index()
    fmt_bal = _build_bal_table(balance)
    display(fmt_bal.style.hide(axis="index"))
    if "pos_ratio" in balance.columns and len(balance) > 0 and config.plot:
        from src.celeba_plots import plot_class_balance_stacked
        splits = balance.index.tolist()
        pos = (balance["pos_ratio"] * balance["total"]).astype(float).tolist()
        total = balance["total"].astype(float).tolist()
        plot_class_balance_stacked(splits, pos, total)
    all_paths = df["source_path"].astype(str).tolist()
    size_paths = sample_paths(all_paths, int(config.size_sample_max), config.random_seed)
    sizes_df = collect_size_stats(size_paths)
    if not sizes_df.empty:
        sizes_df["area_retained_center_square"] = sizes_df.apply(
            lambda r: area_retained_after_center_square(int(r["width"]), int(r["height"])), axis=1
        )
    if not sizes_df.empty and config.plot:
        has_var_width  = _has_variance(sizes_df["width"])
        has_var_height = _has_variance(sizes_df["height"])
        has_var_aspect = _has_variance(sizes_df["aspect"])
        if not (has_var_width or has_var_height or has_var_aspect):
            w = int(sizes_df['width'].iloc[0]); h = int(sizes_df['height'].iloc[0]); ar = float(sizes_df['aspect'].iloc[0])
            _h("Image size summary (sample)")
            display(HTML(f"<div style='color:#374151'>All sampled images share the same size: <b>{w}×{h}</b> px (aspect ratio <b>{ar:.2f}</b>).</div>"))
        else:
            _h("Image size summary (sample)")
            display(describe_numeric(sizes_df, ["width","height","aspect"]))
            cols, titles = [], []
            if has_var_width: cols.append("width");  titles.append("Width")
            if has_var_height: cols.append("height"); titles.append("Height")
            if has_var_aspect: cols.append("aspect"); titles.append("Aspect ratio (W/H)")
            if cols:
                fig, axes = plt.subplots(1, len(cols), figsize=(10, 3.2))
                if len(cols) == 1: axes = [axes]
                for ax, col, ttl in zip(axes, cols, titles):
                    ax.hist(sizes_df[col], bins=config.size_hist_bins)
                    if col == "aspect": ax.axvline(1.0, linestyle="--")
                    ax.set_title(ttl); ax.set_xlabel("")
                    if ax is not axes[0]: ax.set_yticklabels([])
                    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
                plt.tight_layout(); plt.show()
        if "area_retained_center_square" in sizes_df.columns:
            has_var_area = _has_variance(sizes_df["area_retained_center_square"])
            if not has_var_area:
                val = float(sizes_df["area_retained_center_square"].iloc[0])
                _h("Area retained by center square crop", "1.0 means no loss")
                display(HTML(f"<div style='color:#374151'>Center-square crop keeps a constant <b>{val:.2f}</b> fraction (~{val*100:.1f}%) of the original area across all sampled images.</div>"))
            else:
                _h("Area retained by center square crop", "1.0 means no loss")
                display(describe_numeric(sizes_df, ["area_retained_center_square"]))
                from src.celeba_plots import plot_area_retained_hist
                plot_area_retained_hist(sizes_df, bins=config.area_hist_bins)
    train_paths = df[df["partition_name"] == config.channel_stats_split]["source_path"].astype(str).tolist()
    train_paths = sample_paths(train_paths, int(config.channel_sample_max_images), config.random_seed)
    from src.celeba_diagnostics import compute_channel_stats
    mean_rgb, std_rgb = compute_channel_stats(train_paths, scale_01=bool(config.channel_stats_scale_01))
    scale_note = "[0,1] scale" if config.channel_stats_scale_01 else "[0,255] scale"
    _h("Estimated channel mean/std (original)", f"{len(train_paths)} images from TRAIN, {scale_note}")
    m = tuple(round(float(x), 6) for x in mean_rgb); s = tuple(round(float(x), 6) for x in std_rgb)
    display(HTML(f"<pre style='margin:0'>mean_rgb = {m}\nstd_rgb  = {s}</pre>"))
    if config.plot and len(train_paths) > 0:
        plot_original_pixel_hists(train_paths, sample_n=int(config.pixel_hist_sample), scale_01=bool(config.pixel_hist_scale_01), seed=config.random_seed)
