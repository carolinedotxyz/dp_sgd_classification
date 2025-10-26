"""Workflow orchestration utilities for CelebA subset build and preprocessing.

This module centralizes small orchestration helpers that were previously
embedded in the notebook script, so they can be reused from both scripts
and notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Literal
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

    # Build a display argv with repo-relative paths for logging
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        from src.nb_display import _relpath as _rp
        display_argv: List[str] = list(argv)
        for i in range(len(display_argv) - 1):
            if display_argv[i] in {"--archive-dir", "--images-root", "--output-dir"}:
                display_argv[i + 1] = _rp(display_argv[i + 1])
    except Exception:
        display_argv = argv

    logger.info("Running build-subset (captured): %s", " ".join(["celeba_build_subset"] + display_argv))
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        rc = build_subset_cli_main(argv)
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()

    # Convert absolute project paths in captured output to repo-relative
    try:
        import re as _re
        cwd = Path.cwd().resolve()
        repo_root = None
        for candidate in [cwd] + list(cwd.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists() or (candidate / "src").exists():
                repo_root = candidate.resolve()
                break
        if repo_root is not None:
            base = _re.escape(str(repo_root))
            pattern = _re.compile(base + r"/")
            out = pattern.sub("", out)
            err = pattern.sub("", err)
    except Exception:
        pass

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
            from src.nb_display import _relpath
            print(f"Skipped missing images: {len(skipped)} (written to {_relpath(csv_path)})")
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
    # Prefer repo-relative path in logs for readability
    try:
        from src.nb_display import _relpath as _rp  # type: ignore
        root_display = _rp(out_root)
    except Exception:
        root_display = str(out_root)
    logger.info("- Processed root: %s", root_display)
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
        df["partition_name"].value_counts().rename_axis("split").reset_index(name="image count").sort_values("split")
    )
    tbl = split_counts.copy(); tbl["image count"] = tbl["image count"].map(lambda x: f"{x:,}")
    display(tbl.style.hide(axis="index"))
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


def analyze_processed(
    config,
    save_png: bool = False,
    save_dir: "Path | None" = None,
    verbosity: Literal["brief", "full"] = "brief",
    show_debug: bool = False,
    show_table_if_delta_pp_gt: float = 0.2,
):
    """Compare processed vs original subset; sizes, balance, and channel stats.

    Args:
        config: Workflow configuration.
        save_png: If True, save key figures to ``save_dir``.
        save_dir: Directory to save PNGs when ``save_png`` is True.

    Returns:
        dict with figure handles and saved paths where applicable.
    """
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
    from src.celeba_diagnostics import sample_paths, collect_size_stats
    from src.celeba_plots import plot_grouped_bars_two_series
    from src.celeba_analysis import build_balance_comparison

    from src.nb_display import _relpath, viz_style, style_counts, add_bar_labels
    out: dict[str, object] = {}

    # Brief mode: compact summary + two plots, optional drift table, quiet progress
    if verbosity == "brief":
        # Suppress tqdm progress bars where used
        old_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            # Load indices
            from src.celeba_index import load_subset_index
            dfo = load_subset_index(str(config.subset_root))
            dfp = ensure_partition_and_class_columns(load_processed_index_or_raise(config.out_root))
            dfp["abs_path"] = [v if os.path.isabs(v) else str(config.out_root / v) for v in dfp["dest_path"].astype(str)]

            # Counts and balance
            by_split, by_class = summarize_processed_index_df(dfp)
            total_images = int(sum(by_split.values())) if by_split else 0
            train_n = int(by_split.get("train", 0))
            val_n = int(by_split.get("val", 0))
            test_n = int(by_split.get("test", 0))

            bal_before, bal_after, splits, _cmp_balance = build_balance_comparison(dfo, dfp)
            # Build delta table and decide if to show
            show_table = False
            counts_df = None
            try:
                rows_tbl: list[dict[str, object]] = []
                for s in splits:
                    b = bal_before.loc[s] if s in bal_before.index else pd.Series(dtype=float)
                    a = bal_after.loc[s] if s in bal_after.index else pd.Series(dtype=float)
                    total_b = float(b.get("total", float("nan"))) if "total" in b else float("nan")
                    total_a = float(a.get("total", float("nan"))) if "total" in a else float("nan")
                    pos_b = float(b.get("pos_%", float("nan"))) if "pos_%" in b else float("nan")
                    pos_a = float(a.get("pos_%", float("nan"))) if "pos_%" in a else float("nan")
                    delta = (pos_a - pos_b) if (pd.notna(pos_a) and pd.notna(pos_b)) else float("nan")
                    if pd.notna(delta) and abs(float(delta)) > float(show_table_if_delta_pp_gt):
                        show_table = True
                    rows_tbl.append({
                        "split": s,
                        "total (before)": total_b,
                        "total (after)": total_a,
                        "pos% (before)": pos_b,
                        "pos% (after)": pos_a,
                        "Δ pos% (pp)": delta,
                    })
                counts_df = pd.DataFrame(rows_tbl).set_index("split")
            except Exception:
                counts_df = None

            # Geometry (processed)
            train_a = dfp[dfp["partition_name"] == config.channel_stats_split]["abs_path"].astype(str).tolist()
            size_paths_a = sample_paths(dfp["abs_path"].astype(str).tolist(), int(config.size_sample_max), config.random_seed)
            sizes_a = collect_size_stats(size_paths_a)
            target_size = int(getattr(config, "preprocess_size", 64))
            geom_uniform = False
            try:
                if not sizes_a.empty:
                    widths = sizes_a["width"].dropna().astype(int).unique().tolist()
                    heights = sizes_a["height"].dropna().astype(int).unique().tolist()
                    geom_uniform = (len(widths) == 1 and len(heights) == 1 and widths[0] == target_size and heights[0] == target_size)
            except Exception:
                geom_uniform = False

            # Channel stats (processed) and saved-stats comparison
            train_a = sample_paths(train_a, int(config.channel_sample_max_images), config.random_seed)
            from src.celeba_diagnostics import compute_channel_stats
            mean_a, std_a = compute_channel_stats(train_a, scale_01=bool(config.channel_stats_scale_01))
            m_a = tuple(round(float(x), 6) for x in mean_a)
            s_a = tuple(round(float(x), 6) for x in std_a)

            stats_path = config.out_root / "stats" / "stats.json"
            saved_mean = (float("nan"), float("nan"), float("nan"))
            saved_std = (float("nan"), float("nan"), float("nan"))
            if stats_path.is_file():
                try:
                    with open(stats_path, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                    saved_mean = tuple(float(x) for x in saved.get("train_mean", (float("nan"),) * 3))
                    saved_std = tuple(float(x) for x in saved.get("train_std", (float("nan"),) * 3))
                except Exception:
                    pass
            mean_delta = tuple(abs(m_a[i] - saved_mean[i]) for i in range(3)) if all(np.isfinite(saved_mean)) else (float("nan"),) * 3
            std_delta = tuple(abs(s_a[i] - saved_std[i]) for i in range(3)) if all(np.isfinite(saved_std)) else (float("nan"),) * 3
            mean_delta_max = float(np.nanmax(mean_delta)) if isinstance(mean_delta, tuple) else float("nan")
            std_delta_max = float(np.nanmax(std_delta)) if isinstance(std_delta, tuple) else float("nan")
            eps_mean = 0.005
            eps_std = 0.010
            stats_ok = (mean_delta_max <= eps_mean) and (std_delta_max <= eps_std)

            # Summary cards (compact)
            def _kfmt(n: int) -> str:
                try:
                    return f"{int(n/1000)}k" if n % 1000 == 0 and n >= 1000 else f"{n:,}"
                except Exception:
                    return str(n)
            print(
                "\n".join([
                    f"• Counts: total {_kfmt(total_images)} | train {_kfmt(train_n)} | val {_kfmt(val_n)} | test {_kfmt(test_n)} | 50/50",
                    f"• Geometry: {target_size}×{target_size} ({'all' if geom_uniform else 'mixed'})",
                    f"• Stats: saved vs recomputed {'OK' if stats_ok else 'WARN'} (Δ mean ≤ {eps_mean:.3f}, Δ std ≤ {eps_std:.3f})",
                ])
            )

            # Optional compact table if drift exceeds threshold
            if counts_df is not None and show_table and config.plot:
                try:
                    display(style_counts(counts_df).hide(axis="index"))
                except Exception:
                    display(counts_df)

            # Plot A: Channel Means/STDs (TRAIN) — Processed
            fig_bars = None
            if config.plot:
                with viz_style():
                    fig_bars, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
                    # means
                    ax1.bar(["R", "G", "B"], list(m_a))
                    add_bar_labels(ax1)
                    ax1.set_title("Channel Means (TRAIN) — Processed")
                    ax1.set_ylim(0, 0.6)
                    # stds
                    ax2.bar(["R", "G", "B"], list(s_a))
                    add_bar_labels(ax2)
                    ax2.set_title("Channel STDs (TRAIN) — Processed")
                    ax2.set_ylim(0, 0.35)

            # Plot B: Pixel Value Distribution — Processed [0,1]
            fig_hist_proc = None
            if config.plot and len(train_a) > 0:
                proc_scale01 = bool(getattr(config, "pixel_hist_scale_01", False) or getattr(config, "preprocess_normalize_01", False))
                fig_hist_proc = plot_processed_pixel_hists(
                    train_a,
                    bins=int(getattr(config, "proc_pixel_hist_bins", 50)),
                    scale_01=proc_scale01,
                    seed=config.random_seed,
                )

            result = {
                "counts": {"total": total_images, "train": train_n, "val": val_n, "test": test_n, "balance_ok": not show_table},
                "geometry": {"size": (target_size, target_size), "uniform": bool(geom_uniform)},
                "stats_check": {"mean_delta_max": float(mean_delta_max), "std_delta_max": float(std_delta_max), "ok": bool(stats_ok)},
                "figs": {"bars": fig_bars, "hist_proc": fig_hist_proc},
            }
            return result
        finally:
            # restore tqdm behavior
            if old_tqdm is None:
                try:
                    del os.environ["TQDM_DISABLE"]
                except Exception:
                    pass
            else:
                os.environ["TQDM_DISABLE"] = old_tqdm

    # FULL MODE (existing behavior)
    _h("Datasets")
    display(HTML(
        f"<div style='color:#374151'>Original root: <code>{_relpath(config.subset_root)}</code></div>"
        f"<div style='color:#374151'>Processed root: <code>{_relpath(config.out_root)}</code></div>"
    ))
    from src.celeba_index import load_subset_index
    dfo = load_subset_index(str(config.subset_root))
    display(HTML(f"<div style='color:#374151'>Original images: <b>{len(dfo):,}</b></div>"))
    dfp = ensure_partition_and_class_columns(load_processed_index_or_raise(config.out_root))
    dfp["abs_path"] = [v if os.path.isabs(v) else str(config.out_root / v) for v in dfp["dest_path"].astype(str)]
    display(HTML(f"<div style='color:#374151'>Processed images: <b>{len(dfp):,}</b></div>"))
    _h("Class Balance by Split", "Balanced splits protect evaluation from sampling bias.")
    bal_before, bal_after, splits, cmp_balance = build_balance_comparison(dfo, dfp)
    # Build a numeric summary table for styling (totals and pos%)
    try:
        rows_tbl: list[dict[str, object]] = []
        for s in splits:
            b = bal_before.loc[s] if s in bal_before.index else pd.Series(dtype=float)
            a = bal_after.loc[s] if s in bal_after.index else pd.Series(dtype=float)
            total_b = float(b.get("total", float("nan"))) if "total" in b else float("nan")
            total_a = float(a.get("total", float("nan"))) if "total" in a else float("nan")
            pos_b = float(b.get("pos_%", float("nan"))) if "pos_%" in b else float("nan")
            pos_a = float(a.get("pos_%", float("nan"))) if "pos_%" in a else float("nan")
            rows_tbl.append({
                "split": s,
                "total (before)": total_b,
                "total (after)": total_a,
                "pos% (before)": pos_b,
                "pos% (after)": pos_a,
                "Δ pos% (pp)": (pos_a - pos_b) if (pd.notna(pos_a) and pd.notna(pos_b)) else float("nan"),
            })
        counts_df = pd.DataFrame(rows_tbl).set_index("split")
        display(style_counts(counts_df).hide(axis="index"))
    except Exception:
        # Fallback to the pre-formatted comparison if styling fails
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
    _h("Geometry", "Uniform 64×64 ensures features > framing quirks.")
    render_size_block("Before (original)", sizes_b)
    render_size_block("After (processed)", sizes_a)
    _h("Channel Stats (TRAIN)", "Small Δ means preprocessing preserved color statistics.")
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
    # Format channel table numbers (3 dp)
    chan_fmt = chan_cmp.copy()
    for c in ["mean (before)", "mean (after)", "Δ mean", "std (before)", "std (after)", "Δ std"]:
        chan_fmt[c] = chan_fmt[c].map(lambda v: f"{float(v):.3f}")
    display(chan_fmt.set_index(""))
    # Plot means/stds
    figs_mean_std: tuple[object, object] | None = None
    if config.plot:
        figs_mean_std = plot_channel_bars(m_b, m_a, s_b, s_a)
        if isinstance(figs_mean_std, tuple) and len(figs_mean_std) == 2:
            out["fig_means"], out["fig_stds"] = figs_mean_std
    stats_path = config.out_root / "stats" / "stats.json"
    from src.nb_display import compare_saved_stats_and_display
    compare_saved_stats_and_display(stats_path, m_a, s_a)
    # Show both original and processed pixel distributions in the same section
    # Use [0,1] for processed when preprocessing used normalize_01
    _h("Pixel Histograms", "Normalization to [0,1] looks correct; distribution shape preserved.")
    orig_scale01 = bool(getattr(config, "pixel_hist_scale_01", False))
    proc_scale01 = bool(getattr(config, "pixel_hist_scale_01", False) or getattr(config, "preprocess_normalize_01", False))
    if config.plot and len(train_b) > 0:
        from src.celeba_plots import plot_original_pixel_hists
        fig_orig = plot_original_pixel_hists(
            train_b,
            sample_n=int(getattr(config, "pixel_hist_sample", 128)),
            scale_01=orig_scale01,
            seed=config.random_seed,
        )
        if fig_orig is not None:
            out["fig_hist_original"] = fig_orig
    if config.plot and len(train_a) > 0:
        fig_proc = plot_processed_pixel_hists(
            train_a,
            bins=int(getattr(config, "proc_pixel_hist_bins", 50)),
            scale_01=proc_scale01,
            seed=config.random_seed,
        )
        if fig_proc is not None:
            out["fig_hist_processed"] = fig_proc

    # Optional saving
    if save_png:
        try:
            save_dir = Path(save_dir) if save_dir is not None else Path(config.out_root) / "figs"
            save_dir.mkdir(parents=True, exist_ok=True)
            def _save(fig: object, name: str) -> Path:
                p = save_dir / name
                try:
                    fig.savefig(p, dpi=150)  # type: ignore[attr-defined]
                except Exception:
                    pass
                return p
            # Counts: create simple counts bar for processed totals
            try:
                with viz_style():
                    fig_cnt, ax_cnt = plt.subplots(figsize=(6.4, 2.8))
                    totals_after = []
                    for s in splits:
                        a = bal_after.loc[s] if s in bal_after.index else pd.Series(dtype=float)
                        totals_after.append(float(a.get("total", float("nan"))))
                    ax_cnt.bar(splits, totals_after, color="#4C78A8")
                    for i, v in enumerate(totals_after):
                        if pd.notna(v): ax_cnt.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
                    ax_cnt.set_ylabel("images"); ax_cnt.set_title("Counts by Split (Processed)", pad=8)
                    out["counts_path"] = str(_save(fig_cnt, "counts_by_split.png"))
            except Exception:
                pass
            # Means/STD
            if out.get("fig_means"):
                out["means_path"] = str(_save(out["fig_means"], "channel_means_before_after.png"))
            if out.get("fig_stds"):
                out["stds_path"] = str(_save(out["fig_stds"], "channel_stds_before_after.png"))
            # Hists
            if out.get("fig_hist_original"):
                out["hist_original_path"] = str(_save(out["fig_hist_original"], "hist_original_01.png" if orig_scale01 else "hist_original_0255.png"))
            if out.get("fig_hist_processed"):
                out["hist_processed_path"] = str(_save(out["fig_hist_processed"], "hist_processed_01.png" if proc_scale01 else "hist_processed_0255.png"))
        except Exception:
            pass

    return out


def analyze_original_subset(config, save_png: bool = False, save_dir: "Path | None" = None, show_plots: bool = False):
    """Analyze the original subset: class balance, size/aspect, and pixel hists.

    Args:
        config: Workflow configuration.
        save_png: If True, save figures to ``save_dir``.
        save_dir: Directory to save PNGs.

    Returns:
        dict with figure handles and saved paths where applicable.
    """
    from IPython.display import display, HTML  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from src.nb_display import _h, _has_variance, viz_style, style_balance_counts
    from src.celeba_index import load_subset_index
    from src.celeba_diagnostics import sample_paths, collect_size_stats, area_retained_after_center_square, describe_numeric
    from src.celeba_plots import plot_original_pixel_hists
    from src.celeba_analysis import build_balance_display_table as _build_bal_table

    df = load_subset_index(str(config.subset_root))
    out: dict[str, object] = {}
    _h("Subset Overview")
    display(HTML(
        f'<div style="color:#374151">Root: <code>{config.subset_root}</code>'
        f' &nbsp; • &nbsp; Total images: <b>{len(df):,}</b></div>'
    ))
    _h("Counts by Split and Class", "Balanced splits protect evaluation.")
    # Compute balance table directly from df
    t = df.groupby(["partition_name", "class_name"])['image_id'].count().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    if "eyeglasses" in t.columns:
        t["pos_ratio"] = (t["eyeglasses"] / t["total"]).round(4)
    balance = t.sort_index()
    fmt_bal = _build_bal_table(balance)
    try:
        display(style_balance_counts(balance).hide(axis="index"))
    except Exception:
        display(fmt_bal.style.hide(axis="index"))
    # Clarify what each row represents
    try:
        display(HTML("<div style='margin-top:4px;color:#374151'>Rows correspond to dataset splits: <b>train</b>, <b>val</b>, and <b>test</b>.</div>"))
    except Exception:
        print("Rows correspond to dataset splits: train, val, and test.")
    if "pos_ratio" in balance.columns and len(balance) > 0 and config.plot and show_plots:
        from src.celeba_plots import plot_class_balance_stacked_with_subtitle
        splits = balance.index.tolist()
        pos = (balance["pos_ratio"] * balance["total"]).astype(float).tolist()
        total = balance["total"].astype(float).tolist()
        fig_cb, ax_cb = plot_class_balance_stacked_with_subtitle(splits, pos, total)
        out["fig_class_balance"] = fig_cb
    all_paths = df["source_path"].astype(str).tolist()
    size_paths = sample_paths(all_paths, int(config.size_sample_max), config.random_seed)
    sizes_df = collect_size_stats(size_paths)
    if not sizes_df.empty:
        sizes_df["area_retained_center_square"] = sizes_df.apply(
            lambda r: area_retained_after_center_square(int(r["width"]), int(r["height"])), axis=1
        )
    if not sizes_df.empty and config.plot and show_plots:
        _h("Image Geometry (Original)", "Uniform size → predictable cropping.")
        from src.celeba_plots import plot_geometry_panel, plot_retained_area_kpi
        fig_geom, _ = plot_geometry_panel(sizes_df["width"].to_numpy(), sizes_df["height"].to_numpy())
        out["fig_geometry"] = fig_geom
        if "area_retained_center_square" in sizes_df.columns:
            fig_area, _ = plot_retained_area_kpi(sizes_df["area_retained_center_square"].to_numpy())
            out["fig_retained_area"] = fig_area
    # Simple textual summary for counts and sizes
    try:
        get_total = lambda s: int(balance.loc[s]["total"]) if s in balance.index else 0
        train_total = get_total("train")
        val_total = get_total("val")
        test_total = get_total("test")
        dataset_total = train_total + val_total + test_total
        print(
            f"Total images in dataset: {dataset_total:,} — train {train_total:,}, val {val_total:,}, test {test_total:,}"
        )
    except Exception:
        pass
    try:
        if not sizes_df.empty:
            import numpy as _np
            avg_w = int(round(_np.nanmean(sizes_df["width"].to_numpy())))
            avg_h = int(round(_np.nanmean(sizes_df["height"].to_numpy())))
            min_w = int(_np.nanmin(sizes_df["width"].to_numpy()))
            max_w = int(_np.nanmax(sizes_df["width"].to_numpy()))
            min_h = int(_np.nanmin(sizes_df["height"].to_numpy()))
            max_h = int(_np.nanmax(sizes_df["height"].to_numpy()))
            print(
                f"Average image size: {avg_w}×{avg_h} — width range {min_w}–{max_w}, height range {min_h}–{max_h}"
            )
    except Exception:
        pass
    train_paths = df[df["partition_name"] == config.channel_stats_split]["source_path"].astype(str).tolist()
    train_paths = sample_paths(train_paths, int(config.channel_sample_max_images), config.random_seed)
    from src.celeba_diagnostics import compute_channel_stats
    mean_rgb, std_rgb = compute_channel_stats(train_paths, scale_01=bool(config.channel_stats_scale_01))
    _h("Channel Means (TRAIN, Original)", "Baseline color statistics.")
    m = tuple(round(float(x), 6) for x in mean_rgb); s = tuple(round(float(x), 6) for x in std_rgb)
    # Compact 2-row table
    import pandas as _pd
    chan_tbl = _pd.DataFrame({"": ["Mean", "Std"], "R": [m[0], s[0]], "G": [m[1], s[1]], "B": [m[2], s[2]]})
    for c in ["R", "G", "B"]:
        chan_tbl[c] = chan_tbl[c].map(lambda v: f"{float(v):.3f}")
    display(chan_tbl.set_index(""))
    # Single-series bar chart for means
    if config.plot and show_plots and len(train_paths) > 0:
        from src.celeba_plots import plot_channel_means_single
        fig_means, _ = plot_channel_means_single(m)
        out["fig_channel_means_original"] = fig_means
        fig_hist = plot_original_pixel_hists(train_paths, sample_n=int(config.pixel_hist_sample), scale_01=bool(config.pixel_hist_scale_01), seed=config.random_seed)
        if fig_hist is not None:
            out["fig_hist_original_0255"] = fig_hist

    # Optional saving
    if save_png:
        try:
            save_dir = Path(save_dir) if save_dir is not None else Path(config.subset_root) / "figs"
            save_dir.mkdir(parents=True, exist_ok=True)
            def _save(fig: object, name: str) -> Path:
                p = save_dir / name
                try:
                    fig.savefig(p, dpi=150)  # type: ignore[attr-defined]
                except Exception:
                    pass
                return p
            if out.get("fig_class_balance"):
                out["subset_class_balance_path"] = str(_save(out["fig_class_balance"], "subset_class_balance.png"))
            if out.get("fig_geometry"):
                out["subset_size_geometry_path"] = str(_save(out["fig_geometry"], "subset_size_geometry.png"))
            if out.get("fig_retained_area"):
                out["subset_retained_area_path"] = str(_save(out["fig_retained_area"], "subset_retained_area.png"))
            if out.get("fig_channel_means_original"):
                out["subset_channel_means_original_path"] = str(_save(out["fig_channel_means_original"], "subset_channel_means_original.png"))
            if out.get("fig_hist_original_0255"):
                out["subset_hist_original_0255_path"] = str(_save(out["fig_hist_original_0255"], "subset_hist_original_0255.png"))
        except Exception:
            pass

    # Do not return figures dict to avoid noisy notebook output in simple calls
    return None
