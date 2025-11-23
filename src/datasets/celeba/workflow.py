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
    # Use builder API
    from .io import load_archive_paths, load_archive_data
    from .builder import build_subset as build_subset_core

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
            from .index import read_processed_index_csv  # type: ignore
        except Exception:
            pass
        # Lightweight inline summary from src (counts by split/class)
        try:
            from collections import Counter as _Counter
            from .index import load_subset_index as _load_subset_index
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

    Captures stdout/stderr from the builder CLI, parses key information, and displays
    a clean, formatted summary instead of raw CLI output.

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

    import io, contextlib, os, csv, re
    from typing import List, Tuple, Dict, Optional
    from IPython.display import display, HTML  # type: ignore

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

    # Build a display argv with repo-relative paths for logging (suppress in notebook context)
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        from ...notebooks.display import _relpath as _rp
        display_argv: List[str] = list(argv)
        for i in range(len(display_argv) - 1):
            if display_argv[i] in {"--archive-dir", "--images-root", "--output-dir"}:
                display_argv[i + 1] = _rp(display_argv[i + 1])
    except Exception:
        display_argv = argv

    # Suppress verbose INFO log in notebook context (can be enabled via logger level if needed)
    # logger.info("Running build-subset (captured): %s", " ".join(["celeba_build_subset"] + display_argv))
    
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        rc = build_subset_cli_main(argv)
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()

    # Convert absolute project paths in captured output to repo-relative
    try:
        cwd = Path.cwd().resolve()
        repo_root = None
        for candidate in [cwd] + list(cwd.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists() or (candidate / "src").exists():
                repo_root = candidate.resolve()
                break
        if repo_root is not None:
            base = re.escape(str(repo_root))
            pattern = re.compile(base + r"/")
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

    # Parse split information from output
    split_info: Dict[str, Dict[str, int]] = {}  # {split_name: {"pos": int, "neg": int, "total": int}}
    split_pattern = re.compile(r"Split (\w+): target per-class cap (\d+)")
    kept_pattern = re.compile(r"  Kept (\d+) pos, (\d+) neg \(total (\d+)\)")
    
    current_split: Optional[str] = None
    for line in out.splitlines():
        split_match = split_pattern.match(line)
        if split_match:
            current_split = split_match.group(1)
        kept_match = kept_pattern.match(line)
        if kept_match and current_split:
            split_info[current_split] = {
                "pos": int(kept_match.group(1)),
                "neg": int(kept_match.group(2)),
                "total": int(kept_match.group(3))
            }

    # Parse output directory and index CSV path
    index_csv_path: Optional[str] = None
    output_dir_display: Optional[str] = None
    for line in out.splitlines():
        if "Wrote index CSV:" in line:
            try:
                index_csv_path = line.split(":", 1)[1].strip()
            except Exception:
                pass
        if "Created files under:" in line:
            try:
                output_dir_display = line.split(":", 1)[1].strip()
            except Exception:
                pass

    # Calculate totals
    total_images = sum(info["total"] for info in split_info.values())
    
    # Format split summary
    split_parts = []
    for split_name in ["train", "val", "test"]:
        if split_name in split_info:
            info = split_info[split_name]
            split_parts.append(
                f"{split_name.capitalize()}: {info['total']:,} ({info['pos']:,}/{info['neg']:,})"
            )
    splits_str = " | ".join(split_parts)

    # Display clean summary
    print(f"- Built balanced subset: {total_images:,} images")
    print(f"- {splits_str}")

    # Show output directory
    if output_dir_display:
        try:
            from ...notebooks.display import _relpath
            output_dir_display = _relpath(output_dir_display)
        except Exception:
            pass
        print(f"- Output: {output_dir_display}")

    # Write CSV of skipped images and display count
    if skipped and not getattr(config, "dry_run", False):
        try:
            csv_path = Path(config.output_dir) / skip_csv_name
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "source_path"]) 
                for image_id, path in skipped:
                    w.writerow([image_id, path])
            from ...notebooks.display import _relpath
            csv_path_rel = _relpath(csv_path)
            print(f"- Skipped: {len(skipped):,} missing images (see {csv_path_rel})")
        except Exception as e:
            logger.warning("Failed to write skipped images CSV: %s", e)
    elif skipped:
        # In dry-run, still report count
        print(f"- Skipped: {len(skipped):,} missing images (dry-run)")

    # Print errors if any
    if err:
        print(err, end="")

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
    from .preprocess import preprocess_subset
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
    """Display a clean summary of processed images using HTML formatting."""
    from IPython.display import display, HTML  # type: ignore
    total = sum(by_split.values()) if by_split else 0
    
    # Format splits
    splits_str = " | ".join([f"{k.capitalize()}: {v:,}" for k, v in sorted(by_split.items())])
    
    # Format classes
    classes_str = " | ".join([f"{k}: {v:,}" for k, v in sorted(by_class.items())])
    
    # Get relative path
    try:
        from ...notebooks.display import _relpath as _rp
        root_display = _rp(out_root)
    except Exception:
        root_display = str(out_root)
    
    # Display clean summary
    display(HTML(
        f'<div style="margin:8px 0;color:#065f46;font-size:13px">'
        f'✓ Preprocessed <b>{total:,}</b> images'
        f'</div>'
    ))
    
    display(HTML(
        f'<div style="margin:4px 0 8px 0;color:#374151;font-size:12px">'
        f'{splits_str} &nbsp; • &nbsp; {classes_str}'
        f'</div>'
    ))
    
    display(HTML(
        f'<div style="margin:4px 0;color:#6b7280;font-size:11px">'
        f'Output: <code>{root_display}</code>'
        f'</div>'
    ))



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
            from .diagnostics import select_visual_paths, compute_average_original_and_cropped
            from .plots import plot_average_and_diff, plot_center_crop_overlays
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
        from .index import (
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
        from .diagnostics import select_visual_paths, compute_average_original_and_cropped
        from .plots import plot_average_and_diff, plot_center_crop_overlays
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
    """Run the preprocessing CLI and summarize processed index (no diagnostics).
    
    Displays a clean, formatted summary of preprocessing results matching analysis cell format.
    """
    import os
    from .diagnostics import sample_paths, collect_size_stats
    
    # Suppress tqdm progress bars during preprocessing
    old_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        argv = build_preprocess_argv(config)
        rc = run_preprocess_cli(argv)
        if rc != 0:
            logger.error("Preprocess failed with exit code %s", rc)
            return
        
        # Get summary information
        try:
            from .index import (
                augment_processed_index_with_sizes,
                read_processed_index_csv,
                summarize_processed_index_df,
                ensure_partition_and_class_columns,
            )
            from PIL import Image
            
            augment_processed_index_with_sizes(config.out_root)
            dfp = ensure_partition_and_class_columns(read_processed_index_csv(config.out_root))
            dfp["abs_path"] = [v if os.path.isabs(v) else str(config.out_root / v) for v in dfp["dest_path"].astype(str)]
            
            by_split, by_class = summarize_processed_index_df(dfp)
            total_images = int(sum(by_split.values())) if by_split else 0
            train_n = int(by_split.get("train", 0))
            val_n = int(by_split.get("val", 0))
            test_n = int(by_split.get("test", 0))
            
            # Geometry
            target_size = int(getattr(config, "preprocess_size", 64))
            size_paths = sample_paths(dfp["abs_path"].astype(str).tolist(), min(100, len(dfp)), config.random_seed)
            sizes_df = collect_size_stats(size_paths)
            geom_uniform = False
            if not sizes_df.empty:
                widths = sizes_df["width"].dropna().astype(int).unique().tolist()
                heights = sizes_df["height"].dropna().astype(int).unique().tolist()
                geom_uniform = (len(widths) == 1 and len(heights) == 1 and widths[0] == target_size and heights[0] == target_size)
            
            # Format
            img_format = "Unknown"
            img_mode = "Unknown"
            if size_paths:
                try:
                    with Image.open(size_paths[0]) as img:
                        img_format = img.format or "Unknown"
                        img_mode = img.mode
                except Exception:
                    pass
            
            # Simple output matching analysis cells format
            def _kfmt(n: int) -> str:
                try:
                    return f"{int(n/1000)}k" if n % 1000 == 0 and n >= 1000 else f"{n:,}"
                except Exception:
                    return str(n)
            
            counts_msg = f"• Counts: {_kfmt(total_images)} total images ({_kfmt(train_n)} train, {_kfmt(val_n)} val, {_kfmt(test_n)} test) | 50/50 class balance"
            geometry_msg = f"• Geometry: {target_size}×{target_size} pixels ({'uniform' if geom_uniform else 'mixed'})"
            
            format_str = f"{img_format}"
            if img_mode == "RGB":
                format_str += " (RGB, 3 bands)"
            elif img_mode != "Unknown":
                format_str += f" ({img_mode})"
            format_msg = f"• Format: {format_str}"
            
            print(
                "\n".join([
                    counts_msg,
                    geometry_msg,
                    format_msg,
                ])
            )
        except Exception as e:
            logger.warning("Could not summarize processed index: %s", e)
    finally:
        # restore tqdm behavior
        if old_tqdm is None:
            try:
                del os.environ["TQDM_DISABLE"]
            except Exception:
                pass
        else:
            os.environ["TQDM_DISABLE"] = old_tqdm


def review_archive(config) -> None:
    """Review the CelebA archive: validate files, show splits and attribute balance.
    
    Streamlined version with reduced visual clutter while preserving essential information.

    Side effects:
        - Displays compact validation and tables/plots
        - Writes balance plots and CSV under archive_dir
    """
    # Local imports to avoid unnecessary top-level dependencies
    import os
    from IPython.display import display, HTML  # type: ignore
    from .io import load_archive_paths, load_archive_data
    from ...notebooks.display import (
        _h,
        PREFERRED_FOCUS,
        highlight_drift,
    )
    from .analysis import (
        compute_attribute_summary,
        build_focus_table,
        compute_balance_from_df as compute_balance,
    )
    from .plots import plot_attribute_overall

    # Load paths and validate
    attrs_csv, parts_csv, bboxes_csv, landmarks_csv = load_archive_paths(config.archive_dir)

    # Streamlined validation: single line if all files found
    required_files = [
        ("list_attr_celeba.csv", attrs_csv),
        ("list_eval_partition.csv", parts_csv),
    ]
    optional_files = [
        ("list_bbox_celeba.csv", bboxes_csv),
        ("list_landmarks_align_celeba.csv", landmarks_csv),
    ]

    all_required_found = all(os.path.isfile(path) for _, path in required_files)
    all_optional_found = all(os.path.isfile(path) for _, path in optional_files)

    if all_required_found:
        status_text = "- All required files found"
        if all_optional_found:
            status_text += " (including optional files)"
        print(status_text)
    else:
        # Show detailed badges only if something is missing
        from ...notebooks.display import render_validation_badges
        render_validation_badges(attrs_csv, parts_csv, bboxes_csv, landmarks_csv)

    # Load data
    df = load_archive_data(attrs_csv, parts_csv)
    num_rows = df.shape[0]
    all_attr_cols = [c for c in df.columns if c not in ("image_id", "partition", "partition_name")]

    # Compact metadata + splits in one line
    split_counts = df["partition_name"].value_counts().sort_index()
    splits_str = " | ".join([f"{k.capitalize()}: {v:,}" for k, v in split_counts.items()])
    print(f"\n- {num_rows:,} images, {len(all_attr_cols)} attributes   •   {splits_str}")

    # Reduced attribute balance chart (top 10 instead of config.plot_top_n_attrs)
    _h("Attribute balance", "4", "Top 10 attributes by positive fraction")
    summary_all = compute_attribute_summary(df)
    plot_attribute_overall(summary_all, top_n=10)  # Reduced from default 20

    # Streamlined focused attributes table: key columns only
    _h("Focused attributes", "4", "Target attribute (Eyeglasses) and related attributes")
    focus_attrs = [a for a in PREFERRED_FOCUS if a in all_attr_cols] or all_attr_cols[:6]
    summary_focus = compute_balance(df, focus_attrs).copy()

    # Build table but show only essential columns
    focus_tbl = build_focus_table(summary_focus)
    focus_tbl = focus_tbl.sort_values("Overall %", ascending=False).reset_index(drop=True)

    # Select only key columns for display
    key_cols = ["Attribute", "Overall %", "Train %", "Val %", "Test %", "Δ Train (pp)", "Δ Val (pp)", "Δ Test (pp)"]
    display_cols = [col for col in key_cols if col in focus_tbl.columns]
    streamlined_tbl = focus_tbl[display_cols].copy()

    # Apply styling
    fmt = {
        "Overall %": "{:.1f}%", "Train %": "{:.1f}%", "Val %": "{:.1f}%", "Test %": "{:.1f}%",
        "Δ Train (pp)": "{:+.1f}", "Δ Val (pp)": "{:+.1f}", "Δ Test (pp)": "{:+.1f}",
    }
    styler = (
        streamlined_tbl.style
        .format({k: v for k, v in fmt.items() if k in streamlined_tbl.columns})
        .hide(axis="index")
    )

    # Highlight Eyeglasses row with green outline if present
    if "Eyeglasses" in streamlined_tbl["Attribute"].values:
        eyeglasses_idx = streamlined_tbl[streamlined_tbl["Attribute"] == "Eyeglasses"].index[0]
        # Add green border to the entire row
        styler = styler.set_table_styles([
            {
                'selector': f'tbody tr:nth-child({eyeglasses_idx + 1})',
                'props': [('border-top', '2px solid #10b981'), ('border-bottom', '2px solid #10b981')]
            },
            {
                'selector': f'tbody tr:nth-child({eyeglasses_idx + 1}) td:first-child',
                'props': [('border-left', '2px solid #10b981')]
            },
            {
                'selector': f'tbody tr:nth-child({eyeglasses_idx + 1}) td:last-child',
                'props': [('border-right', '2px solid #10b981')]
            }
        ], overwrite=False)

    # Highlight drift columns
    for col in ["Δ Train (pp)", "Δ Val (pp)", "Δ Test (pp)"]:
        if col in streamlined_tbl.columns:
            styler = styler.apply(highlight_drift, subset=[col])

    display(styler)

    # Write outputs (footer handled by write_archive_outputs)
    from .io import write_archive_outputs
    write_archive_outputs(summary_all, config.archive_dir)
    logger.info("Archive review complete.")


def compute_channel_stats_for_paths(paths: list[str], scale_01: bool):
    from .diagnostics import compute_channel_stats
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
    from ...notebooks.display import _h
    from .plots import (
        plot_channel_bars,
        plot_processed_pixel_hists,
        render_size_block,
    )
    from .index import (
        read_processed_index_csv,
        augment_processed_index_with_sizes,
        summarize_processed_index_df,
        load_processed_index_or_raise,
        ensure_partition_and_class_columns,
    )
    from .diagnostics import sample_paths, collect_size_stats
    from .plots import plot_grouped_bars_two_series
    from .analysis import build_balance_comparison

    from ...notebooks.display import _relpath, viz_style, style_counts, add_bar_labels
    out: dict[str, object] = {}

    # Brief mode: compact summary + two plots, optional drift table, quiet progress
    if verbosity == "brief":
        # Suppress tqdm progress bars where used
        old_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            # Load indices
            from .index import load_subset_index
            dfo = load_subset_index(str(config.subset_root))
            dfp = ensure_partition_and_class_columns(load_processed_index_or_raise(config.out_root))
            dfp["abs_path"] = [v if os.path.isabs(v) else str(config.out_root / v) for v in dfp["dest_path"].astype(str)]

            # Counts and balance
            by_split, by_class = summarize_processed_index_df(dfp)
            total_images = int(sum(by_split.values())) if by_split else 0
            train_n = int(by_split.get("train", 0))
            val_n = int(by_split.get("val", 0))
            test_n = int(by_split.get("test", 0))

            # Original subset counts
            try:
                by_split_o, _ = summarize_processed_index_df(dfo)
                total_images_o = int(sum(by_split_o.values())) if by_split_o else len(dfo)
            except Exception:
                total_images_o = len(dfo)

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
            geom_after = f"{target_size}×{target_size}"
            try:
                if not sizes_a.empty:
                    widths = sizes_a["width"].dropna().astype(int).unique().tolist()
                    heights = sizes_a["height"].dropna().astype(int).unique().tolist()
                    geom_uniform = (len(widths) == 1 and len(heights) == 1 and widths[0] == target_size and heights[0] == target_size)
                    if not geom_uniform:
                        w_range = f"{min(widths)}-{max(widths)}" if len(widths) > 1 else str(widths[0])
                        h_range = f"{min(heights)}-{max(heights)}" if len(heights) > 1 else str(heights[0])
                        geom_after = f"{w_range}×{h_range}"
            except Exception:
                geom_uniform = False

            # Geometry (original) - sample original images
            geom_before = "Unknown"
            try:
                orig_paths = dfo["source_path"].astype(str).tolist()
                size_paths_o = sample_paths(orig_paths, min(100, len(orig_paths)), config.random_seed)
                sizes_o = collect_size_stats(size_paths_o)
                if not sizes_o.empty:
                    widths_o = sizes_o["width"].dropna().astype(int).unique().tolist()
                    heights_o = sizes_o["height"].dropna().astype(int).unique().tolist()
                    if len(widths_o) == 1 and len(heights_o) == 1:
                        geom_before = f"{widths_o[0]}×{heights_o[0]}"
                    else:
                        w_range_o = f"{min(widths_o)}-{max(widths_o)}" if len(widths_o) > 1 else str(widths_o[0])
                        h_range_o = f"{min(heights_o)}-{max(heights_o)}" if len(heights_o) > 1 else str(heights_o[0])
                        geom_before = f"{w_range_o}×{h_range_o}"
            except Exception:
                pass

            # Skip channel stats computation in brief mode - not needed for educational notebook
            # This also eliminates progress bars and unnecessary computation

            # Get image format and mode info from samples (before and after)
            img_format_before = "Unknown"
            img_mode_before = "Unknown"
            img_format_after = "Unknown"
            img_mode_after = "Unknown"
            
            # Original format
            try:
                from PIL import Image
                if len(dfo) > 0:
                    orig_sample_path = dfo["source_path"].iloc[0] if "source_path" in dfo.columns else None
                    if orig_sample_path:
                        orig_abs = orig_sample_path if os.path.isabs(str(orig_sample_path)) else str(config.subset_root / orig_sample_path)
                        with Image.open(orig_abs) as img:
                            img_format_before = img.format or "Unknown"
                            img_mode_before = img.mode
            except Exception:
                pass
            
            # Processed format
            sample_paths_for_format = size_paths_a if size_paths_a else (dfp["abs_path"].astype(str).tolist()[:1] if len(dfp) > 0 else [])
            if sample_paths_for_format:
                try:
                    sample_img_path = sample_paths_for_format[0]
                    with Image.open(sample_img_path) as img:
                        img_format_after = img.format or "Unknown"
                        img_mode_after = img.mode
                except Exception:
                    pass

            # Summary cards (compact) - Before/After comparison
            def _kfmt(n: int) -> str:
                try:
                    return f"{int(n/1000)}k" if n % 1000 == 0 and n >= 1000 else f"{n:,}"
                except Exception:
                    return str(n)
            
            # Build before/after comparison messages
            print("Before → After Preprocessing:")
            print("")
            
            # Counts comparison
            counts_before_str = f"{_kfmt(total_images_o)} total"
            counts_after_str = f"{_kfmt(total_images)} total ({_kfmt(train_n)} train, {_kfmt(val_n)} val, {_kfmt(test_n)} test)"
            counts_msg = f"• Counts: {counts_before_str} → {counts_after_str} | 50/50 class balance"
            
            # Geometry comparison
            geometry_msg = f"• Geometry: {geom_before} → {geom_after} pixels ({'uniform' if geom_uniform else 'mixed'})"
            
            # Format comparison
            format_before_str = f"{img_format_before}"
            if img_mode_before == "RGB":
                format_before_str += " (RGB)"
            elif img_mode_before != "Unknown":
                format_before_str += f" ({img_mode_before})"
            
            format_after_str = f"{img_format_after}"
            if img_mode_after == "RGB":
                format_after_str += " (RGB, 3 bands)"
            elif img_mode_after != "Unknown":
                format_after_str += f" ({img_mode_after})"
            
            format_msg = f"• Format: {format_before_str} → {format_after_str}"
            
            print(
                "\n".join([
                    counts_msg,
                    geometry_msg,
                    format_msg,
                ])
            )

            # Optional compact table if drift exceeds threshold
            if counts_df is not None and show_table and config.plot:
                try:
                    display(style_counts(counts_df).hide(axis="index"))
                except Exception:
                    display(counts_df)

            # Skip detailed visualizations in brief mode - they're not educational for DP-SGD training focus
            # The text summary (counts, geometry) is sufficient for validation
            # Return None to avoid displaying dict in notebook
            return None
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
    from .index import load_subset_index
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
    from .diagnostics import collect_size_stats
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
    from ...notebooks.display import compare_saved_stats_and_display
    compare_saved_stats_and_display(stats_path, m_a, s_a)
    # Show both original and processed pixel distributions in the same section
    # Use [0,1] for processed when preprocessing used normalize_01
    _h("Pixel Histograms", "Normalization to [0,1] looks correct; distribution shape preserved.")
    orig_scale01 = bool(getattr(config, "pixel_hist_scale_01", False))
    proc_scale01 = bool(getattr(config, "pixel_hist_scale_01", False) or getattr(config, "preprocess_normalize_01", False))
    if config.plot and len(train_b) > 0:
        from .plots import plot_original_pixel_hists
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


def analyze_original_subset(
    config, 
    save_png: bool = False, 
    save_dir: "Path | None" = None, 
    show_plots: bool = False,
    verbosity: Literal["brief", "full"] = "brief",
):
    """Analyze the original subset: streamlined CV-focused metrics for training preparation.

    Args:
        config: Workflow configuration.
        save_png: If True, save figures to ``save_dir``.
        save_dir: Directory to save PNGs.
        show_plots: If True, display optional plots.
        verbosity: "brief" for compact summary, "full" for detailed analysis.

    Returns:
        dict with figure handles and saved paths where applicable, or None in brief mode.
    """
    from IPython.display import display, HTML  # type: ignore
    import os
    import numpy as np
    from pathlib import Path
    from ...notebooks.display import _h
    from .index import load_subset_index, summarize_processed_index_df
    from .diagnostics import sample_paths, collect_size_stats
    from .plots import plot_original_pixel_hists
    from .diagnostics import compute_channel_stats

    df = load_subset_index(str(config.subset_root))
    out: dict[str, object] = {}

    # Brief mode: compact summary matching analyze_processed format
    if verbosity == "brief":
        # Suppress tqdm progress bars
        old_tqdm = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            # Counts
            by_split, by_class = summarize_processed_index_df(df)
            total_images = int(sum(by_split.values())) if by_split else len(df)
            train_n = int(by_split.get("train", 0)) if by_split else 0
            val_n = int(by_split.get("val", 0)) if by_split else 0
            test_n = int(by_split.get("test", 0)) if by_split else 0

            # Geometry
            all_paths = df["source_path"].astype(str).tolist()
            size_paths = sample_paths(all_paths, min(int(config.size_sample_max), len(all_paths)), config.random_seed)
            sizes_df = collect_size_stats(size_paths)
            geom_str = "Unknown"
            geom_uniform = False
            if not sizes_df.empty:
                widths = sizes_df["width"].dropna().astype(int).unique().tolist()
                heights = sizes_df["height"].dropna().astype(int).unique().tolist()
                if len(widths) == 1 and len(heights) == 1:
                    geom_str = f"{widths[0]}×{heights[0]}"
                    geom_uniform = True
                else:
                    w_range = f"{min(widths)}-{max(widths)}" if len(widths) > 1 else str(widths[0])
                    h_range = f"{min(heights)}-{max(heights)}" if len(heights) > 1 else str(heights[0])
                    geom_str = f"{w_range}×{h_range}"

            # Format
            img_format = "Unknown"
            img_mode = "Unknown"
            if len(df) > 0:
                try:
                    from PIL import Image
                    sample_path = df["source_path"].iloc[0] if "source_path" in df.columns else None
                    if sample_path:
                        abs_path = sample_path if os.path.isabs(str(sample_path)) else str(config.subset_root / sample_path)
                        with Image.open(abs_path) as img:
                            img_format = img.format or "Unknown"
                            img_mode = img.mode
                except Exception:
                    pass

            # Summary output (matching analyze_processed format)
            def _kfmt(n: int) -> str:
                try:
                    return f"{int(n/1000)}k" if n % 1000 == 0 and n >= 1000 else f"{n:,}"
                except Exception:
                    return str(n)

            counts_msg = f"• Counts: {_kfmt(total_images)} total images ({_kfmt(train_n)} train, {_kfmt(val_n)} val, {_kfmt(test_n)} test) | 50/50 class balance"
            geometry_msg = f"• Geometry: {geom_str} pixels ({'uniform' if geom_uniform else 'mixed'})"
            
            format_str = f"{img_format}"
            if img_mode == "RGB":
                format_str += " (RGB, 3 bands)"
            elif img_mode != "Unknown":
                format_str += f" ({img_mode})"
            format_msg = f"• Format: {format_str}"

            print(
                "\n".join([
                    counts_msg,
                    geometry_msg,
                    format_msg,
                ])
            )

            return None
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

    # Sample paths for analysis
    all_paths = df["source_path"].astype(str).tolist()
    size_paths = sample_paths(all_paths, int(config.size_sample_max), config.random_seed)
    sizes_df = collect_size_stats(size_paths)
    
    # Get train paths for channel/pixel stats
    train_paths = df[df["partition_name"] == config.channel_stats_split]["source_path"].astype(str).tolist()
    train_paths = sample_paths(train_paths, int(config.channel_sample_max_images), config.random_seed)
    # === Image Dimensions & Uniformity ===
    _h("Image Dimensions", "Size uniformity affects batching and preprocessing.")
    if not sizes_df.empty:
        widths = sizes_df["width"].to_numpy()
        heights = sizes_df["height"].to_numpy()
        aspects = sizes_df["aspect"].to_numpy()
        
        # Uniformity check
        width_uniform = np.all(widths == widths[0])
        height_uniform = np.all(heights == heights[0])
        is_uniform = width_uniform and height_uniform
        
        if is_uniform:
            size_str = f"<b>{int(widths[0])}×{int(heights[0])}</b> (uniform)"
            status_color = "#065f46"
        else:
            avg_w = int(round(np.nanmean(widths)))
            avg_h = int(round(np.nanmean(heights)))
            min_w, max_w = int(np.nanmin(widths)), int(np.nanmax(widths))
            min_h, max_h = int(np.nanmin(heights)), int(np.nanmax(heights))
            size_str = f"<b>{avg_w}×{avg_h}</b> avg (range: {min_w}–{max_w}×{min_h}–{max_h})"
            status_color = "#92400e"
        
        # Aspect ratio summary
        aspect_min, aspect_max = float(np.nanmin(aspects)), float(np.nanmax(aspects))
        aspect_avg = float(np.nanmean(aspects))
        if abs(aspect_max - aspect_min) < 0.01:
            aspect_str = f"<b>{aspect_avg:.2f}</b> (consistent)"
        else:
            aspect_str = f"<b>{aspect_avg:.2f}</b> (range: {aspect_min:.2f}–{aspect_max:.2f})"
        
        display(HTML(
            f'<div style="margin:8px 0;color:#374151;font-size:13px">'
            f'Size: {size_str} &nbsp; • &nbsp; Aspect ratio: {aspect_str}'
            f'</div>'
        ))
    else:
        display(HTML('<div style="margin:8px 0;color:#92400e;font-size:13px">⚠ Could not read image sizes</div>'))

    # === File Format & Memory Footprint ===
    _h("Data Quality", "Format consistency and storage requirements.")
    if all_paths:
        # Check file formats
        formats = {}
        total_size_bytes = 0
        corrupted_count = 0
        
        sample_for_format = sample_paths(all_paths, min(100, len(all_paths)), config.random_seed)
        for p in sample_for_format:
            try:
                ext = Path(p).suffix.lower()
                formats[ext] = formats.get(ext, 0) + 1
                if os.path.exists(p):
                    total_size_bytes += os.path.getsize(p)
            except Exception:
                corrupted_count += 1
        
        # Estimate total size (scale up from sample)
        if sample_for_format:
            sample_size = sum(os.path.getsize(p) for p in sample_for_format if os.path.exists(p))
            estimated_total_size = int((sample_size / len(sample_for_format)) * len(all_paths))
        else:
            estimated_total_size = 0
        
        # Format summary
        if formats:
            primary_format = max(formats.items(), key=lambda x: x[1])
            format_pct = (primary_format[1] / len(sample_for_format)) * 100
            if format_pct >= 99:
                format_str = f"{primary_format[0][1:].upper()} ({format_pct:.0f}%)"
            else:
                format_str = f"Mixed ({', '.join(f'{k[1:].upper()}: {v/len(sample_for_format)*100:.0f}%' for k, v in sorted(formats.items(), key=lambda x: -x[1])[:2])})"
        else:
            format_str = "Unknown"
        
        # Memory footprint
        def format_bytes(b: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if b < 1024.0:
                    return f"{b:.1f} {unit}"
                b /= 1024.0
            return f"{b:.1f} TB"
        
        display(HTML(
            f'<div style="margin:8px 0;color:#374151;font-size:13px">'
            f'Format: <b>{format_str}</b> &nbsp; • &nbsp; '
            f'Estimated size: <b>{format_bytes(estimated_total_size)}</b>'
            f'</div>'
        ))
        
        if corrupted_count > 0:
            display(HTML(
                f'<div style="margin:4px 0;color:#92400e;font-size:11px">'
                f'⚠ {corrupted_count} files could not be accessed in sample'
                f'</div>'
            ))

    # === Pixel Value Statistics ===
    _h("Pixel Values", "Range and distribution inform normalization strategy.")
    if train_paths:
        try:
            from PIL import Image
            pixel_values = []
            pixel_min, pixel_max = 255, 0
            for p in sample_paths(train_paths, min(50, len(train_paths)), config.random_seed):
                try:
                    with Image.open(p) as img:
                        arr = np.array(img.convert("RGB"))
                        pixel_values.extend(arr.flatten())
                        pixel_min = min(pixel_min, int(arr.min()))
                        pixel_max = max(pixel_max, int(arr.max()))
                except Exception:
                    continue
            
            if pixel_values:
                pixel_values = np.array(pixel_values)
                scale_01 = bool(getattr(config, "channel_stats_scale_01", False))
                if scale_01:
                    range_str = "[0,1]"
                    pixel_min, pixel_max = 0.0, 1.0
                else:
                    range_str = "[0,255]"
                
                display(HTML(
                    f'<div style="margin:8px 0;color:#374151;font-size:13px">'
                    f'Range: <b>{range_str}</b> &nbsp; • &nbsp; '
                    f'Sample min: <b>{pixel_min}</b>, max: <b>{pixel_max}</b>'
                    f'</div>'
                ))
        except Exception:
            pass

    # === Channel Statistics ===
    _h("Channel Statistics (TRAIN)", "RGB means and stds for normalization.")
    if train_paths:
        try:
            mean_rgb, std_rgb = compute_channel_stats(train_paths, scale_01=bool(config.channel_stats_scale_01))
            m = tuple(round(float(x), 3) for x in mean_rgb)
            s = tuple(round(float(x), 3) for x in std_rgb)
            
            display(HTML(
                f'<div style="margin:8px 0;color:#374151;font-size:13px">'
                f'Mean: R=<b>{m[0]}</b> G=<b>{m[1]}</b> B=<b>{m[2]}</b> &nbsp; • &nbsp; '
                f'Std: R=<b>{s[0]}</b> G=<b>{s[1]}</b> B=<b>{s[2]}</b>'
                f'</div>'
            ))
            
            # Optional plots
            if config.plot and show_plots:
                from .plots import plot_channel_means_single
                fig_means, _ = plot_channel_means_single(m)
                out["fig_channel_means_original"] = fig_means
                fig_hist = plot_original_pixel_hists(
                    train_paths, 
                    sample_n=int(getattr(config, "pixel_hist_sample", 100)), 
                    scale_01=bool(getattr(config, "pixel_hist_scale_01", False)), 
                    seed=config.random_seed
                )
                if fig_hist is not None:
                    out["fig_hist_original_0255"] = fig_hist
        except Exception as e:
            display(HTML(f'<div style="margin:8px 0;color:#92400e;font-size:13px">⚠ Could not compute channel stats: {e}</div>'))

    # Optional geometry plots
    if not sizes_df.empty and config.plot and show_plots:
        from .plots import plot_geometry_panel
        fig_geom, _ = plot_geometry_panel(sizes_df["width"].to_numpy(), sizes_df["height"].to_numpy())
        out["fig_geometry"] = fig_geom

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
            if out.get("fig_geometry"):
                out["subset_size_geometry_path"] = str(_save(out["fig_geometry"], "subset_size_geometry.png"))
            if out.get("fig_channel_means_original"):
                out["subset_channel_means_original_path"] = str(_save(out["fig_channel_means_original"], "subset_channel_means_original.png"))
            if out.get("fig_hist_original_0255"):
                out["subset_hist_original_0255_path"] = str(_save(out["fig_hist_original_0255"], "subset_hist_original_0255.png"))
        except Exception:
            pass

    return None
