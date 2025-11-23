"""Notebook display helpers and constants for a cleaner, consistent look.

This module centralizes small HTML helpers, pandas display configuration, and
display-only constants to keep notebooks minimal and focused on analysis.
"""

from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from contextlib import contextmanager

@contextmanager
def viz_style(dpi: int = 150, base: int = 12, title: int = 18, label: int = 12):
    """Temporary Matplotlib style for consistent visuals.

    Usage:
        with viz_style():
            ... plotting code ...
    """
    import matplotlib.pyplot as plt  # lazy import
    old = plt.rcParams.copy()
    try:
        plt.rcParams.update({
            "figure.dpi": dpi,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": title,
            "axes.labelsize": label,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "legend.fontsize": base,
            "figure.constrained_layout.use": True,
        })
        yield
    finally:
        try:
            import matplotlib.pyplot as plt  # re-import to be safe in some envs
            plt.rcParams.update(old)
        except Exception:
            pass

def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return ""

def fmt_float(x, nd: int = 3) -> str:
    try:
        return f"{float(x):.{int(nd)}f}"
    except Exception:
        return ""

def percent_formatter():
    from matplotlib.ticker import PercentFormatter  # lazy import
    return PercentFormatter(xmax=1.0, decimals=2)

def add_bar_labels(ax, fmt=lambda v: f"{v:.3f}") -> None:
    """Add small value labels above bars on the provided axis."""
    for p in getattr(ax, "patches", []):
        try:
            v = p.get_height()
            ax.annotate(
                fmt(v),
                (p.get_x() + p.get_width() / 2.0, v),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )
        except Exception:
            continue

 
def _h(title: str, level: int = 4, subtitle: Optional[str] = None) -> None:
    """Render a simple header block in notebooks."""
    from IPython.display import display, HTML  # lazy import for non-notebook contexts

    # Backward-compatible arg handling: allow _h(title, subtitle) or _h(title, "4", subtitle)
    if isinstance(level, str):
        if subtitle is None:
            subtitle = level
            level = 4
        else:
            try:
                level = int(level)
            except Exception:
                level = 4

    subtitle_html = (
        f'<div style="color:#6b7280;font-size:12px;margin-top:4px">{subtitle}</div>'
        if subtitle
        else ""
    )
    display(
        HTML(
            f"""
    <div style="margin:18px 0 8px 0">
      <div style="font-weight:600;font-size:{18 if level==4 else 22}px">{title}</div>
      {subtitle_html}
    </div>"""
        )
    )


def _badge(text: str, tone: str = "ok") -> str:
    """Return a colored HTML badge string for inline display."""
    colors = {
        "ok": ("#065f46", "#d1fae5"),
        "warn": ("#92400e", "#fef3c7"),
        "error": ("#991b1b", "#fee2e2"),
        "info": ("#1e40af", "#dbeafe"),
    }
    fg, bg = colors.get(tone, colors["info"])
    return (
        f'<span style="padding:2px 8px;border-radius:999px;background:{bg};color:{fg};font-size:12px">{text}</span>'
    )


def _pct(x) -> float:
    """Convert a fraction to percentage, returning NaN on failure."""
    try:
        return float(x) * 100.0
    except Exception:
        return float("nan")


def _fmt_pct(x, decimals: int = 1) -> str:
    """Format a numeric value as a percentage string with given decimals.

    Backward compatible with previous signature where one decimal was assumed.
    """
    try:
        if pd.isna(x):
            return ""
        fmt = f"{{:.{int(decimals)}f}}%"
        return fmt.format(float(x))
    except Exception:
        return ""


def _pp(x) -> str:
    """Format a numeric value already in percentage space as signed p.p."""
    try:
        return f"{float(x):+.1f}"
    except Exception:
        return ""


def _has_variance(series: pd.Series, eps: float = 1e-9) -> bool:
    """Return True if a pandas Series shows nontrivial variance.

    Fast path checks cardinality; falls back to numeric std with jitter threshold.
    """
    s = series.dropna()
    if s.empty:
        return False
    if s.nunique(dropna=True) > 1:
        return True
    try:
        return float(s.astype(float).std()) > eps
    except Exception:
        return False


def highlight_drift(s: pd.Series) -> list[str]:
    """Return style strings to highlight large absolute deltas (>5 p.p.)."""
    out: list[str] = [""] * len(s)
    for i, v in enumerate(s):
        try:
            if pd.notna(v) and abs(float(v)) > 5.0:
                out[i] = "background-color: #38220b; color: #f59e0b"
        except Exception:
            continue
    return out


def bold_extremes(col: pd.Series) -> list[str]:
    """Return style strings that bold the min and max values in a column."""
    out: list[str] = [""] * len(col)
    try:
        vals = col.astype(float)
        out[vals.idxmax()] = "font-weight: 700"
        out[vals.idxmin()] = "font-weight: 700"
    except Exception:
        pass
    return out


# Optional: nicer display names for a few attributes (display only)
ATTR_RENAME: Dict[str, str] = {
    "5_o_Clock_Shadow": "5 o’clock shadow",
    "High_Cheekbones": "High cheekbones",
    "Big_Lips": "Big lips",
    "Big_Nose": "Big nose",
    "Arched_Eyebrows": "Arched eyebrows",
    "Wearing_Lipstick": "Wearing lipstick",
}


# Choose a small focused set (edit to taste)
PREFERRED_FOCUS = [
    "Smiling",
    "Young",
    "Eyeglasses",
    "Male",
    "Blond_Hair",
    "Heavy_Makeup",
]


__all__ = [
    "_h",
    "_badge",
    "_pct",
    "_fmt_pct",
    "_pp",
    "_has_variance",
    "highlight_drift",
    "bold_extremes",
    "_relpath",
    "ATTR_RENAME",
    "PREFERRED_FOCUS",
]


def style_focus_table(focus_tbl: "pd.DataFrame"):
    fmt = {
        "Total": "{:,}", "Pos": "{:,}", "Neg": "{:,}",
        "Overall %": "{:.1f}%", "Train %": "{:.1f}%", "Val %": "{:.1f}%", "Test %": "{:.1f}%",
        "Δ Train (pp)": "{:+.1f}", "Δ Val (pp)": "{:+.1f}", "Δ Test (pp)": "{:+.1f}",
    }
    styler = (
        focus_tbl.style
        .format({k:v for k,v in fmt.items() if k in focus_tbl.columns})
        .hide(axis="index")
        .set_properties(subset=["Total","Pos","Neg"], **{"text-align":"right"})
        .apply(bold_extremes, subset=["Overall %"]) 
    )
    for col in ["Δ Train (pp)","Δ Val (pp)","Δ Test (pp)"]:
        if col in focus_tbl.columns:
            styler = styler.apply(highlight_drift, subset=[col])
    return styler


def render_validation_badges(attrs_csv: str, parts_csv: str, bboxes_csv: str, landmarks_csv: str) -> None:
    import os
    from IPython.display import display, HTML
    val_html = f"""
    <div style=\"display:flex;gap:12px;flex-wrap:wrap;margin-top:8px\">
      <div>{_badge('list_attr_celeba.csv: found','ok') if os.path.isfile(attrs_csv) else _badge('list_attr_celeba.csv: missing','error')}</div>
      <div>{_badge('list_eval_partition.csv: found','ok') if os.path.isfile(parts_csv) else _badge('list_eval_partition.csv: missing','error')}</div>
      <div>{_badge('list_bbox_celeba.csv: found','ok') if os.path.isfile(bboxes_csv) else _badge('list_bbox_celeba.csv: missing','warn')}</div>
      <div>{_badge('list_landmarks_align_celeba.csv: found','ok') if os.path.isfile(landmarks_csv) else _badge('list_landmarks_align_celeba.csv: missing','warn')}</div>
    </div>"""
    _h("Validation"); display(HTML(val_html))


def compare_saved_stats_and_display(stats_path, m_a, s_a) -> None:
    from IPython.display import display, HTML
    import json
    from pathlib import Path
    stats_path = Path(stats_path)
    if stats_path.is_file():
        with open(stats_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_mean = tuple(float(x) for x in saved.get("train_mean", (float("nan"),) * 3))
        saved_std  = tuple(float(x) for x in saved.get("train_std",  (float("nan"),) * 3))
        saved_scale = "[0,1]" if bool(saved.get("normalize_01", False)) else "[0,255]"
        _h("Saved stats.json (processed)", f"scale {saved_scale}")
        display(HTML(
            "<pre style='margin:0'>"
            f"saved mean_rgb = {tuple(round(x,6) for x in saved_mean)}\n"
            f"saved std_rgb  = {tuple(round(x,6) for x in saved_std)}\n"
            f"abs diff (mean) vs computed = {tuple(round(abs(saved_mean[i]-m_a[i]),6) for i in range(3))}\n"
            f"abs diff (std)  vs computed = {tuple(round(abs(saved_std[i]-s_a[i]),6) for i in range(3))}"
            "</pre>"
        ))
    else:
        _h("Saved stats.json (processed)")
        display(HTML("<div style='color:#6b7280'>stats.json not found; skipping saved-stats comparison.</div>"))


def _relpath(p: "str | Path") -> str:
    """Return a path relative to the repository root when possible.

    Falls back to CWD-relative, else returns the original string.
    The repo root is detected by searching upward from CWD for
    one of: ``pyproject.toml``, ``.git`` directory, or ``src``.
    """
    try:
        path = Path(p).resolve()
        cwd = Path.cwd().resolve()
        # Detect repo root from CWD
        repo_root: Path | None = None
        for candidate in [cwd] + list(cwd.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists() or (candidate / "src").exists():
                repo_root = candidate
                break
        if repo_root and str(path).startswith(str(repo_root)):
            return str(path.relative_to(repo_root))
        if str(path).startswith(str(cwd)):
            return str(path.relative_to(cwd))
        return str(path)
    except Exception:
        return str(p)

# ---------- Styled tables for counts/deltas ----------

def style_counts(df: "pd.DataFrame"):
    """Return a pandas Styler for a counts/delta table with readable formatting.

    Expects columns similar to the balance comparison output: totals and pos% before/after
    plus a Δ column in percentage points.
    """
    fmt_map = {}
    for col in df.columns:
        name = str(col)
        if name.startswith("total"):
            fmt_map[name] = fmt_int
        elif name.startswith("pos%"):
            fmt_map[name] = "{:.1f}%"
        elif "Δ" in name:
            fmt_map[name] = "{:+.1f}"
    styler = (
        df.style
        .format(fmt_map)
        .set_table_styles([{"selector": "th", "props": [("font-weight", "600")]}])
    )
    # Light highlight for any non-zero drift
    delta_cols = [c for c in df.columns if "Δ" in str(c)]
    for c in delta_cols:
        try:
            styler = styler.highlight_between(subset=[c], left=0.001, right=999, color="#fff2cc")
            styler = styler.highlight_between(subset=[c], left=-999, right=-0.001, color="#ffe6e6")
        except Exception:
            continue
    return styler


def style_balance_counts(balance_df: "pd.DataFrame"):
    """Return a styled counts-by-split/class table with Δ vs 50%.

    Expects columns like class counts plus 'total' and either 'pos_ratio' [0,1]
    or 'pos_%' [0,100]. Computes pos_% and Δ from 50% (pp) if needed, and
    formats with thousands separators and fixed decimals. Highlights non-zero Δ.
    """
    import pandas as _pd
    df = balance_df.copy()
    if "pos_%" not in df.columns:
        if "pos_ratio" in df.columns:
            df["pos_%"] = (_pd.to_numeric(df["pos_ratio"], errors="coerce") * 100.0).astype(float)
        else:
            if "eyeglasses" in df.columns and "total" in df.columns:
                df["pos_%"] = (_pd.to_numeric(df["eyeglasses"], errors="coerce") / _pd.to_numeric(df["total"], errors="coerce") * 100.0).astype(float)
    df["Δ from 50% (pp)"] = df["pos_%"].astype(float) - 50.0 if "pos_%" in df.columns else _pd.Series(dtype=float)
    ordered = []
    for c in ["eyeglasses", "no_eyeglasses"]:
        if c in df.columns:
            ordered.append(c)
    if "total" in df.columns:
        ordered.append("total")
    for c in ["pos_%", "Δ from 50% (pp)"]:
        if c in df.columns:
            ordered.append(c)
    for c in df.columns:
        if c not in ordered:
            ordered.append(c)
    show = df[ordered]
    fmt = {}
    for c in show.columns:
        if c in ("eyeglasses", "no_eyeglasses", "total"):
            fmt[c] = fmt_int
        elif c == "pos_%":
            fmt[c] = "{:.1f}%"
        elif c == "Δ from 50% (pp)":
            fmt[c] = "{:+.1f}"
    styler = (
        show.style
        .format(fmt)
        .set_table_styles([{"selector": "th", "props": [("font-weight", "600")]}])
    )
    if "Δ from 50% (pp)" in show.columns:
        try:
            styler = styler.highlight_between(subset=["Δ from 50% (pp)"], left=0.001, right=999, color="#fff2cc")
            styler = styler.highlight_between(subset=["Δ from 50% (pp)"], left=-999, right=-0.001, color="#ffe6e6")
        except Exception:
            pass
    return styler


def find_and_sample_images(
    images_root: "str | Path",
    n_samples: int = 3,
    seed: int = 42,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> tuple[list[Path], int]:
    """Find all image files in a directory and randomly sample N of them.
    
    Args:
        images_root: Root directory containing image files
        n_samples: Number of images to sample (default: 3)
        seed: Random seed for sampling (default: 42)
        image_extensions: Tuple of valid image file extensions
        
    Returns:
        Tuple of (sampled image paths, total image count)
        
    Raises:
        FileNotFoundError: If images_root doesn't exist or is not a directory
        ValueError: If no image files are found
    """
    import random
    from pathlib import Path
    
    images_root = Path(images_root)
    if not images_root.exists() or not images_root.is_dir():
        raise FileNotFoundError(f"Images root directory not found: {_relpath(images_root)}")
    
    # Find all image files
    image_files = [
        f for f in images_root.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No image files found in {_relpath(images_root)}")
    
    # Sample images
    random.seed(seed)
    sampled = random.sample(image_files, min(n_samples, len(image_files)))
    
    return sampled, len(image_files)


