"""Notebook display helpers and constants for a cleaner, consistent look.

This module centralizes small HTML helpers, pandas display configuration, and
display-only constants to keep notebooks minimal and focused on analysis.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


 


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


