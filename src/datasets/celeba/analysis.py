"""Balance and comparison utilities for CelebA analysis.

Pure data utilities extracted from the notebook to enable reuse.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd

from ...notebooks.utils import pick_column, to_percent_series
from ...notebooks.display import ATTR_RENAME


def compute_attribute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute overall positive percentage per attribute from a CelebA DataFrame.

    This function requires an upstream `compute_balance(df, focus_attrs)` and will
    handle both `pos_frac` ([0,1]) and `pos_pct` ([0,100]) columns.
    """
    summary_all = compute_balance_from_df(df, [])
    if "pos_frac" in summary_all.columns:
        summary_all["overall_pos_pct"] = summary_all["pos_frac"].map(lambda x: float(x) * 100.0)
    elif "pos_pct" in summary_all.columns:
        summary_all["overall_pos_pct"] = summary_all["pos_pct"].astype(float)
    else:
        summary_all["overall_pos_pct"] = (summary_all["pos"] / summary_all["total"] * 100.0)
    summary_all["display_name"] = summary_all["attribute"].map(lambda a: ATTR_RENAME.get(a, a))
    return summary_all


def build_focus_table(summary_focus: pd.DataFrame) -> pd.DataFrame:
    col_overall = pick_column(["pos_frac", "pos_pct"], summary_focus.columns)
    col_train = pick_column(["train_pos_frac", "train_pos_pct"], summary_focus.columns)
    col_val = pick_column(["val_pos_frac", "val_pos_pct"], summary_focus.columns)
    col_test = pick_column(["test_pos_frac", "test_pos_pct"], summary_focus.columns)
    ov = to_percent_series(summary_focus, col_overall)
    trn = to_percent_series(summary_focus, col_train)
    val = to_percent_series(summary_focus, col_val)
    tst = to_percent_series(summary_focus, col_test)
    focus_tbl = pd.DataFrame({
        "Attribute": summary_focus["attribute"].map(lambda a: ATTR_RENAME.get(a, a)),
        "Total": summary_focus["total"].astype(int),
        "Pos": summary_focus["pos"].astype(int),
        "Neg": summary_focus["neg"].astype(int),
        "Overall %": ov,
    })
    if trn is not None:
        focus_tbl["Train %"] = trn
    if val is not None:
        focus_tbl["Val %"] = val
    if tst is not None:
        focus_tbl["Test %"] = tst
    for name in ["Train %", "Val %", "Test %"]:
        if name in focus_tbl.columns:
            focus_tbl[f"Δ {name.split()[0]} (pp)"] = focus_tbl[name] - focus_tbl["Overall %"]
    return focus_tbl


def build_balance_comparison(dfo: pd.DataFrame, dfp: pd.DataFrame):
    import numpy as np
    import pandas as pd

    def _balance(df: pd.DataFrame) -> pd.DataFrame:
        t = df.groupby(["partition_name", "class_name"]).size().unstack(fill_value=0)
        t["total"] = t.sum(axis=1)
        if "eyeglasses" in t.columns:
            t["pos_%"] = t["eyeglasses"] / t["total"] * 100.0
        return t

    bal_before = _balance(dfo)
    bal_after = _balance(dfp)
    splits = sorted(set(bal_before.index).union(set(bal_after.index)))
    rows = []
    from ...notebooks.display import _fmt_pct, _pp
    for s in splits:
        b = bal_before.loc[s] if s in bal_before.index else pd.Series(dtype=float)
        a = bal_after.loc[s] if s in bal_after.index else pd.Series(dtype=float)
        total_b = int(b.get("total", np.nan)) if not pd.isna(b.get("total", np.nan)) else np.nan
        total_a = int(a.get("total", np.nan)) if not pd.isna(a.get("total", np.nan)) else np.nan
        pos_b = float(b.get("pos_%", np.nan))
        pos_a = float(a.get("pos_%", np.nan))
        rows.append({
            "split": s,
            "total (before)": f"{total_b:,}" if pd.notna(total_b) else "",
            "total (after)": f"{total_a:,}" if pd.notna(total_a) else "",
            "pos% (before)": _fmt_pct(pos_b, 1) if pd.notna(pos_b) else "",
            "pos% (after)": _fmt_pct(pos_a, 1) if pd.notna(pos_a) else "",
            "Δ pos% (pp)": _pp(pos_a - pos_b) if pd.notna(pos_a) and pd.notna(pos_b) else "",
        })
    cmp_balance = pd.DataFrame(rows).set_index("split")
    return bal_before, bal_after, splits, cmp_balance


def build_balance_display_table(balance: pd.DataFrame) -> pd.DataFrame:
    bal_show = balance.copy()
    if "pos_ratio" in bal_show.columns:
        bal_show["pos_%"] = (bal_show["pos_ratio"] * 100.0).astype(float)
        bal_show["Δ from 50% (pp)"] = bal_show["pos_%"] - 50.0
    ordered = []
    cls_cols = [c for c in bal_show.columns if c not in ("total", "pos_ratio", "pos_%", "Δ from 50% (pp)")]
    if len(cls_cols) >= 2:
        ordered += cls_cols[:2]
    if "total" in bal_show:
        ordered += ["total"]
    if "pos_%" in bal_show:
        ordered += ["pos_%"]
    if "Δ from 50% (pp)" in bal_show:
        ordered += ["Δ from 50% (pp)"]
    for c in bal_show.columns:
        if c not in ordered:
            ordered.append(c)
    fmt_bal = bal_show[ordered].copy()
    if hasattr(pd.api.types, "is_numeric_dtype"):
        for c in fmt_bal.columns:
            if c not in ("pos_%", "Δ from 50% (pp)") and pd.api.types.is_numeric_dtype(fmt_bal[c]):
                fmt_bal[c] = fmt_bal[c].map(lambda x: f"{int(x):,}")
    if "pos_%" in fmt_bal:
        from ...notebooks.display import _fmt_pct
        fmt_bal["pos_%"] = fmt_bal["pos_%"].map(lambda x: _fmt_pct(x, 1))
    if "Δ from 50% (pp)" in fmt_bal:
        from ...notebooks.display import _pp
        fmt_bal["Δ from 50% (pp)"] = fmt_bal["Δ from 50% (pp)"].map(_pp)
    return fmt_bal


def compute_balance_from_df(df: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """Compute per-attribute balance overall and by split from merged DF.

    If ``attributes`` is empty, use all attribute columns in ``df``.
    Expects columns: ``image_id``, ``partition_name``, and attribute columns.
    """
    available = [c for c in df.columns if c not in ("image_id", "partition", "partition_name")]
    if not attributes:
        attributes = available
    else:
        invalid = [a for a in attributes if a not in available]
        if invalid:
            raise ValueError(f"Attributes not found: {invalid}")

    rows: list[dict[str, object]] = []
    PARTS = ("train", "val", "test")
    for attr in attributes:
        series = df[attr]
        total = int(series.notna().sum())
        pos = int((series == 1).sum())
        neg = int((series == -1).sum())
        row: dict[str, object] = {
            "attribute": attr,
            "total": total,
            "pos": pos,
            "neg": neg,
            "pos_frac": (pos / total) if total else 0.0,
            "neg_frac": (neg / total) if total else 0.0,
        }
        for pname in PARTS:
            sub = df[df["partition_name"] == pname]
            stotal = int(sub.shape[0])
            npos = int((sub[attr] == 1).sum())
            nneg = int((sub[attr] == -1).sum())
            row[f"{pname}_total"] = stotal
            row[f"{pname}_pos"] = npos
            row[f"{pname}_neg"] = nneg
            row[f"{pname}_pos_frac"] = (npos / stotal) if stotal else 0.0
            row[f"{pname}_neg_frac"] = (nneg / stotal) if stotal else 0.0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(by=["attribute"]).reset_index(drop=True)
    # Backward-compatible columns: pos_pct mirrors pos_frac (fraction in [0,1])
    if "pos_frac" in out.columns and "pos_pct" not in out.columns:
        out["pos_pct"] = out["pos_frac"].astype(float)
    for pname in ("train", "val", "test"):
        frac_col = f"{pname}_pos_frac"
        pct_col = f"{pname}_pos_pct"
        if frac_col in out.columns and pct_col not in out.columns:
            out[pct_col] = out[frac_col].astype(float)
    return out


