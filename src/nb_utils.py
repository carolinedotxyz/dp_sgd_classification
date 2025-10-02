"""Small notebook utility helpers used across analysis cells.

- _pick: return the first existing column name from a preference list
- _to_pct: convert a fraction column (0-1) to percentage (0-100) if needed
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def _pick(preferred_columns: Iterable[str], actual_columns: Iterable[str]) -> Optional[str]:
    """Return the first column name from ``preferred_columns`` that exists in ``actual_columns``.

    Args:
        preferred_columns: Ordered candidates to try.
        actual_columns: Concrete set/list of available column names.
    Returns:
        The first matching column name, or None if none are present.
    """
    actual = set(actual_columns)
    for name in preferred_columns:
        if name in actual:
            return name
    return None


def _to_pct(df: pd.DataFrame, colname: Optional[str]) -> Optional[pd.Series]:
    """Return a percentage Series for the given column if provided.

    If the column name contains "pct", assumes it's already in [0,100].
    Otherwise treats values as fractions in [0,1] and multiplies by 100.
    Returns None when ``colname`` is None.
    """
    if colname is None:
        return None
    s = df[colname].astype(float)
    return s if ("pct" in colname) else (s * 100.0)


# Public-friendly aliases
pick_column = _pick
to_percent_series = _to_pct

__all__ = ["_pick", "_to_pct"]
__all__ += ["pick_column", "to_percent_series"]


