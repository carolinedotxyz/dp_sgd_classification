"""Small notebook utility helpers used across analysis cells.

- _pick: return the first existing column name from a preference list
- _to_pct: convert a fraction column (0-1) to percentage (0-100) if needed
- generate_timestamp: generate timestamp string for filenames
- validate_training_histories: validate that required training history variables exist
- print_config: print configuration dictionaries in a consistent format
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, List, Tuple

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


def generate_timestamp() -> str:
    """Generate a timestamp string in format YYYYMMDD_HHMMSS for filenames.
    
    Returns:
        Timestamp string like "20240101_123045"
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_training_histories(
    required_vars: List[str],
    var_descriptions: Optional[dict[str, str]] = None
) -> Tuple[bool, List[str]]:
    """Validate that required training history variables exist in globals.
    
    Args:
        required_vars: List of variable names to check (e.g., ['history_df_quick', 'dp_history_df_quick'])
        var_descriptions: Optional dict mapping var names to descriptions for error messages
        
    Returns:
        Tuple of (all_exist: bool, missing_vars: List[str])
    """
    import sys
    frame = sys._getframe(1)  # Get caller's frame
    globals_dict = frame.f_globals
    
    missing = []
    for var in required_vars:
        if var not in globals_dict:
            missing.append(var)
    
    return len(missing) == 0, missing


def print_config(title: str, config_dict: dict, indent: int = 0) -> None:
    """Print a configuration dictionary in a consistent, readable format.
    
    Args:
        title: Title/description of the configuration being printed
        config_dict: Dictionary of configuration key-value pairs
        indent: Number of spaces to indent (default: 0)
    """
    indent_str = " " * indent
    print(f"{indent_str}{title}:")
    for key, value in config_dict.items():
        print(f"{indent_str}  {key}: {value}")


# Public-friendly aliases
pick_column = _pick
to_percent_series = _to_pct

__all__ = ["_pick", "_to_pct"]
__all__ += ["pick_column", "to_percent_series"]
__all__ += ["generate_timestamp", "validate_training_histories", "print_config"]


