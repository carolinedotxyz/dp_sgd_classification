"""
Notebook-specific utilities.

This module contains code that is specific to Jupyter notebooks and not
intended for use in production scripts. These utilities help keep notebooks
clean and focused on analysis.
"""

from .utils import (
    _pick,
    _to_pct,
    pick_column,
    to_percent_series,
    generate_timestamp,
    validate_training_histories,
    print_config,
)
from .display import (
    _h,
    _badge,
    _pct,
    _fmt_pct,
    _pp,
    _has_variance,
    highlight_drift,
    bold_extremes,
    _relpath,
    ATTR_RENAME,
    PREFERRED_FOCUS,
    style_focus_table,
    render_validation_badges,
    compare_saved_stats_and_display,
    style_counts,
    style_balance_counts,
    find_and_sample_images,
)
from .setup import (
    NotebookSetup,
    get_setup,
    setup_training_cell,
    setup_analysis_cell,
    setup_workflow_cell,
    log_cell_completion,
)

__all__ = [
    # Utils
    "_pick",
    "_to_pct",
    "pick_column",
    "to_percent_series",
    "generate_timestamp",
    "validate_training_histories",
    "print_config",
    # Display
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
    "style_focus_table",
    "render_validation_badges",
    "compare_saved_stats_and_display",
    "style_counts",
    "style_balance_counts",
    "find_and_sample_images",
    # Setup
    "NotebookSetup",
    "get_setup",
    "setup_training_cell",
    "setup_analysis_cell",
    "setup_workflow_cell",
    "log_cell_completion",
]

