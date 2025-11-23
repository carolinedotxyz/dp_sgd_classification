"""Training visualization and plotting functionality.

DEPRECATED: This module has been moved to src.visualization.training.
Import from src.visualization or src.training instead:

    from src.visualization import plot_training_curves
    # or
    from src.training import plot_training_curves

This file will be removed in a future version.
"""

import warnings

warnings.warn(
    "src.training.visualization is deprecated. Use 'from src.visualization import plot_training_curves' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from ..visualization.training import (
    plot_training_curves,
    plot_matched_pair_comparison,
    plot_privacy_cost_summary,
    extract_final_test_accuracies,
)

__all__ = [
    "plot_training_curves",
    "plot_matched_pair_comparison",
    "plot_privacy_cost_summary",
    "extract_final_test_accuracies",
]
