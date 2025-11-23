"""
Visualization utilities.

This module provides plotting functions organized by domain:
- General plotting utilities (histograms, etc.)
- Training-specific visualizations (curves, comparisons, privacy plots)
"""

from .general import (
    hist_multi,
    size_histograms,
    plot_grouped_bars_two_series,
)
from .training import (
    plot_training_curves,
    plot_matched_pair_comparison,
    plot_privacy_cost_summary,
    extract_final_test_accuracies,
    plot_training_dynamics_comparison,
    plot_privacy_accuracy_tradeoff,
    exponential_moving_average,
    simple_moving_average,
)

__all__ = [
    # General plotting
    "hist_multi",
    "size_histograms",
    "plot_grouped_bars_two_series",
    # Training visualizations
    "plot_training_curves",
    "plot_matched_pair_comparison",
    "plot_privacy_cost_summary",
    "extract_final_test_accuracies",
    "plot_training_dynamics_comparison",
    "plot_privacy_accuracy_tradeoff",
    # Smoothing utilities
    "exponential_moving_average",
    "simple_moving_average",
]

