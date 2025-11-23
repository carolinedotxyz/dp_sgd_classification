"""
Training module for DP-SGD classification experiments.

This module contains all training-related functionality including:
- Core training loops and evaluation
- Hyperparameter sweep orchestration  
- Run management and artifact saving
- Training visualization and progress tracking
"""

from .trainer import train_one_epoch_with_progress, train_one_epoch_dp_sgd_with_progress
from .run_manager import (
    create_run_directory,
    save_run_artifacts,
    display_run_summary
)
from .sweep_manager import (
    run_sgd_sweep_with_progress, 
    run_dp_sgd_sweep_with_progress,
    run_dp_sgd_privacy_parameter_sweep
)
from ..visualization.training import (
    plot_training_curves,
    plot_matched_pair_comparison,
    plot_privacy_cost_summary,
    extract_final_test_accuracies,
    plot_training_dynamics_comparison,
    plot_privacy_accuracy_tradeoff,
)

__all__ = [
    'train_one_epoch_with_progress',
    'train_one_epoch_dp_sgd_with_progress',
    'create_run_directory',
    'save_run_artifacts', 
    'display_run_summary',
    'run_sgd_sweep_with_progress',
    'run_dp_sgd_sweep_with_progress',
    'run_dp_sgd_privacy_parameter_sweep',
    'plot_training_curves',
    'plot_matched_pair_comparison',
    'plot_privacy_cost_summary',
    'extract_final_test_accuracies',
    'plot_training_dynamics_comparison',
    'plot_privacy_accuracy_tradeoff',
]
