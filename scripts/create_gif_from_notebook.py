"""
Notebook helper function to create animated GIF from training histories.

This module provides a convenient function to call from the notebook after training
to generate the animated GIF showing training dynamics comparison.

Usage in notebook:
    from scripts.create_gif_from_notebook import create_training_dynamics_gif
    
    create_training_dynamics_gif(
        baseline_history=history_df_quick,
        dp_history=dp_history_df_quick,
        output_path='docs/assets/training_dynamics_comparison.gif'
    )
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_training_dynamics_gif import create_animated_gif


def create_training_dynamics_gif(
    baseline_history,
    dp_history,
    output_path: str = 'docs/assets/training_dynamics_comparison.gif',
    max_configs: int = 3,
    fps: int = 2,
    duration_per_epoch: float = 0.5
):
    """
    Create animated GIF showing training dynamics comparison.
    
    Convenience function for use in notebooks.
    
    Args:
        baseline_history: DataFrame with baseline training history
        dp_history: DataFrame with DP-SGD training history
        output_path: Path to save the GIF (default: docs/assets/training_dynamics_comparison.gif)
        max_configs: Maximum number of configurations to show (default: 3)
        fps: Frames per second for animation (default: 2)
        duration_per_epoch: Duration to show each epoch in seconds (default: 0.5)
    
    Example:
        >>> from scripts.create_gif_from_notebook import create_training_dynamics_gif
        >>> create_training_dynamics_gif(
        ...     baseline_history=history_df_quick,
        ...     dp_history=dp_history_df_quick
        ... )
    """
    output_path = Path(output_path)
    create_animated_gif(
        baseline_history=baseline_history,
        dp_history=dp_history,
        output_path=output_path,
        max_configs=max_configs,
        fps=fps,
        duration_per_epoch=duration_per_epoch
    )

