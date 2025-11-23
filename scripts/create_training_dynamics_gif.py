#!/usr/bin/env python3
"""
Create an animated GIF showing training dynamics comparison: Baseline vs DP-SGD.

This script generates an animated visualization that shows how DP-SGD converges
slower than baseline due to privacy-preserving noise. The animation progresses
epoch-by-epoch, making the privacy-accuracy trade-off visually clear.

Usage:
    # From notebook (after training):
    python scripts/create_training_dynamics_gif.py \
        --baseline-history history_df_quick \
        --dp-history dp_history_df_quick \
        --output docs/assets/training_dynamics_comparison.gif
    
    # From CSV files:
    python scripts/create_training_dynamics_gif.py \
        --baseline-csv runs/baseline/sweep_*/run_*/history.csv \
        --dp-csv runs/dp_sgd/sweep_*/run_*/history.csv \
        --output docs/assets/training_dynamics_comparison.gif
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.training import extract_final_test_accuracies


def get_config_key(history_df: pd.DataFrame, run_name: str) -> Optional[Tuple]:
    """Extract configuration key (lr, batch_size, weight_decay) from run."""
    run_data = history_df[history_df['run'] == run_name]
    if len(run_data) == 0:
        return None
    row = run_data.iloc[0]
    return (
        float(row.get('lr', 0)),
        int(row.get('batch_size', 0)),
        float(row.get('weight_decay', 0))
    )


def find_matched_pairs(
    baseline_history: pd.DataFrame,
    dp_history: pd.DataFrame
) -> List[Dict]:
    """Find matched pairs of baseline and DP-SGD runs with identical hyperparameters."""
    matched_pairs = []
    baseline_keys = {}
    
    for run in baseline_history['run'].unique():
        key = get_config_key(baseline_history, run)
        if key:
            baseline_keys[key] = run
    
    for run in dp_history['run'].unique():
        key = get_config_key(dp_history, run)
        if key and key in baseline_keys:
            baseline_run = baseline_keys[key]
            matched_pairs.append({
                'key': key,
                'baseline_run': baseline_run,
                'dp_run': run,
                'label': f"LR={key[0]:g}, BS={key[1]}, WD={key[2]:g}"
            })
    
    return matched_pairs


def create_animated_gif(
    baseline_history: pd.DataFrame,
    dp_history: pd.DataFrame,
    output_path: Path,
    max_configs: int = 3,
    fps: int = 2,
    duration_per_epoch: float = 0.5
) -> None:
    """
    Create animated GIF showing training dynamics comparison.
    
    Args:
        baseline_history: DataFrame with baseline training history
        dp_history: DataFrame with DP-SGD training history
        output_path: Path to save the GIF
        max_configs: Maximum number of configurations to show
        fps: Frames per second for the animation
        duration_per_epoch: Duration to show each epoch (seconds)
    """
    # Find matched pairs
    matched_pairs = find_matched_pairs(baseline_history, dp_history)
    
    if len(matched_pairs) == 0:
        raise ValueError("No matched pairs found between baseline and DP-SGD histories")
    
    # Select top configurations by baseline accuracy
    baseline_accs = extract_final_test_accuracies(baseline_history)
    matched_pairs.sort(key=lambda x: baseline_accs.get(x['baseline_run'], 0), reverse=True)
    matched_pairs = matched_pairs[:max_configs]
    
    # Get maximum epochs across all runs
    max_epochs = max(
        baseline_history['epoch'].max(),
        dp_history['epoch'].max()
    )
    
    # Create figure
    fig, axes = plt.subplots(2, len(matched_pairs), figsize=(5*len(matched_pairs), 8))
    if len(matched_pairs) == 1:
        axes = axes.reshape(-1, 1)
    
    # Store line objects for animation
    lines_dict = {}
    ax_dict = {}
    
    # Initialize plots
    for col_idx, pair in enumerate(matched_pairs):
        baseline_run = pair['baseline_run']
        dp_run = pair['dp_run']
        
        # Get training data
        baseline_data = baseline_history[baseline_history['run'] == baseline_run].sort_values('epoch')
        dp_data = dp_history[dp_history['run'] == dp_run].sort_values('epoch')
        
        # Loss plot (top row)
        ax_loss = axes[0, col_idx]
        ax_loss.set_title(f'Training Loss\n{pair["label"]}', fontsize=11, fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_xlim([0, max_epochs + 1])
        
        # Get y-axis limits for loss
        all_losses = list(baseline_data['train_loss']) + list(dp_data['train_loss'])
        loss_min, loss_max = min(all_losses), max(all_losses)
        loss_margin = (loss_max - loss_min) * 0.1
        ax_loss.set_ylim([max(0, loss_min - loss_margin), loss_max + loss_margin])
        
        # Create empty lines for animation
        line_baseline_loss, = ax_loss.plot([], [], label='Baseline', 
                                          color='#3498db', linewidth=2.5, marker='o', markersize=4)
        line_dp_loss, = ax_loss.plot([], [], label='DP-SGD', 
                                    color='#e74c3c', linewidth=2.5, marker='s', markersize=4, linestyle='--')
        ax_loss.legend(fontsize=9)
        
        # Accuracy plot (bottom row)
        ax_acc = axes[1, col_idx]
        ax_acc.set_title(f'Validation Accuracy\n{pair["label"]}', fontsize=11, fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_xlim([0, max_epochs + 1])
        ax_acc.set_ylim([0, 1])
        
        # Create empty lines for animation
        line_baseline_acc, = ax_acc.plot([], [], label='Baseline', 
                                         color='#3498db', linewidth=2.5, marker='o', markersize=4)
        line_dp_acc, = ax_acc.plot([], [], label='DP-SGD', 
                                   color='#e74c3c', linewidth=2.5, marker='s', markersize=4, linestyle='--')
        ax_acc.legend(fontsize=9)
        
        # Store references
        lines_dict[col_idx] = {
            'baseline_loss': (line_baseline_loss, baseline_data),
            'dp_loss': (line_dp_loss, dp_data),
            'baseline_acc': (line_baseline_acc, baseline_data),
            'dp_acc': (line_dp_acc, dp_data),
        }
        ax_dict[col_idx] = {
            'loss': ax_loss,
            'acc': ax_acc,
        }
    
    plt.suptitle('Training Dynamics: Baseline vs DP-SGD', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Animation function
    def animate(frame):
        """Update plots for current epoch."""
        current_epoch = frame + 1
        
        for col_idx in range(len(matched_pairs)):
            # Update loss plot
            baseline_line, baseline_data = lines_dict[col_idx]['baseline_loss']
            dp_line, dp_data = lines_dict[col_idx]['dp_loss']
            
            # Get data up to current epoch
            baseline_epochs = baseline_data[baseline_data['epoch'] <= current_epoch]['epoch']
            baseline_losses = baseline_data[baseline_data['epoch'] <= current_epoch]['train_loss']
            dp_epochs = dp_data[dp_data['epoch'] <= current_epoch]['epoch']
            dp_losses = dp_data[dp_data['epoch'] <= current_epoch]['train_loss']
            
            baseline_line.set_data(baseline_epochs, baseline_losses)
            dp_line.set_data(dp_epochs, dp_losses)
            
            # Update accuracy plot
            baseline_line_acc, baseline_data = lines_dict[col_idx]['baseline_acc']
            dp_line_acc, dp_data = lines_dict[col_idx]['dp_acc']
            
            baseline_accs = baseline_data[baseline_data['epoch'] <= current_epoch]['val_acc']
            dp_accs = dp_data[dp_data['epoch'] <= current_epoch]['val_acc']
            
            baseline_line_acc.set_data(baseline_epochs, baseline_accs)
            dp_line_acc.set_data(dp_epochs, dp_accs)
        
        return [line for lines in lines_dict.values() for line in [lines['baseline_loss'][0], 
                                                                   lines['dp_loss'][0],
                                                                   lines['baseline_acc'][0],
                                                                   lines['dp_acc'][0]]]
    
    # Create animation
    print(f"🎬 Creating animation with {max_epochs} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=max_epochs, 
        interval=int(1000 * duration_per_epoch),
        blit=True, repeat=True
    )
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save GIF
    print(f"💾 Saving GIF to {output_path}...")
    try:
        # Try using pillow writer (requires pillow, which is already in dependencies)
        anim.save(str(output_path), writer='pillow', fps=fps)
    except Exception as e:
        print(f"⚠️  Pillow writer failed: {e}")
        print("   Trying imageio writer as fallback...")
        try:
            import imageio.v2 as imageio
            # Save frames and create GIF
            frames = []
            for i in range(max_epochs):
                animate(i)
                fig.canvas.draw()
                # Convert canvas to numpy array
                buf = fig.canvas.buffer_rgba()
                frame = np.asarray(buf)
                frames.append(frame)
            imageio.mimsave(str(output_path), frames, fps=fps, format='GIF')
        except ImportError:
            print("❌ Error: Need either 'pillow' or 'imageio' to create GIF")
            print("   Pillow should already be installed (in dependencies)")
            print("   If missing, install with: pip install pillow")
            print("   Or install imageio as fallback: pip install imageio")
            sys.exit(1)
        except Exception as e2:
            print(f"❌ Error creating GIF: {e2}")
            print("   Please ensure pillow is installed: pip install pillow")
            sys.exit(1)
    
    print(f"✅ GIF saved successfully: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Frames: {max_epochs}, FPS: {fps}")


def load_history_from_csv(csv_path: str) -> pd.DataFrame:
    """Load training history from CSV file(s)."""
    # Handle glob patterns in the path
    # Pattern like "notebooks/runs/baseline/*/history.csv" 
    # Extract base directory and search for history.csv files
    csv_path_str = str(csv_path)
    
    if '*' in csv_path_str or '?' in csv_path_str:
        # Extract the base directory before the wildcard
        # e.g., "notebooks/runs/baseline/*/history.csv" -> "notebooks/runs/baseline"
        parts = csv_path_str.split('*')[0].split('?')[0]
        base_dir = Path(parts.rstrip('/'))
        # Search for all history.csv files in this directory tree
        csv_paths = list(base_dir.rglob('history.csv'))
    else:
        # Simple path without wildcards
        csv_paths = [Path(csv_path)] if Path(csv_path).exists() else []
    
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found matching: {csv_path}")
    
    dfs = []
    for path in csv_paths:
        if path.exists() and path.is_file():
            df = pd.read_csv(path)
            # Add run name from parent directory if not present
            if 'run' not in df.columns:
                df['run'] = path.parent.name
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No valid CSV files found matching: {csv_path}")
    
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIF showing training dynamics comparison"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--baseline-csv',
        type=str,
        help='Path pattern to baseline history CSV files (e.g., "runs/baseline/*/history.csv")'
    )
    input_group.add_argument(
        '--baseline-history',
        type=str,
        help='Variable name for baseline history DataFrame (for notebook use)'
    )
    
    dp_group = parser.add_mutually_exclusive_group(required=True)
    dp_group.add_argument(
        '--dp-csv',
        type=str,
        help='Path pattern to DP-SGD history CSV files'
    )
    dp_group.add_argument(
        '--dp-history',
        type=str,
        help='Variable name for DP-SGD history DataFrame (for notebook use)'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='docs/assets/training_dynamics_comparison.gif',
        help='Output path for GIF file (default: docs/assets/training_dynamics_comparison.gif)'
    )
    
    # Options
    parser.add_argument(
        '--max-configs',
        type=int,
        default=3,
        help='Maximum number of configurations to show (default: 3)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='Frames per second for animation (default: 2)'
    )
    parser.add_argument(
        '--duration-per-epoch',
        type=float,
        default=0.5,
        help='Duration to show each epoch in seconds (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Load histories
    if args.baseline_csv:
        print(f"📂 Loading baseline history from: {args.baseline_csv}")
        baseline_history = load_history_from_csv(args.baseline_csv)
    else:
        # For notebook use, this would need to be passed differently
        # For now, we'll require CSV files
        raise ValueError("Please use --baseline-csv for command-line usage")
    
    if args.dp_csv:
        print(f"📂 Loading DP-SGD history from: {args.dp_csv}")
        dp_history = load_history_from_csv(args.dp_csv)
    else:
        raise ValueError("Please use --dp-csv for command-line usage")
    
    # Create GIF
    output_path = Path(args.output)
    create_animated_gif(
        baseline_history=baseline_history,
        dp_history=dp_history,
        output_path=output_path,
        max_configs=args.max_configs,
        fps=args.fps,
        duration_per_epoch=args.duration_per_epoch
    )


if __name__ == '__main__':
    main()

