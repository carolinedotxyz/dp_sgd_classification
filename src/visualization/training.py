"""
Training visualization and plotting functionality.

This module contains functions for creating training curves, progress plots,
and other visualizations used during and after training experiments.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple


def exponential_moving_average(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential moving average smoothing to a sequence of values.
    
    This is a simple and effective smoothing technique that reduces noise
    while preserving trends. Lower alpha (0.1-0.3) = more smoothing,
    higher alpha (0.5-0.9) = less smoothing, closer to raw values.
    
    Args:
        values: Array of values to smooth
        alpha: Smoothing factor (0 < alpha <= 1). Default 0.3 provides good balance.
               Lower values = more smoothing, higher values = less smoothing.
        
    Returns:
        Smoothed array of same length as input
    """
    if len(values) == 0:
        return values
    if len(values) == 1:
        return values
    
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def simple_moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply simple moving average smoothing to a sequence of values.
    
    This averages values over a sliding window, which helps reduce noise.
    Window size of 3-5 works well for epoch-level metrics.
    
    Args:
        values: Array of values to smooth
        window: Size of the moving average window (must be odd, >= 3).
                Default 3 provides light smoothing, 5 provides more smoothing.
        
    Returns:
        Smoothed array of same length as input
    """
    if len(values) == 0:
        return values
    if len(values) < window:
        # If not enough values, just return the mean
        return np.full_like(values, np.mean(values), dtype=float)
    
    # Ensure window is odd for symmetric smoothing
    if window % 2 == 0:
        window += 1
    
    half_window = window // 2
    smoothed = np.zeros_like(values, dtype=float)
    
    # Handle edges by using available data
    for i in range(len(values)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        smoothed[i] = np.mean(values[start_idx:end_idx])
    
    return smoothed


def plot_training_curves(
    history_df: pd.DataFrame, 
    save_path: Optional[str] = None,
    smooth: bool = True,
    smoothing_method: str = 'ema',
    smoothing_param: float = 0.3
) -> plt.Figure:
    """
    Plot training curves with each run as a separate line.
    
    Args:
        history_df: DataFrame containing training history
        save_path: Optional path to save the plot
        smooth: Whether to apply smoothing to curves (default: True)
        smoothing_method: Method to use ('ema' for exponential moving average, 
                        'sma' for simple moving average, default: 'ema')
        smoothing_param: Smoothing parameter. For EMA: alpha (0.1-0.9, default 0.3).
                         For SMA: window size (3-7, default 3). Lower EMA alpha = more smoothing.
        
    Returns:
        Matplotlib figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Get unique runs and colors
    runs = history_df['run'].unique()
    colors = plt.cm.tab10(range(len(runs)))
    
    for i, run in enumerate(runs):
        run_data = history_df[history_df['run'] == run].sort_values('epoch')
        epochs = run_data['epoch']
        
        # Apply smoothing if requested
        if smooth:
            if smoothing_method == 'ema':
                train_loss_smooth = exponential_moving_average(
                    run_data['train_loss'].values, alpha=smoothing_param
                )
                val_loss_smooth = exponential_moving_average(
                    run_data['val_loss'].values, alpha=smoothing_param
                )
                train_acc_smooth = exponential_moving_average(
                    run_data['train_acc'].values, alpha=smoothing_param
                )
                val_acc_smooth = exponential_moving_average(
                    run_data['val_acc'].values, alpha=smoothing_param
                )
            elif smoothing_method == 'sma':
                window = int(smoothing_param) if smoothing_param >= 3 else 3
                train_loss_smooth = simple_moving_average(
                    run_data['train_loss'].values, window=window
                )
                val_loss_smooth = simple_moving_average(
                    run_data['val_loss'].values, window=window
                )
                train_acc_smooth = simple_moving_average(
                    run_data['train_acc'].values, window=window
                )
                val_acc_smooth = simple_moving_average(
                    run_data['val_acc'].values, window=window
                )
            else:
                raise ValueError(f"Unknown smoothing method: {smoothing_method}. Use 'ema' or 'sma'.")
        else:
            train_loss_smooth = run_data['train_loss'].values
            val_loss_smooth = run_data['val_loss'].values
            train_acc_smooth = run_data['train_acc'].values
            val_acc_smooth = run_data['val_acc'].values
        
        # Plot 1: Training Loss
        ax1.plot(epochs, train_loss_smooth, 
                label=run, color=colors[i], marker='o', markersize=4, linewidth=2)
        
        # Plot 2: Validation Loss  
        ax2.plot(epochs, val_loss_smooth, 
                label=run, color=colors[i], marker='s', markersize=4, linewidth=2)
        
        # Plot 3: Training Accuracy
        ax3.plot(epochs, train_acc_smooth, 
                label=run, color=colors[i], marker='o', markersize=4, linewidth=2)
        
        # Plot 4: Validation Accuracy
        ax4.plot(epochs, val_acc_smooth, 
                label=run, color=colors[i], marker='s', markersize=4, linewidth=2)
    
    # Styling
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax4.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training curves saved to: {save_path}")
    
    plt.show()
    return fig


def extract_final_test_accuracies(history_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract final test accuracies for each run from history DataFrame.
    
    Note: If test_acc is not in the DataFrame, uses final validation accuracy as a proxy.
    This is because test_acc is typically computed once at the end of training,
    not per-epoch.
    
    Args:
        history_df: DataFrame containing training history with 'run' column
        
    Returns:
        Dictionary mapping run names to final accuracies (test_acc if available, else val_acc)
    """
    results = {}
    for run in history_df['run'].unique():
        run_data = history_df[history_df['run'] == run].sort_values('epoch')
        if len(run_data) == 0:
            continue
        
        # Prefer test_acc if available, otherwise use final val_acc
        if 'test_acc' in run_data.columns:
            # If test_acc is per-epoch, take the last one
            final_acc = run_data['test_acc'].iloc[-1]
        elif 'val_acc' in run_data.columns:
            # Use final validation accuracy as proxy
            final_acc = run_data['val_acc'].iloc[-1]
        else:
            continue
            
        results[run] = float(final_acc)
    return results


def plot_matched_pair_comparison(
    baseline_history: pd.DataFrame,
    dp_history: pd.DataFrame,
    save_path: Optional[str] = None,
    min_baseline_acc: float = 0.6
) -> plt.Figure:
    """
    Plot side-by-side comparison of baseline vs DP-SGD for matched configurations.
    
    This visualization shows the privacy-accuracy trade-off clearly by comparing
    identical hyperparameter configurations with and without privacy.
    
    Configurations where baseline accuracy is below min_baseline_acc are excluded
    from the main analysis, as they don't represent a fair comparison (baseline
    didn't learn, so privacy cost cannot be meaningfully measured).
    
    Args:
        baseline_history: DataFrame with baseline training history
        dp_history: DataFrame with DP-SGD training history
        save_path: Optional path to save the plot
        min_baseline_acc: Minimum baseline accuracy threshold (default: 0.6).
                         Configurations below this are excluded from analysis.
        
    Returns:
        Matplotlib figure object
    """
    # Extract final test accuracies
    baseline_accs = extract_final_test_accuracies(baseline_history)
    dp_accs = extract_final_test_accuracies(dp_history)
    
    # Match configurations by hyperparameters (lr, batch_size, weight_decay)
    # Create a matching key from hyperparameters
    def get_config_key(history_df: pd.DataFrame, run_name: str) -> Optional[Tuple]:
        run_data = history_df[history_df['run'] == run_name]
        if len(run_data) == 0:
            return None
        row = run_data.iloc[0]
        return (float(row.get('lr', 0)), int(row.get('batch_size', 0)), float(row.get('weight_decay', 0)))
    
    # Build matched pairs
    matched_pairs = []
    excluded_pairs = []
    baseline_keys = {}
    for run in baseline_history['run'].unique():
        key = get_config_key(baseline_history, run)
        if key:
            baseline_keys[key] = run
    
    for run in dp_history['run'].unique():
        key = get_config_key(dp_history, run)
        if key and key in baseline_keys:
            baseline_run = baseline_keys[key]
            if baseline_run in baseline_accs and run in dp_accs:
                baseline_acc = baseline_accs[baseline_run]
                dp_acc = dp_accs[run]
                pair_data = {
                    'config_key': key,
                    'baseline_run': baseline_run,
                    'dp_run': run,
                    'baseline_acc': baseline_acc,
                    'dp_acc': dp_acc,
                    'privacy_cost': baseline_acc - dp_acc,
                    'label': f"LR={key[0]:g}, BS={key[1]}, WD={key[2]:g}"
                }
                
                # Filter: only include if baseline learned (above threshold)
                if baseline_acc >= min_baseline_acc:
                    matched_pairs.append(pair_data)
                else:
                    excluded_pairs.append(pair_data)
    
    # Report excluded configurations
    if excluded_pairs:
        print(f"\n⚠️  Excluded {len(excluded_pairs)} configuration(s) where baseline accuracy < {min_baseline_acc}:")
        for p in excluded_pairs:
            print(f"   {p['label']}: Baseline={p['baseline_acc']:.3f}, DP-SGD={p['dp_acc']:.3f} "
                  f"(Baseline did not learn - not a fair comparison)")
        print()
    
    if len(matched_pairs) == 0:
        print("⚠️  No matched pairs found. Ensure both sweeps use identical hyperparameters.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No matched pairs found", ha='center', va='center', fontsize=14)
        return fig
    
    # Sort by privacy cost (largest first)
    matched_pairs.sort(key=lambda x: x['privacy_cost'], reverse=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Side-by-side bar chart
    n_pairs = len(matched_pairs)
    x = np.arange(n_pairs)
    width = 0.35
    
    baseline_vals = [p['baseline_acc'] for p in matched_pairs]
    dp_vals = [p['dp_acc'] for p in matched_pairs]
    labels = [p['label'] for p in matched_pairs]
    
    bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline (No Privacy)', 
                    color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, dp_vals, width, label='DP-SGD (Privacy)', 
                    color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Matched Pair Comparison: Baseline vs DP-SGD', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, max(max(baseline_vals), max(dp_vals)) * 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Privacy cost (accuracy difference)
    privacy_costs = [p['privacy_cost'] for p in matched_pairs]
    colors_cost = ['#c0392b' if cost > 0.05 else '#e67e22' if cost > 0.02 else '#f39c12' 
                   for cost in privacy_costs]
    
    bars3 = ax2.barh(x, privacy_costs, color=colors_cost, alpha=0.8)
    ax2.set_xlabel('Privacy Cost (Accuracy Drop)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_title('Privacy Cost Analysis', fontsize=14, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars3, privacy_costs)):
        ax2.text(cost, bar.get_y() + bar.get_height()/2.,
                f'{cost:.3f}',
                ha='left' if cost > 0 else 'right', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Matched pair comparison saved to: {save_path}")
    
    plt.show()
    return fig


def plot_privacy_cost_summary(
    baseline_history: pd.DataFrame,
    dp_history: pd.DataFrame,
    save_path: Optional[str] = None,
    min_baseline_acc: float = 0.6
) -> plt.Figure:
    """
    Create a summary table and visualization of privacy costs across all matched pairs.
    
    Configurations where baseline accuracy is below min_baseline_acc are excluded
    from the main analysis, as they don't represent a fair comparison (baseline
    didn't learn, so privacy cost cannot be meaningfully measured).
    
    Args:
        baseline_history: DataFrame with baseline training history
        dp_history: DataFrame with DP-SGD training history (should include 'epsilon' column)
        save_path: Optional path to save the plot
        min_baseline_acc: Minimum baseline accuracy threshold (default: 0.6).
                         Configurations below this are excluded from analysis.
        
    Returns:
        Matplotlib figure object
    """
    # Extract final metrics
    baseline_accs = extract_final_test_accuracies(baseline_history)
    dp_accs = extract_final_test_accuracies(dp_history)
    
    # Match configurations
    def get_config_key(history_df: pd.DataFrame, run_name: str) -> Optional[Tuple]:
        run_data = history_df[history_df['run'] == run_name]
        if len(run_data) == 0:
            return None
        row = run_data.iloc[0]
        return (float(row.get('lr', 0)), int(row.get('batch_size', 0)), float(row.get('weight_decay', 0)))
    
    matched_pairs = []
    excluded_pairs = []
    baseline_keys = {}
    for run in baseline_history['run'].unique():
        key = get_config_key(baseline_history, run)
        if key:
            baseline_keys[key] = run
    
    for run in dp_history['run'].unique():
        key = get_config_key(dp_history, run)
        if key and key in baseline_keys:
            baseline_run = baseline_keys[key]
            if baseline_run in baseline_accs and run in dp_accs:
                baseline_acc = baseline_accs[baseline_run]
                dp_acc = dp_accs[run]
                
                # Get epsilon if available
                dp_data = dp_history[dp_history['run'] == run]
                epsilon = dp_data['epsilon'].iloc[-1] if 'epsilon' in dp_data.columns and len(dp_data) > 0 else None
                
                pair_data = {
                    'config_key': key,
                    'baseline_run': baseline_run,
                    'dp_run': run,
                    'baseline_acc': baseline_acc,
                    'dp_acc': dp_acc,
                    'privacy_cost': baseline_acc - dp_acc,
                    'epsilon': epsilon,
                    'label': f"LR={key[0]:g}, BS={key[1]}, WD={key[2]:g}"
                }
                
                # Filter: only include if baseline learned (above threshold)
                if baseline_acc >= min_baseline_acc:
                    matched_pairs.append(pair_data)
                else:
                    excluded_pairs.append(pair_data)
    
    # Report excluded configurations
    if excluded_pairs:
        print(f"\n⚠️  Excluded {len(excluded_pairs)} configuration(s) where baseline accuracy < {min_baseline_acc}:")
        for p in excluded_pairs:
            eps_str = f", ε={p['epsilon']:.2f}" if p['epsilon'] is not None else ""
            print(f"   {p['label']}: Baseline={p['baseline_acc']:.3f}, DP-SGD={p['dp_acc']:.3f}{eps_str} "
                  f"(Baseline did not learn - not a fair comparison)")
        print()
    
    if len(matched_pairs) == 0:
        print("⚠️  No matched pairs found.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No matched pairs found", ha='center', va='center', fontsize=14)
        return fig
    
    # Create summary table
    print("\n" + "="*80)
    print("📊 MATCHED PAIR ANALYSIS: Privacy Cost Summary")
    print("="*80)
    print(f"{'Configuration':<30} {'Baseline Acc':<15} {'DP-SGD Acc':<15} {'Cost':<12} {'Epsilon':<10}")
    print("-"*80)
    for p in sorted(matched_pairs, key=lambda x: x['privacy_cost'], reverse=True):
        eps_str = f"{p['epsilon']:.2f}" if p['epsilon'] is not None else "N/A"
        print(f"{p['label']:<30} {p['baseline_acc']:<15.4f} {p['dp_acc']:<15.4f} "
              f"{p['privacy_cost']:<12.4f} {eps_str:<10}")
    print("="*80)
    print(f"Average Privacy Cost: {np.mean([p['privacy_cost'] for p in matched_pairs]):.4f}")
    print(f"Max Privacy Cost: {max([p['privacy_cost'] for p in matched_pairs]):.4f}")
    print(f"Min Privacy Cost: {min([p['privacy_cost'] for p in matched_pairs]):.4f}")
    print()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_pairs = len(matched_pairs)
    x = np.arange(n_pairs)
    privacy_costs = [p['privacy_cost'] for p in matched_pairs]
    labels = [p['label'] for p in matched_pairs]
    
    # Color by cost magnitude
    colors = ['#c0392b' if cost > 0.05 else '#e67e22' if cost > 0.02 else '#f39c12' 
              for cost in privacy_costs]
    
    bars = ax.bar(x, privacy_costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Privacy Cost (Accuracy Drop)', fontsize=12, fontweight='bold')
    ax.set_title('Privacy Cost Across Matched Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels
    for bar, cost in zip(bars, privacy_costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.3f}',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Privacy cost summary saved to: {save_path}")
    
    plt.show()
    return fig


def plot_training_dynamics_comparison(
    baseline_history: pd.DataFrame,
    dp_history: pd.DataFrame,
    save_path: Optional[str] = None,
    max_configs: int = 3,
    smooth: bool = True,
    smoothing_method: str = 'ema',
    smoothing_param: float = 0.3
) -> plt.Figure:
    """
    Plot training dynamics comparison showing baseline vs DP-SGD side-by-side.
    
    This visualization helps teach DP-SGD by showing:
    - How DP-SGD converges slower (noise in gradients)
    - How final accuracy is typically lower (privacy cost)
    - Training dynamics differences for matched configurations
    
    Args:
        baseline_history: DataFrame with baseline training history
        dp_history: DataFrame with DP-SGD training history
        save_path: Optional path to save the plot
        max_configs: Maximum number of configurations to show (default: 3)
        smooth: Whether to apply smoothing to curves (default: True)
        smoothing_method: Method to use ('ema' for exponential moving average, 
                        'sma' for simple moving average, default: 'ema')
        smoothing_param: Smoothing parameter. For EMA: alpha (0.1-0.9, default 0.3).
                         For SMA: window size (3-7, default 3). Lower EMA alpha = more smoothing.
        
    Returns:
        Matplotlib figure object
    """
    # Match configurations by hyperparameters
    def get_config_key(history_df: pd.DataFrame, run_name: str) -> Optional[Tuple]:
        run_data = history_df[history_df['run'] == run_name]
        if len(run_data) == 0:
            return None
        row = run_data.iloc[0]
        return (float(row.get('lr', 0)), int(row.get('batch_size', 0)), float(row.get('weight_decay', 0)))
    
    # Find matched pairs
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
    
    if len(matched_pairs) == 0:
        print("⚠️  No matched pairs found.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No matched pairs found", ha='center', va='center', fontsize=14)
        return fig
    
    # Limit to max_configs (select ones with best baseline accuracy)
    baseline_accs = extract_final_test_accuracies(baseline_history)
    matched_pairs.sort(key=lambda x: baseline_accs.get(x['baseline_run'], 0), reverse=True)
    matched_pairs = matched_pairs[:max_configs]
    
    # Create figure with subplots: 2 rows (loss, accuracy) x max_configs columns
    fig, axes = plt.subplots(2, len(matched_pairs), figsize=(5*len(matched_pairs), 8))
    if len(matched_pairs) == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, pair in enumerate(matched_pairs):
        baseline_run = pair['baseline_run']
        dp_run = pair['dp_run']
        
        # Get training data
        baseline_data = baseline_history[baseline_history['run'] == baseline_run].sort_values('epoch')
        dp_data = dp_history[dp_history['run'] == dp_run].sort_values('epoch')
        
        epochs_baseline = baseline_data['epoch']
        epochs_dp = dp_data['epoch']
        
        # Apply smoothing if requested
        if smooth:
            if smoothing_method == 'ema':
                baseline_train_loss_smooth = exponential_moving_average(
                    baseline_data['train_loss'].values, alpha=smoothing_param
                )
                dp_train_loss_smooth = exponential_moving_average(
                    dp_data['train_loss'].values, alpha=smoothing_param
                )
                baseline_val_acc_smooth = exponential_moving_average(
                    baseline_data['val_acc'].values, alpha=smoothing_param
                )
                dp_val_acc_smooth = exponential_moving_average(
                    dp_data['val_acc'].values, alpha=smoothing_param
                )
            elif smoothing_method == 'sma':
                window = int(smoothing_param) if smoothing_param >= 3 else 3
                baseline_train_loss_smooth = simple_moving_average(
                    baseline_data['train_loss'].values, window=window
                )
                dp_train_loss_smooth = simple_moving_average(
                    dp_data['train_loss'].values, window=window
                )
                baseline_val_acc_smooth = simple_moving_average(
                    baseline_data['val_acc'].values, window=window
                )
                dp_val_acc_smooth = simple_moving_average(
                    dp_data['val_acc'].values, window=window
                )
            else:
                raise ValueError(f"Unknown smoothing method: {smoothing_method}. Use 'ema' or 'sma'.")
        else:
            baseline_train_loss_smooth = baseline_data['train_loss'].values
            dp_train_loss_smooth = dp_data['train_loss'].values
            baseline_val_acc_smooth = baseline_data['val_acc'].values
            dp_val_acc_smooth = dp_data['val_acc'].values
        
        # Plot 1: Training Loss (top row)
        ax_loss = axes[0, col_idx]
        ax_loss.plot(epochs_baseline, baseline_train_loss_smooth, 
                    label='Baseline', color='#3498db', linewidth=2.5, marker='o', markersize=4)
        ax_loss.plot(epochs_dp, dp_train_loss_smooth, 
                    label='DP-SGD', color='#e74c3c', linewidth=2.5, marker='s', markersize=4, linestyle='--')
        ax_loss.set_title(f'Training Loss\n{pair["label"]}', fontsize=11, fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=9)
        ax_loss.grid(True, alpha=0.3)
        
        # Plot 2: Validation Accuracy (bottom row)
        ax_acc = axes[1, col_idx]
        ax_acc.plot(epochs_baseline, baseline_val_acc_smooth, 
                   label='Baseline', color='#3498db', linewidth=2.5, marker='o', markersize=4)
        ax_acc.plot(epochs_dp, dp_val_acc_smooth, 
                   label='DP-SGD', color='#e74c3c', linewidth=2.5, marker='s', markersize=4, linestyle='--')
        
        # Add final accuracy annotations
        final_baseline = baseline_data['val_acc'].iloc[-1]
        final_dp = dp_data['val_acc'].iloc[-1]
        ax_acc.axhline(y=final_baseline, color='#3498db', linestyle=':', alpha=0.5, linewidth=1)
        ax_acc.axhline(y=final_dp, color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1)
        ax_acc.text(len(epochs_baseline), final_baseline, f' {final_baseline:.3f}', 
                   va='center', color='#3498db', fontsize=8, fontweight='bold')
        ax_acc.text(len(epochs_dp), final_dp, f' {final_dp:.3f}', 
                   va='center', color='#e74c3c', fontsize=8, fontweight='bold')
        
        ax_acc.set_title(f'Validation Accuracy\n{pair["label"]}', fontsize=11, fontweight='bold')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend(fontsize=9)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim([0, 1])
    
    plt.suptitle('Training Dynamics: Baseline vs DP-SGD', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training dynamics comparison saved to: {save_path}")
    
    plt.show()
    return fig


def plot_privacy_accuracy_tradeoff(
    baseline_history: pd.DataFrame,
    dp_history: pd.DataFrame,
    save_path: Optional[str] = None,
    min_baseline_acc: float = 0.6
) -> plt.Figure:
    """
    Plot privacy-accuracy trade-off: epsilon (ε) vs final accuracy.
    
    This visualization teaches the core DP-SGD concept: stronger privacy
    (lower epsilon) typically comes at the cost of lower accuracy.
    
    Args:
        baseline_history: DataFrame with baseline training history
        dp_history: DataFrame with DP-SGD training history (must include 'epsilon' column)
        save_path: Optional path to save the plot
        min_baseline_acc: Minimum baseline accuracy threshold (default: 0.6)
        
    Returns:
        Matplotlib figure object
    """
    # Extract final accuracies
    baseline_accs = extract_final_test_accuracies(baseline_history)
    dp_accs = extract_final_test_accuracies(dp_history)
    
    # Match configurations
    def get_config_key(history_df: pd.DataFrame, run_name: str) -> Optional[Tuple]:
        run_data = history_df[history_df['run'] == run_name]
        if len(run_data) == 0:
            return None
        row = run_data.iloc[0]
        return (float(row.get('lr', 0)), int(row.get('batch_size', 0)), float(row.get('weight_decay', 0)))
    
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
            if baseline_run in baseline_accs and run in dp_accs:
                baseline_acc = baseline_accs[baseline_run]
                dp_acc = dp_accs[run]
                
                # Get epsilon
                dp_data = dp_history[dp_history['run'] == run]
                if 'epsilon' in dp_data.columns and len(dp_data) > 0:
                    epsilon = dp_data['epsilon'].iloc[-1]
                else:
                    continue
                
                # Only include if baseline learned
                if baseline_acc >= min_baseline_acc:
                    matched_pairs.append({
                        'baseline_acc': baseline_acc,
                        'dp_acc': dp_acc,
                        'epsilon': epsilon,
                        'privacy_cost': baseline_acc - dp_acc,
                        'label': f"LR={key[0]:g}, BS={key[1]}, WD={key[2]:g}",
                        'lr': key[0],
                        'batch_size': key[1],
                        'weight_decay': key[2]
                    })
    
    if len(matched_pairs) == 0:
        print("⚠️  No matched pairs with epsilon data found.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No matched pairs found", ha='center', va='center', fontsize=14)
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract data
    epsilons = [p['epsilon'] for p in matched_pairs]
    dp_accuracies = [p['dp_acc'] for p in matched_pairs]
    baseline_accs_vals = [p['baseline_acc'] for p in matched_pairs]
    labels = [p['label'] for p in matched_pairs]
    
    # Color by learning rate for visual distinction
    lrs = [p['lr'] for p in matched_pairs]
    unique_lrs = sorted(set(lrs))
    colors = plt.cm.tab10(range(len(unique_lrs)))
    lr_to_color = {lr: colors[i] for i, lr in enumerate(unique_lrs)}
    point_colors = [lr_to_color[lr] for lr in lrs]
    
    # Plot DP-SGD points
    scatter = ax.scatter(epsilons, dp_accuracies, c=point_colors, s=150, alpha=0.7, 
                        edgecolors='black', linewidths=1.5, zorder=3)
    
    # Add baseline reference line (average baseline accuracy)
    avg_baseline = np.mean(baseline_accs_vals)
    ax.axhline(y=avg_baseline, color='#3498db', linestyle='--', linewidth=2, 
              label=f'Avg Baseline Accuracy ({avg_baseline:.3f})', zorder=1)
    
    # Add individual baseline points (lighter, behind)
    for i, (eps, baseline_acc) in enumerate(zip(epsilons, baseline_accs_vals)):
        ax.scatter(eps, baseline_acc, c=point_colors[i], s=80, alpha=0.3, 
                  marker='x', linewidths=2, zorder=2)
    
    # Add labels for each point
    for i, (eps, acc, label) in enumerate(zip(epsilons, dp_accuracies, labels)):
        ax.annotate(label, (eps, acc), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    # Styling
    ax.set_xlabel('Privacy Budget (Epsilon ε)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Privacy-Accuracy Trade-off: DP-SGD', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # Add annotation explaining the trade-off
    ax.text(0.02, 0.98, 'Lower ε = Stronger Privacy\nLower Accuracy', 
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Privacy-accuracy trade-off plot saved to: {save_path}")
    
    plt.show()
    return fig
