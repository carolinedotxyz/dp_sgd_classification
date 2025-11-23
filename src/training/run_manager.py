"""
Run management functionality for experiment tracking and artifact saving.

This module handles the creation of run directories, saving of model artifacts,
configuration files, and summary statistics for reproducible experiments.
"""

import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from IPython.display import display, HTML, clear_output


def create_run_directory(run_name: str, sweep_id: str, base_dir: str = "runs/baseline") -> Path:
    """
    Create a directory for saving run artifacts.
    
    Args:
        run_name: Name of the current run
        sweep_id: Identifier for the hyperparameter sweep
        base_dir: Base directory for storing runs
        
    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{timestamp}_{run_name}_{sweep_id}"
    run_dir = Path(base_dir) / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: Path, 
    model: torch.nn.Module, 
    run_config: Dict[str, Any], 
    run_results: Dict[str, Any], 
    history_df: pd.DataFrame
) -> Path:
    """
    Save all artifacts for a training run.
    
    Args:
        run_dir: Directory to save artifacts to
        model: Trained model to save
        run_config: Configuration used for this run
        run_results: Results dictionary with metrics
        history_df: DataFrame with training history
        
    Returns:
        Path to the run directory
    """
    
    # Save model
    model_path = run_dir / "best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': run_config,
        'results': run_results
    }, model_path)
    
    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Save summary
    summary_path = run_dir / "summary.json"
    summary_data = {
        'test_acc': run_results['test_acc'],
        'val_acc': run_results['val_acc'],
        'test_loss': run_results.get('test_loss', 0.0),
        'val_loss': run_results.get('val_loss', 0.0),
        'total_time': run_results['total_time'],
        'best_val_acc': history_df[history_df['run'] == run_config['run_name']]['val_acc'].max(),
        'final_train_acc': history_df[history_df['run'] == run_config['run_name']]['train_acc'].iloc[-1],
        'final_train_loss': history_df[history_df['run'] == run_config['run_name']]['train_loss'].iloc[-1]
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Save detailed history
    history_path = run_dir / "history.csv"
    run_history = history_df[history_df['run'] == run_config['run_name']].copy()
    run_history.to_csv(history_path, index=False)
    
    return run_dir


def display_run_summary(run_results: List[Dict[str, Any]], best_run: str, saved_runs: List[Path]):
    """
    Display a clean run summary with best run highlighted.
    
    Args:
        run_results: List of run result dictionaries
        best_run: Name of the best performing run
        saved_runs: List of paths to saved run directories
    """
    print("\n" + "="*80)
    print("🏆 TRAINING RUNS COMPLETED")
    print("="*80)
    
    for i, result in enumerate(run_results, 1):
        status_icon = "🥇" if result['run_name'] == best_run else "✅"
        
        print(f"{status_icon} Run {i}: {result['run_name']}")
        print(f"   Test Acc: {result['test_acc']:.4f} | Val Acc: {result['val_acc']:.4f} | Time: {result['total_time']:.1f}s")
        print()
    
    print(f"🎯 Best Run: {best_run} (Test Acc: {max(r['test_acc'] for r in run_results):.4f})")
    print("\n💾 Saved Run Artifacts:")
    for run_dir in saved_runs:
        # Resolve to absolute path first
        abs_run_dir = run_dir.resolve()
        cwd = Path.cwd().resolve()
        
        # Try to show relative path if under cwd, otherwise show absolute
        try:
            relative_path = abs_run_dir.relative_to(cwd)
            print(f"   📁 {relative_path}")
        except ValueError:
            # Path is not under current working directory, show absolute path
            print(f"   📁 {abs_run_dir}")
    print("="*80)
