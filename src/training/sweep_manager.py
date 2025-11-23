"""
Hyperparameter sweep orchestration for training experiments.

This module handles the execution of hyperparameter sweeps, managing multiple
training runs with different configurations and collecting results.
"""

import time
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from .trainer import train_one_epoch_with_progress, train_one_epoch_dp_sgd_with_progress
from .run_manager import create_run_directory, save_run_artifacts, display_run_summary
from ..core.utils import evaluate

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    PrivacyEngine = None


def run_sgd_sweep_with_progress(
    grid: List[Dict[str, Any]], 
    epochs: int = 5,
    num_workers: int = 0, 
    aug_flip: bool = True, 
    aug_jitter: bool = False,
    aug_rotation: bool = False,
    rotation_degrees: float = 10.0,
    sweep_id: Optional[str] = None,
    device: torch.device = None,
    make_model_fn: Callable = None,
    make_loaders_fn: Callable = None,
    base_run_dir: str = "runs/baseline"
) -> pd.DataFrame:
    """
    Run SGD sweep with clean progress tracking, real-time summary, and run saving.
    
    Args:
        grid: List of hyperparameter dictionaries to sweep over
        epochs: Number of training epochs per run
        num_workers: Number of data loader workers
        aug_flip: Whether to use horizontal flip augmentation
        aug_jitter: Whether to use color jitter augmentation
        aug_rotation: Whether to use random rotation augmentation
        rotation_degrees: Maximum rotation angle in degrees (only used if aug_rotation=True)
        sweep_id: Identifier for this sweep (auto-generated if None)
        device: Device to run training on
        make_model_fn: Function to create model instances
        make_loaders_fn: Function to create data loaders
        base_run_dir: Base directory for saving runs
        
    Returns:
        DataFrame with training history for all runs
    """
    
    rows: List[Dict[str, Any]] = []
    run_results: List[Dict] = []
    saved_runs: List[Path] = []
    best_test_acc = 0.0
    
    # Generate sweep ID if not provided
    if sweep_id is None:
        sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n🚀 Starting Baseline (AdamW) Hyperparameter Sweep ({len(grid)} runs)")
    print(f"📋 Sweep ID: {sweep_id}")
    print("="*60)

    for i, hp in enumerate(grid, start=1):
        lr = float(hp.get("lr", 1e-2))
        batch_size = int(hp.get("batch_size", 64))
        weight_decay = float(hp.get("weight_decay", 0.0))
        run_name = hp.get("run_name", f"baseline_lr{lr:g}_bs{batch_size}_wd{weight_decay:g}")

        print(f"\n📊 Run {i}/{len(grid)}: {run_name}")
        print(f"   Config: lr={lr}, bs={batch_size}, wd={weight_decay}")
        
        # Create data loaders and model
        train_loader, val_loader, test_loader = make_loaders_fn(
            batch_size=batch_size, 
            num_workers=num_workers,
            aug_flip=aug_flip, 
            aug_jitter=aug_jitter
        )
        model = make_model_fn()
        criterion = torch.nn.CrossEntropyLoss()
        # Use AdamW for fair comparison with DP-SGD (both handle weight_decay the same way)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )

        run_start_time = time.time()
        
        # Training loop
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            # Train with progress bar
            train_loss, train_acc = train_one_epoch_with_progress(
                model, train_loader, criterion, optimizer, epoch, epochs, run_name, device
            )
            
            # Validation (no progress bar needed)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            dur = time.time() - t0

            row = {
                "run": run_name,
                "epoch": epoch,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_s": dur,
            }
            rows.append(row)

        # Final test eval for the run
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        total_time = time.time() - run_start_time
        
        # Determine trend
        trend = ""
        if test_acc > best_test_acc:
            trend = "NEW BEST!"
            best_test_acc = test_acc
        elif test_acc < best_test_acc * 0.95:  # 5% worse
            trend = ""
        
        # Store run result
        run_result = {
            'run_name': run_name,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'val_loss': val_loss,
            'total_time': total_time,
            'trend': trend,
            'status': 'completed'
        }
        run_results.append(run_result)
        
        # Save run artifacts
        run_dir = create_run_directory(run_name, sweep_id, base_run_dir)
        run_config = {
            'run_name': run_name,
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'optimizer': 'adamw',  # Using AdamW for fair comparison with DP-SGD
            'sweep_id': sweep_id,
            'timestamp': datetime.now().isoformat()
        }
        
        saved_run_dir = save_run_artifacts(run_dir, model, run_config, run_result, pd.DataFrame(rows))
        saved_runs.append(saved_run_dir)
        
        # Display completion message
        print(f"✅ Completed: test_acc={test_acc:.4f}, val_acc={val_acc:.4f}, time={total_time:.1f}s")
        
        # Show real-time progress summary
        if i < len(grid):  # Don't show summary after last run
            print(f"\n📈 Progress: {i}/{len(grid)} runs completed")

    # Find best run
    best_run = max(run_results, key=lambda x: x['test_acc'])['run_name']
    
    # Display final summary
    display_run_summary(run_results, best_run, saved_runs)
    
    return pd.DataFrame(rows)


def run_dp_sgd_sweep_with_progress(
    grid: List[Dict[str, Any]], 
    epochs: int = 5,
    num_workers: int = 0, 
    aug_flip: bool = True, 
    aug_jitter: bool = False,
    aug_rotation: bool = False,
    rotation_degrees: float = 10.0,
    sweep_id: Optional[str] = None,
    device: torch.device = None,
    make_model_fn: Callable = None,
    make_loaders_fn: Callable = None,
    base_run_dir: str = "runs/dp_sgd",
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.1,
    target_delta: float = 1e-5
) -> pd.DataFrame:
    """
    Run DP-SGD sweep with clean progress tracking, real-time summary, and run saving.
    
    Args:
        grid: List of hyperparameter dictionaries to sweep over
        epochs: Number of training epochs per run
        num_workers: Number of data loader workers
        aug_flip: Whether to use horizontal flip augmentation
        aug_jitter: Whether to use color jitter augmentation
        aug_rotation: Whether to use random rotation augmentation
        rotation_degrees: Maximum rotation angle in degrees (only used if aug_rotation=True)
        sweep_id: Identifier for this sweep (auto-generated if None)
        device: Device to run training on
        make_model_fn: Function to create model instances
        make_loaders_fn: Function to create data loaders
        base_run_dir: Base directory for saving runs
        max_grad_norm: Per-sample gradient clipping norm for DP-SGD
        noise_multiplier: Gaussian noise multiplier for DP-SGD
        target_delta: Target delta for privacy accounting
        
    Returns:
        DataFrame with training history for all runs (includes epsilon column)
    """
    
    if not OPACUS_AVAILABLE:
        raise ImportError(
            "Opacus is required for DP-SGD training. "
            "Install it with: pip install opacus"
        )
    
    rows: List[Dict[str, Any]] = []
    run_results: List[Dict] = []
    saved_runs: List[Path] = []
    best_test_acc = 0.0
    
    # Generate sweep ID if not provided
    if sweep_id is None:
        sweep_id = f"dp_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n🔒 Starting DP-SGD Hyperparameter Sweep ({len(grid)} runs)")
    print(f"📋 Sweep ID: {sweep_id}")
    print(f"🔐 Privacy params: noise_multiplier={noise_multiplier}, max_grad_norm={max_grad_norm}, delta={target_delta}")
    print("="*60)

    for i, hp in enumerate(grid, start=1):
        lr = float(hp.get("lr", 1e-3))
        batch_size = int(hp.get("batch_size", 64))
        weight_decay = float(hp.get("weight_decay", 0.0))
        run_name = hp.get("run_name", f"dp_lr{lr:g}_bs{batch_size}_wd{weight_decay:g}_nm{noise_multiplier:g}")

        print(f"\n📊 Run {i}/{len(grid)}: {run_name}")
        print(f"   Config: lr={lr}, bs={batch_size}, wd={weight_decay}, noise_mult={noise_multiplier}")
        
        # Create data loaders and model
        train_loader, val_loader, test_loader = make_loaders_fn(
            batch_size=batch_size, 
            num_workers=num_workers,
            aug_flip=aug_flip, 
            aug_jitter=aug_jitter,
            aug_rotation=aug_rotation,
            rotation_degrees=rotation_degrees
        )
        model = make_model_fn()
        criterion = torch.nn.CrossEntropyLoss()
        # Use AdamW (better than Adam, properly handles weight_decay like SGD)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Wrap model and optimizer with PrivacyEngine
        privacy_engine = PrivacyEngine(accountant="rdp")
        model, optimizer, train_dp_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        accountant = privacy_engine.accountant

        # Determinism settings for reproducibility
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        run_start_time = time.time()
        best_val_acc = -float("inf")
        best_state = None
        
        # Training loop
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            # Train with progress bar
            train_loss, train_acc, eps = train_one_epoch_dp_sgd_with_progress(
                model, train_dp_loader, criterion, optimizer, epoch, epochs, run_name, device,
                accountant=accountant, target_delta=target_delta
            )
            
            # Validation (no progress bar needed)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            dur = time.time() - t0
            
            # Track best validation model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            
            # Get final epsilon for this epoch
            final_eps = eps if eps is not None else accountant.get_epsilon(delta=target_delta)

            row = {
                "run": run_name,
                "epoch": epoch,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "noise_multiplier": noise_multiplier,
                "max_grad_norm": max_grad_norm,
                "epsilon": final_eps,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "time_s": dur,
            }
            rows.append(row)

        # Restore best validation model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Final test eval for the run
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        final_eps = accountant.get_epsilon(delta=target_delta)
        total_time = time.time() - run_start_time
        
        # Determine trend
        trend = ""
        if test_acc > best_test_acc:
            trend = "NEW BEST!"
            best_test_acc = test_acc
        elif test_acc < best_test_acc * 0.95:  # 5% worse
            trend = ""
        
        # Store run result
        run_result = {
            'run_name': run_name,
            'test_acc': test_acc,
            'val_acc': best_val_acc,
            'test_loss': test_loss,
            'val_loss': val_loss,
            'epsilon': final_eps,
            'total_time': total_time,
            'trend': trend,
            'status': 'completed'
        }
        run_results.append(run_result)
        
        # Save run artifacts
        run_dir = create_run_directory(run_name, sweep_id, base_run_dir)
        run_config = {
            'run_name': run_name,
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
            'target_delta': target_delta,
            'epsilon': final_eps,
            'epochs': epochs,
            'optimizer': 'adamw_dp',  # AdamW with DP-SGD privacy mechanisms
            'sweep_id': sweep_id,
            'timestamp': datetime.now().isoformat()
        }
        
        saved_run_dir = save_run_artifacts(run_dir, model, run_config, run_result, pd.DataFrame(rows))
        saved_runs.append(saved_run_dir)
        
        # Display completion message
        print(f"✅ Completed: test_acc={test_acc:.4f}, val_acc={best_val_acc:.4f}, eps={final_eps:.2f}, time={total_time:.1f}s")
        
        # Show real-time progress summary
        if i < len(grid):  # Don't show summary after last run
            print(f"\n📈 Progress: {i}/{len(grid)} runs completed")

    # Find best run
    best_run = max(run_results, key=lambda x: x['test_acc'])['run_name']
    
    # Display final summary
    display_run_summary(run_results, best_run, saved_runs)
    
    return pd.DataFrame(rows)


def run_dp_sgd_privacy_parameter_sweep(
    grid: List[Dict[str, Any]],
    privacy_combinations: List[Dict[str, float]],
    epochs: int = 5,
    num_workers: int = 0,
    aug_flip: bool = True,
    aug_jitter: bool = False,
    aug_rotation: bool = False,
    rotation_degrees: float = 10.0,
    device: torch.device = None,
    make_model_fn: Callable = None,
    make_loaders_fn: Callable = None,
    base_run_dir: str = "runs/dp_sgd",
    target_delta: float = 1e-5
) -> Dict[str, pd.DataFrame]:
    """
    Run DP-SGD sweeps with multiple privacy parameter combinations.
    
    This function systematically tests different combinations of noise_multiplier
    and max_grad_norm to find settings that enable learning.
    
    Args:
        grid: List of hyperparameter dictionaries (lr, batch_size, weight_decay)
        privacy_combinations: List of dicts with 'noise_multiplier' and 'max_grad_norm'
        epochs: Number of training epochs per run
        num_workers: Number of data loader workers
        aug_flip: Whether to use horizontal flip augmentation
        aug_jitter: Whether to use color jitter augmentation
        aug_rotation: Whether to use random rotation augmentation
        rotation_degrees: Maximum rotation angle in degrees (only used if aug_rotation=True)
        device: Device to run training on
        make_model_fn: Function to create model instances
        make_loaders_fn: Function to create data loaders
        base_run_dir: Base directory for saving runs
        target_delta: Target delta for privacy accounting
        
    Returns:
        Dictionary mapping privacy combination names to DataFrames of results
        Each key is formatted as "nm{noise_mult}_mgn{max_grad_norm}"
        
    Example:
        privacy_combinations = [
            {'noise_multiplier': 0.7, 'max_grad_norm': 1.0},
            {'noise_multiplier': 0.5, 'max_grad_norm': 1.5},
        ]
        results = run_dp_sgd_privacy_parameter_sweep(
            grid=hp_grid,
            privacy_combinations=privacy_combinations,
            epochs=5,
            device=device,
            make_model_fn=make_model,
            make_loaders_fn=make_loaders
        )
    """
    if not OPACUS_AVAILABLE:
        raise ImportError(
            "Opacus is required for DP-SGD training. "
            "Install it with: pip install opacus"
        )
    
    results = {}
    total_experiments = len(privacy_combinations)
    
    print(f"\n🔬 Starting Privacy Parameter Exploration")
    print(f"📊 Testing {total_experiments} privacy parameter combinations")
    print(f"📋 Each combination will run {len(grid)} hyperparameter configurations")
    print(f"⏱️  Total runs: {total_experiments * len(grid)}")
    print("="*80)
    
    for exp_idx, privacy_params in enumerate(privacy_combinations, start=1):
        noise_mult = privacy_params['noise_multiplier']
        max_grad_norm = privacy_params['max_grad_norm']
        
        # Create unique sweep ID for this privacy combination
        combo_name = f"nm{noise_mult:g}_mgn{max_grad_norm:g}"
        sweep_id = f"privacy_exploration_{combo_name}"
        
        print(f"\n{'='*80}")
        print(f"🔬 Experiment {exp_idx}/{total_experiments}: {combo_name}")
        print(f"   noise_multiplier = {noise_mult}")
        print(f"   max_grad_norm = {max_grad_norm}")
        print(f"{'='*80}\n")
        
        # Run sweep with these privacy parameters
        df = run_dp_sgd_sweep_with_progress(
            grid=grid,
            epochs=epochs,
            num_workers=num_workers,
            aug_flip=aug_flip,
            aug_jitter=aug_jitter,
            aug_rotation=aug_rotation,
            rotation_degrees=rotation_degrees,
            sweep_id=sweep_id,
            device=device,
            make_model_fn=make_model_fn,
            make_loaders_fn=make_loaders_fn,
            base_run_dir=base_run_dir,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_mult,
            target_delta=target_delta
        )
        
        results[combo_name] = df
        
        # Quick summary for this combination
        # Note: test_acc is not in per-epoch DataFrame, use final val_acc as proxy
        if len(df) > 0:
            # Get final validation accuracy for each run (last epoch per run)
            final_val_accs = df.groupby('run')['val_acc'].last()
            best_acc = final_val_accs.max()
            avg_acc = final_val_accs.mean()
            print(f"\n📊 Summary for {combo_name}:")
            print(f"   Best final validation accuracy: {best_acc:.4f}")
            print(f"   Average final validation accuracy: {avg_acc:.4f}")
            print(f"   Runs with learning (>60% acc): {(final_val_accs > 0.60).sum()}/{len(final_val_accs)}")
    
    # Final comparison
    print(f"\n{'='*80}")
    print("📈 Privacy Parameter Comparison Summary")
    print(f"{'='*80}\n")
    
    comparison_data = []
    for combo_name, df in results.items():
        if len(df) > 0:
            # Get final validation accuracy for each run (last epoch per run)
            final_val_accs = df.groupby('run')['val_acc'].last()
            comparison_data.append({
                'combination': combo_name,
                'best_val_acc': final_val_accs.max(),
                'avg_val_acc': final_val_accs.mean(),
                'runs_above_60pct': (final_val_accs > 0.60).sum(),
                'total_runs': len(final_val_accs)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('best_val_acc', ascending=False)
        print(comparison_df.to_string(index=False))
        print()
    
    return results
