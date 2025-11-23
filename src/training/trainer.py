"""
Core training functionality for neural network experiments.

This module contains the fundamental training loop and evaluation functions
used across different experiment types (SGD, DP-SGD, etc.).
"""

import torch
from tqdm.auto import tqdm
from typing import Tuple, Optional, Any
from ..core.utils import accuracy, evaluate


def train_one_epoch_with_progress(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    run_name: str,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train one epoch with clean progress bar.
    
    Args:
        model: The neural network model to train
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        epoch: Current epoch number (1-indexed)
        total_epochs: Total number of epochs
        run_name: Name of the current run (for display)
        device: Device to run training on
        
    Returns:
        Tuple of (average_loss, average_accuracy) for this epoch
    """
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    
    # Create clean progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", 
                leave=False, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        bsz = images.size(0)
        total_loss += float(loss.item()) * bsz
        total_acc += accuracy(logits, targets) * bsz
        total_n += bsz
        
        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{total_loss/max(1,total_n):.3f}',
            'acc': f'{total_acc/max(1,total_n):.3f}'
        })
    
    return total_loss / max(1, total_n), total_acc / max(1, total_n)


def train_one_epoch_dp_sgd_with_progress(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    run_name: str,
    device: torch.device,
    accountant: Optional[Any] = None,
    target_delta: float = 1e-5
) -> Tuple[float, float, Optional[float]]:
    """
    Train one epoch with DP-SGD and clean progress bar.
    
    Args:
        model: The neural network model to train (wrapped with Opacus PrivacyEngine)
        loader: Training data loader (private DP loader from Opacus)
        criterion: Loss function
        optimizer: Optimizer (wrapped with Opacus PrivacyEngine)
        epoch: Current epoch number (1-indexed)
        total_epochs: Total number of epochs
        run_name: Name of the current run (for display)
        device: Device to run training on
        accountant: PrivacyEngine accountant for epsilon computation
        target_delta: Delta for privacy accounting
        
    Returns:
        Tuple of (average_loss, average_accuracy, epsilon) for this epoch
    """
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    
    # Create clean progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", 
                leave=False, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        bsz = images.size(0)
        total_loss += float(loss.item()) * bsz
        total_acc += accuracy(logits, targets) * bsz
        total_n += bsz
        
        # Compute epsilon if accountant is available
        eps = None
        if accountant is not None:
            try:
                eps = accountant.get_epsilon(delta=target_delta)
            except Exception:
                pass
        
        # Update progress bar with current metrics
        eps_str = f"eps={eps:.2f}" if eps is not None else ""
        pbar.set_postfix({
            'loss': f'{total_loss/max(1,total_n):.3f}',
            'acc': f'{total_acc/max(1,total_n):.3f}',
            **({'eps': f'{eps:.2f}'} if eps is not None else {})
        })
    
    # Final epsilon computation
    final_eps = None
    if accountant is not None:
        try:
            final_eps = accountant.get_epsilon(delta=target_delta)
        except Exception:
            pass
    
    return total_loss / max(1, total_n), total_acc / max(1, total_n), final_eps
