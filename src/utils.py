import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            bsz = images.size(0)
            total_loss += loss.item() * bsz
            total_acc += accuracy(logits, targets) * bsz
            total_n += bsz
    return total_loss / max(1, total_n), total_acc / max(1, total_n)


