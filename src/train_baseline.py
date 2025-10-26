from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import json, time, os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

from .config import BaselineConfig
from .utils import accuracy, evaluate


def _get_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def train_baseline(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                   device: torch.device, cfg: BaselineConfig) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]]]:
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    # Determinism settings (MPS-safe): disable cuDNN benchmark on CUDA; enable deterministic where supported
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        # This is a global flag; may raise on unsupported ops; keep in try/except
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = _get_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay)

    run_dir = None
    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        run_dir = os.path.join(cfg.output_dir, cfg.run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)

    history: List[Dict[str, Any]] = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, epoch_acc, epoch_n = 0.0, 0.0, 0
        start = time.time()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            bsz = images.size(0)
            epoch_loss += loss.item() * bsz
            epoch_acc += accuracy(logits, targets) * bsz
            epoch_n += bsz
        train_loss = epoch_loss / max(1, epoch_n)
        train_acc = epoch_acc / max(1, epoch_n)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        dur = time.time() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "time_s": dur})
        print(f"[Baseline][{cfg.run_name}][Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={dur:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            if cfg.save_best and run_dir:
                torch.save(best_state, os.path.join(run_dir, "best.pth"))

    if cfg.save_best and best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    summary = {"test_loss": float(test_loss), "test_acc": float(test_acc), "best_val_acc": float(best_val_acc)}
    print(f"[Baseline][{cfg.run_name}][Test] loss={test_loss:.4f} acc={test_acc:.4f}")

    if run_dir:
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    return model, summary, history


