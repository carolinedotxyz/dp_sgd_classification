from typing import Dict, Any, Tuple
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from .config import DPConfig, DPConfigV2
from .utils import accuracy, evaluate


def train_dp_sgd(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                  device: torch.device, cfg: DPConfig) -> Tuple[nn.Module, Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=cfg.noise_multiplier,
        max_grad_norm=cfg.max_grad_norm,
    )
    accountant = privacy_engine.accountant

    best_val_acc = -float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, epoch_acc, epoch_n = 0.0, 0.0, 0
        start = time.time()
        for images, targets in train_dp_loader:
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
        eps = accountant.get_epsilon(delta=cfg.target_delta)
        dur = time.time() - start
        print(f"[DP][Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} eps={eps:.2f} time={dur:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    eps = accountant.get_epsilon(delta=cfg.target_delta)
    print(f"[DP][Test] loss={test_loss:.4f} acc={test_acc:.4f} eps={eps:.2f}")

    metrics = {"test_loss": float(test_loss), "test_acc": float(test_acc), "epsilon": float(eps)}
    return model, metrics


def train_dp_sgd_v2(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                     device: torch.device, cfg: DPConfigV2) -> Tuple[nn.Module, Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, train_dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=cfg.noise_multiplier,
        max_grad_norm=cfg.max_grad_norm,
    )
    accountant = privacy_engine.accountant

    scheduler = None
    if cfg.scheduler is not None:
        name = cfg.scheduler.lower()
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        elif name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    best_val_acc = -float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, epoch_acc, epoch_n = 0.0, 0.0, 0
        start = time.time()
        for images, targets in train_dp_loader:
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
        eps = accountant.get_epsilon(delta=cfg.target_delta)
        dur = time.time() - start
        print(f"[DPv2][Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} eps={eps:.2f} time={dur:.1f}s")
        if scheduler is not None:
            scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    eps = accountant.get_epsilon(delta=cfg.target_delta)
    print(f"[DPv2][Test] loss={test_loss:.4f} acc={test_acc:.4f} eps={eps:.2f}")

    metrics = {"test_loss": float(test_loss), "test_acc": float(test_acc), "epsilon": float(eps), "best_val_acc": float(best_val_acc)}
    return model, metrics


