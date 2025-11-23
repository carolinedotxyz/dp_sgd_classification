import sys
import importlib
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core import SimpleCNN
from src.config import DPConfig, DPConfigV2


def _make_tiny_loaders(num_samples: int = 16, num_classes: int = 2, image_size: int = 8, batch_size: int = 4) -> (DataLoader, DataLoader, DataLoader):
    c, h, w = 3, image_size, image_size
    def _mk_loader(seed_offset: int) -> DataLoader:
        g = torch.Generator().manual_seed(4321 + seed_offset)
        images = torch.randn(num_samples, c, h, w, generator=g)
        labels = torch.randint(low=0, high=num_classes, size=(num_samples,), generator=g)
        ds = TensorDataset(images, labels)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    return _mk_loader(0), _mk_loader(1), _mk_loader(2)


def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _state_dicts_allclose(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], atol: float = 5e-2, rtol: float = 0.0) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        if not torch.allclose(a[k], b[k], atol=atol, rtol=rtol):
            return False
    return True


class FakeAccountant:
    def __init__(self, eps: float = 1.23):
        self._eps = float(eps)
    def get_epsilon(self, delta: float):  # signature-compatible
        return self._eps


class FakePrivacyEngine:
    def __init__(self, accountant: str = "rdp"):
        self.accountant = FakeAccountant()
    def make_private(self, module, optimizer, data_loader, noise_multiplier, max_grad_norm):
        # Return the inputs unchanged and a passthrough loader
        return module, optimizer, data_loader


class FakeEvaluate:
    def __init__(self, val_loader: DataLoader, test_loader: DataLoader, val_accs: List[float], test_acc: float):
        self.val_loader_id = id(val_loader)
        self.test_loader_id = id(test_loader)
        self.val_accs = val_accs
        self.test_acc = test_acc
        self.val_call_index = 0
        self.snapshots: List[Dict[str, torch.Tensor]] = []

    def __call__(self, model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
        if id(loader) == self.val_loader_id:
            self.snapshots.append(_clone_state_dict(model.state_dict()))
            acc = float(self.val_accs[self.val_call_index])
            self.val_call_index += 1
            loss = float(1.0 - acc)
            return loss, acc
        if id(loader) == self.test_loader_id:
            acc = float(self.test_acc)
            loss = float(1.0 - acc)
            return loss, acc
        return 1.0, 0.0


def _import_train_dp_with_fake_opacus():
    # Inject a dummy 'opacus' providing PrivacyEngine before importing
    class _DummyOpacus:
        PrivacyEngine = FakePrivacyEngine

    sys.modules.pop("opacus", None)
    sys.modules["opacus"] = _DummyOpacus()  # type: ignore
    return importlib.import_module("src.train_dp")


def test_train_dp_sgd_restores_best_and_reports_epsilon():
    tdp = _import_train_dp_with_fake_opacus()

    train_loader, val_loader, test_loader = _make_tiny_loaders()
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=2).to(device)

    cfg = DPConfig(
        max_grad_norm=1.0,
        noise_multiplier=0.5,
        target_delta=1e-5,
        epochs=3,
        lr=1e-2,
        weight_decay=0.0,
    )

    fake_eval = FakeEvaluate(val_loader, test_loader, val_accs=[0.50, 0.80, 0.60], test_acc=0.70)

    # monkeypatch evaluate symbol used inside train_dp module
    original_evaluate = tdp.evaluate
    try:
        tdp.evaluate = fake_eval  # type: ignore
        model, metrics = tdp.train_dp_sgd(model, train_loader, val_loader, test_loader, device, cfg)
    finally:
        tdp.evaluate = original_evaluate

    assert isinstance(metrics, dict)
    assert metrics["test_acc"] == 0.70
    assert metrics["epsilon"] == 1.23  # from FakeAccountant

    # Model should be restored to the best val snapshot (epoch index 1)
    best_snapshot = fake_eval.snapshots[1]
    current_state = _clone_state_dict(model.state_dict())
    assert _state_dicts_allclose(current_state, best_snapshot)


def test_train_dp_sgd_v2_scheduler_step_and_label_smoothing_and_best_acc():
    tdp = _import_train_dp_with_fake_opacus()

    train_loader, val_loader, test_loader = _make_tiny_loaders()
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=2).to(device)

    cfg = DPConfigV2(
        max_grad_norm=1.0,
        noise_multiplier=0.7,
        target_delta=1e-5,
        epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        label_smoothing=0.1,
        scheduler="step",
        step_size=1,
        gamma=0.5,
    )

    fake_eval = FakeEvaluate(val_loader, test_loader, val_accs=[0.55, 0.60], test_acc=0.58)

    original_evaluate = tdp.evaluate
    try:
        tdp.evaluate = fake_eval  # type: ignore
        model, metrics = tdp.train_dp_sgd_v2(model, train_loader, val_loader, test_loader, device, cfg)
    finally:
        tdp.evaluate = original_evaluate

    assert metrics["test_acc"] == 0.58
    assert metrics["epsilon"] == 1.23
    assert metrics["best_val_acc"] == 0.60

    # Model should be restored to the best val snapshot (epoch index 1)
    best_snapshot = fake_eval.snapshots[1]
    current_state = _clone_state_dict(model.state_dict())
    assert _state_dicts_allclose(current_state, best_snapshot)


