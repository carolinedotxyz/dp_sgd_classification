import json
import os
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core import SimpleCNN
from src.config import BaselineConfig
from src import train_baseline as tb


def _make_tiny_loaders(num_samples: int = 16, num_classes: int = 2, image_size: int = 8, batch_size: int = 4) -> (DataLoader, DataLoader, DataLoader):
    c, h, w = 3, image_size, image_size
    def _mk_loader(seed_offset: int) -> DataLoader:
        g = torch.Generator().manual_seed(1234 + seed_offset)
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
            # snapshot model at validation time for each epoch
            self.snapshots.append(_clone_state_dict(model.state_dict()))
            acc = float(self.val_accs[self.val_call_index])
            self.val_call_index += 1
            loss = float(1.0 - acc)
            return loss, acc
        if id(loader) == self.test_loader_id:
            acc = float(self.test_acc)
            loss = float(1.0 - acc)
            return loss, acc
        # default
        return 1.0, 0.0


def test_get_optimizer_variants():
    params = [nn.Parameter(torch.zeros(1))]
    opt = tb._get_optimizer("adam", params, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.Adam)

    opt = tb._get_optimizer("adamw", params, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.AdamW)

    opt = tb._get_optimizer("sgd", params, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.SGD)

    # default fallback
    opt = tb._get_optimizer("unknown", params, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.Adam)


def test_train_baseline_saves_best_and_outputs(tmp_path):
    train_loader, val_loader, test_loader = _make_tiny_loaders()
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=2).to(device)

    cfg = BaselineConfig(
        epochs=3,
        lr=1e-2,
        weight_decay=0.0,
        optimizer="adam",
        run_name="unit_baseline_best",
        output_dir=str(tmp_path),
        seed=123,
        save_best=True,
    )

    fake_eval = FakeEvaluate(val_loader, test_loader, val_accs=[0.50, 0.70, 0.60], test_acc=0.65)

    # monkeypatch the evaluate function that train_baseline imports
    original_evaluate = tb.evaluate
    try:
        tb.evaluate = fake_eval  # type: ignore
        model, summary, history = tb.train_baseline(model, train_loader, val_loader, test_loader, device, cfg)
    finally:
        tb.evaluate = original_evaluate  # restore

    # history length equals epochs
    assert len(history) == cfg.epochs
    # summary values reflect fake evaluator
    assert summary["best_val_acc"] == 0.70
    assert summary["test_acc"] == 0.65

    # model state should equal the on-disk best checkpoint
    current_state = _clone_state_dict(model.state_dict())
    run_dir = os.path.join(str(tmp_path), cfg.run_name)
    best_path = os.path.join(run_dir, "best.pth")
    assert os.path.isfile(best_path)
    best_on_disk = torch.load(best_path, map_location="cpu")
    assert _state_dicts_allclose(current_state, best_on_disk)

    # files are written
    assert os.path.isdir(run_dir)
    assert os.path.isfile(os.path.join(run_dir, "config.json"))
    assert os.path.isfile(os.path.join(run_dir, "history.csv"))
    assert os.path.isfile(os.path.join(run_dir, "summary.json"))
    assert os.path.isfile(os.path.join(run_dir, "best.pth"))

    # config.json includes some expected keys
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        config_json = json.load(f)
    assert config_json["run_name"] == cfg.run_name
    assert config_json["optimizer"].lower() == "adam"


def test_train_baseline_no_save_best_keeps_last_state_and_no_best_file(tmp_path):
    train_loader, val_loader, test_loader = _make_tiny_loaders()
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=2).to(device)

    cfg = BaselineConfig(
        epochs=3,
        lr=1e-2,
        weight_decay=0.0,
        optimizer="adam",
        run_name="unit_baseline_nosave",
        output_dir=str(tmp_path),
        seed=321,
        save_best=False,
    )

    fake_eval = FakeEvaluate(val_loader, test_loader, val_accs=[0.50, 0.70, 0.60], test_acc=0.55)

    original_evaluate = tb.evaluate
    try:
        tb.evaluate = fake_eval  # type: ignore
        model, summary, history = tb.train_baseline(model, train_loader, val_loader, test_loader, device, cfg)
    finally:
        tb.evaluate = original_evaluate

    # still tracks best val acc
    assert summary["best_val_acc"] == 0.70
    # history length equals epochs
    assert len(history) == cfg.epochs

    # model should reflect the final epoch state (snapshot at index 2)
    last_snapshot = fake_eval.snapshots[2]
    current_state = _clone_state_dict(model.state_dict())
    assert _state_dicts_allclose(current_state, last_snapshot)

    # best.pth should not exist when save_best is False
    run_dir = os.path.join(str(tmp_path), cfg.run_name)
    assert os.path.isdir(run_dir)
    assert not os.path.isfile(os.path.join(run_dir, "best.pth"))
    # but other files should still exist
    assert os.path.isfile(os.path.join(run_dir, "config.json"))
    assert os.path.isfile(os.path.join(run_dir, "history.csv"))
    assert os.path.isfile(os.path.join(run_dir, "summary.json"))


