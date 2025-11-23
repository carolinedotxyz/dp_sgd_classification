import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.core import get_device, set_seed, accuracy, evaluate


def test_get_device_reports_available_backend():
    dev = get_device()
    # We don't know user's hardware; just ensure torch recognizes it
    assert isinstance(dev, torch.device)
    assert dev.type in {"cpu", "cuda", "mps"}


def test_set_seed_makes_results_reproducible():
    set_seed(123)
    a1 = random.randint(0, 100000)
    b1 = np.random.rand(3)
    c1 = torch.randn(4)

    set_seed(123)
    a2 = random.randint(0, 100000)
    b2 = np.random.rand(3)
    c2 = torch.randn(4)

    assert a1 == a2
    assert np.allclose(b1, b2)
    assert torch.allclose(c1, c2)


def test_accuracy_matches_argmax_mean():
    logits = torch.tensor([[2.0, 1.0], [0.1, 0.2], [0.3, 0.2]])
    targets = torch.tensor([0, 1, 0])
    acc = accuracy(logits, targets)

    preds = logits.argmax(dim=1)
    expected = (preds == targets).float().mean().item()
    assert abs(acc - expected) < 1e-8


def test_evaluate_iterates_loader_and_averages():
    # simple linear model to keep deterministic
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)
        def forward(self, x):
            return self.fc(x)

    device = torch.device("cpu")
    set_seed(42)
    model = TinyModel().to(device)
    criterion = nn.CrossEntropyLoss()

    # create two batches of data
    set_seed(7)
    x = torch.randn(8, 3)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    loss, acc = evaluate(model, loader, criterion, device)

    # recompute reference metrics manually
    with torch.no_grad():
        total_loss, total_acc, total_n = 0.0, 0.0, 0
        for xb, yb in loader:
            logits = model(xb)
            l = criterion(logits, yb)
            bsz = xb.size(0)
            total_loss += l.item() * bsz
            total_acc += ((logits.argmax(dim=1) == yb).float().mean().item()) * bsz
            total_n += bsz
        ref_loss = total_loss / max(1, total_n)
        ref_acc = total_acc / max(1, total_n)

    assert abs(loss - ref_loss) < 1e-6
    assert abs(acc - ref_acc) < 1e-6


