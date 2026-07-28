"""Tests for the fp32 baseline. Correctness first — a benchmark on a broken
model is a lie."""

import os

import pytest
import torch
import torch.nn.functional as F

from baseline import Net

WEIGHTS = os.path.join(os.path.dirname(__file__), "mnist_baseline.pt")


@pytest.fixture(autouse=True, scope="module")
def _chdir():
    """Run from this file's directory so relative weight/data paths resolve."""
    os.chdir(os.path.dirname(__file__))


def test_forward_shape():
    out = Net()(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)          # (batch, one score per digit)


def test_forward_finite():
    out = Net()(torch.randn(4, 1, 28, 28))
    assert torch.isfinite(out).all()     # no NaN / inf


def test_backward_runs():
    model = Net()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())   # every weight got a gradient


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="train baseline.py first")
def test_trained_accuracy_floor():
    from benchmark import get_test_loader, measure_accuracy
    from optimized import load_baseline

    acc = measure_accuracy(load_baseline(), get_test_loader())
    assert acc > 0.97                    # trained model must clear the floor
