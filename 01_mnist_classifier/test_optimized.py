"""Tests for the int8-quantized model. The quantized net must (a) still be a
valid classifier and (b) stay within accuracy tolerance of the baseline."""

import os

import pytest
import torch

WEIGHTS = os.path.join(os.path.dirname(__file__), "mnist_baseline.pt")
needs_weights = pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="train baseline.py first")


@pytest.fixture(autouse=True, scope="module")
def _chdir():
    os.chdir(os.path.dirname(__file__))


@needs_weights
def test_quantized_forward_shape():
    from optimized import load_baseline, quantize

    out = quantize(load_baseline())(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


@needs_weights
def test_linear_layers_are_quantized():
    """Dynamic quantization should replace nn.Linear with quantized modules."""
    from optimized import load_baseline, quantize

    qmodel = quantize(load_baseline())
    assert qmodel.fc1.__class__.__module__.startswith("torch.ao.nn.quantized")
    assert qmodel.fc2.__class__.__module__.startswith("torch.ao.nn.quantized")


@needs_weights
def test_accuracy_within_tolerance():
    """Quantized accuracy must be within 1% of the fp32 baseline."""
    from benchmark import get_test_loader, measure_accuracy
    from optimized import load_baseline, quantize

    loader = get_test_loader()
    base_acc = measure_accuracy(load_baseline(), loader)
    q_acc = measure_accuracy(quantize(load_baseline()), loader)
    assert abs(base_acc - q_acc) < 0.01
