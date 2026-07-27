# whittle

Take a canonical model. Build the textbook version. Then whittle it down — smaller, faster, cheaper — and **prove the tradeoff with numbers**.

Three rungs, one per major model family. Each follows the same spine:

```
baseline  ->  optimized  ->  benchmark
```

The point isn't training a model. Anyone can call `.fit()`. The point is: measure it, make it leaner, and show the before/after table. That's systems ML.

## The ladder

| # | Model | Family | Optimization axis | Deliverable |
|---|-------|--------|-------------------|-------------|
| 01 | MNIST classifier | Discriminative (CNN) | int8 quantization + portable inference | size ↓, latency ↓, accuracy ~flat |
| 02 | Tiny LLM (speaks Python) | Autoregressive (Transformer) | KV-cache + quantization | tokens/sec ↑ |
| 03 | Diffusion model | Generative | fewer sampling steps (DDIM / distillation) | steps ↓, quality held |

Rung 04 is intentionally unplanned — a paper reproduction on alternate hardware (TPU/JAX) or a custom GPU kernel. Picked later, once the first three are done.

## Method — every rung, no exceptions

```
README.md      the spec + the results table (filled in at the end)
baseline.py    the standard PyTorch implementation
optimized.py   ONE efficiency technique, pushed hard
benchmark.py   runs both, measures size / latency / accuracy, prints the table
test_*.py      correctness first — a claim with no passing test is not a claim
```

Rule: **one optimization axis per rung.** Stack too many and you learn none.

## Stack

PyTorch · uv · pytest · ruff. Runs on a single GPU or plain CPU. No cloud bill.

## Run

```bash
uv sync                              # one-time: install torch etc.
uv run pytest 01_mnist_classifier    # tests for one rung
uv run python 01_mnist_classifier/benchmark.py
```
