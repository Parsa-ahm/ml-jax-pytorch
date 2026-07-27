# 01 — MNIST classifier, whittled

The vision rite of passage. Build a small CNN that reads handwritten digits, then shrink it and prove the shrink was nearly free.

## Goal

A digit classifier that is **~4x smaller and faster after quantization, with < 1% accuracy loss** — and a benchmark that proves it.

## The three files you write

### 1. `baseline.py` — the textbook CNN
A small convolutional network trained on MNIST to ~99% test accuracy.

**Your architecture decisions** (own these — you'll be asked about them):
- How many conv layers? What channel widths?
- Kernel size, stride, padding — and why.
- Pooling vs strided conv for downsampling.
- Where activations go, whether you use dropout/batchnorm.
- Optimizer + learning rate + epochs.

There's no single right answer. Pick, justify, and be ready to defend it. A 2-conv-block net hits 99% here — resist the urge to over-build.

### 2. `optimized.py` — the whittled version
Take the trained baseline and apply **post-training int8 quantization** (start with dynamic quantization — it's the simplest and needs no calibration data).

Concepts to understand *before* you write it (so you can speak on it cold):
- Why int8 is ~4x smaller than fp32 (do the arithmetic — it's not magic).
- What a quantization scale + zero-point are, and how a real value maps to an int8.
- Dynamic vs static vs quantization-aware training — what each trades.
- Why quantization barely dents accuracy here but *would* hurt a more sensitive model.

### 3. `benchmark.py` — the proof
Load both models, measure and print a table:

| model | size (MB) | latency (ms/img) | test accuracy |
|-------|-----------|------------------|---------------|
| baseline (fp32) | ? | ? | ? |
| optimized (int8) | ? | ? | ? |

Measure honestly: warm up before timing, average over many images, same test set for both. If the numbers don't move the way you predicted, that's a finding — investigate it, don't hide it.

## Tests (`test_baseline.py`, `test_optimized.py`)

You write these too. At minimum:
- Model outputs the right shape (batch, 10) and valid log-probs.
- A tiny forward/backward pass runs without error on fake data.
- Trained baseline clears an accuracy floor (e.g. > 97%).
- Quantized model stays within the accuracy tolerance of the baseline.

Correctness first. A benchmark on a broken model is a lie.

## Done when

- Baseline ≥ 99% test accuracy.
- Quantized model < 1% absolute accuracy drop.
- `benchmark.py` prints the filled table.
- The table gets pasted into this README under **## Results**.
- All tests green.

## Results

_(you fill this in — the whole point of the rung)_
