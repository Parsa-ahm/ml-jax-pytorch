"""
Benchmark — baseline fp32 vs int8 quantized. The proof.
"""

import os
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from optimized import load_baseline, quantize

BASELINE_PATH = "mnist_baseline.pt"
INT8_PATH = "mnist_int8.pt"


def get_test_loader(batch_size=64):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return DataLoader(test, batch_size=batch_size, shuffle=False)


def measure_size_mb(path):
    """On-disk size of a saved state_dict, in megabytes."""
    return os.path.getsize(path) / 1e6


def measure_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def measure_latency_ms(model, sample, n_warmup=50, n_iter=500):
    """Average ms to classify one image. Warms up first so lazy init / caching
    don't poison the timed run."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):          # untimed — burn off the slow first passes
            model(sample)
        start = time.perf_counter()
        for _ in range(n_iter):
            model(sample)
        elapsed = time.perf_counter() - start
    return elapsed / n_iter * 1000


def latency_sweep(models, batch, batch_sizes, n_warmup=10, n_iter=50):
    """Per-image latency (ms) for each model across a range of batch sizes.

    Batch-1 is int8's worst case (all quant/dequant overhead, no amortization).
    Bigger batches amortize that overhead over more matmul work, so this sweep
    shows *where* — if anywhere — int8 overtakes fp32."""
    sweep = {name: [] for name in models}
    for b in batch_sizes:
        x = batch[:b]
        for name, model in models.items():
            per_call = measure_latency_ms(model, x, n_warmup, n_iter)
            sweep[name].append(per_call / b)   # normalize to per-image
    return sweep


def print_sweep(batch_sizes, sweep):
    fp32 = sweep["baseline (fp32)"]
    int8 = sweep["optimized (int8)"]
    print(f"\n| {'batch':>5} | {'fp32 ms/img':>11} | {'int8 ms/img':>11} | {'int8 speedup':>12} |")
    print(f"| {'-' * 5} | {'-' * 11} | {'-' * 11} | {'-' * 12} |")
    for b, f, q in zip(batch_sizes, fp32, int8):
        print(f"| {b:>5} | {f:>11.4f} | {q:>11.4f} | {f / q:>11.2f}x |")


def plot_sweep(batch_sizes, sweep):
    colors = {"baseline (fp32)": "#9ca3af", "optimized (int8)": "#7C3AED"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ys in sweep.items():
        ax.plot(batch_sizes, ys, marker="o", label=name, color=colors.get(name))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("latency (ms / image)")
    ax.set_title("Per-image latency vs batch size", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("benchmark_sweep.png", dpi=150)
    print("saved sweep chart -> benchmark_sweep.png")


def print_table(results):
    print(f"\n| {'model':<18} | {'size (MB)':>9} | {'latency (ms/img)':>16} | {'accuracy':>8} |")
    print(f"| {'-' * 18} | {'-' * 9} | {'-' * 16} | {'-' * 8} |")
    for name, r in results.items():
        print(f"| {name:<18} | {r['size']:>9.3f} | {r['latency']:>16.3f} | {r['acc']:>8.4f} |")


def plot(results):
    names = list(results)
    colors = ["#9ca3af", "#7C3AED"]   # gray baseline, violet optimized
    metrics = [("size", "size (MB)"), ("latency", "latency (ms/img)"), ("acc", "test accuracy")]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, title) in zip(axes, metrics):
        vals = [results[n][key] for n in names]
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(title, fontweight="bold")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.margins(y=0.15)
    fig.suptitle("MNIST — fp32 baseline vs int8 quantized", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("benchmark.png", dpi=150)
    print("saved chart -> benchmark.png")


def main():
    torch.set_num_threads(1)   # single-threaded = stable, comparable timing
    loader = get_test_loader()
    sample = next(iter(loader))[0][:1]   # one image, shape (1,1,28,28), CPU

    baseline = load_baseline()           # fp32
    qmodel = quantize(load_baseline())   # int8 (fresh copy so baseline stays fp32)

    results = {
        "baseline (fp32)": {
            "size": measure_size_mb(BASELINE_PATH),
            "latency": measure_latency_ms(baseline, sample),
            "acc": measure_accuracy(baseline, loader),
        },
        "optimized (int8)": {
            "size": measure_size_mb(INT8_PATH),
            "latency": measure_latency_ms(qmodel, sample),
            "acc": measure_accuracy(qmodel, loader),
        },
    }

    print_table(results)

    b, q = results["baseline (fp32)"], results["optimized (int8)"]
    print(f"\nsize:    {b['size'] / q['size']:.2f}x smaller")
    print(f"latency: {b['latency'] / q['latency']:.2f}x faster")
    print(f"accuracy drop: {(b['acc'] - q['acc']) * 100:+.2f} pts")

    plot(results)

    # --- batch-size sweep: find where (if ever) int8 overtakes fp32 ---
    batch = next(iter(get_test_loader(512)))[0]   # up to 512 images, CPU
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    models = {"baseline (fp32)": baseline, "optimized (int8)": qmodel}
    sweep = latency_sweep(models, batch, batch_sizes)

    print_sweep(batch_sizes, sweep)
    cross = next(
        (b for b, f, q in zip(batch_sizes, sweep["baseline (fp32)"], sweep["optimized (int8)"]) if q <= f),
        None,
    )
    print(
        f"\nint8 overtakes fp32 at batch >= {cross}"
        if cross else "\nint8 never overtakes fp32 in the tested range"
    )
    plot_sweep(batch_sizes, sweep)


if __name__ == "__main__":
    main()
