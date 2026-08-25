# whittle

Take a canonical model, build the textbook version, then whittle it down (smaller,
faster, cheaper) and prove the tradeoff with numbers. The point isn't training a
model; it's measuring it, making it leaner, and showing the before/after table.
That's systems ML.

Two projects so far, each self-contained in its own folder with its own README.

## Projects

### 01 - MNIST classifier (PyTorch)

A textbook fp32 CNN quantized to int8, with an honest look at when quantization
actually helps.

- **What it shows:** int8 gives a reliable 3.5x size cut with ~0 accuracy drop, but
  the latency win is conditional. A batch-size sweep pins down where int8 overtakes
  fp32 (batch >= 4) and where it stops mattering (convs dominate at large batch).
- **Stack:** PyTorch, dynamic int8 quantization.
- **Details:** [`01_mnist_classifier/README.md`](01_mnist_classifier/README.md)

### 02 - MoE instruction-follower (JAX / Flax)

A from-scratch Mixture-of-Experts transformer against a dense baseline on a
synthetic list-operation task, plus a sparse routing kernel.

- **What it shows:** a controlled 346-run study across capacity, task diversity,
  vocabulary, and seeds. Matched by parameters, MoE is not a better model than
  dense and its experts do not specialize; its edge is a compute one, and a sparse
  `ragged_dot` kernel turns that into a real speedup only at scale (0.65x to 9.76x).
- **Stack:** JAX, Flax (nnx), a single-command CLI.
- **Details:** [`02_moe_instruction_follower/README.md`](02_moe_instruction_follower/README.md)
  and [`02_moe_instruction_follower/WRITEUP.md`](02_moe_instruction_follower/WRITEUP.md)

## Access

One install covers both projects:

```bash
uv sync
```

Then run each project from the repo root:

```bash
# 01 - MNIST classifier
uv run python 01_mnist_classifier/baseline.py     # train + save fp32
uv run python 01_mnist_classifier/optimized.py    # quantize + save int8
uv run python 01_mnist_classifier/benchmark.py    # compare + chart

# 02 - MoE instruction-follower (run from inside the folder)
cd 02_moe_instruction_follower
uv run python app.py train      # train one config
uv run python app.py sweep      # the accuracy grid
uv run python app.py bench      # sparse vs naive compute
uv run python app.py plot       # render figures

# tests, either project
uv run pytest 01_mnist_classifier
uv run pytest 02_moe_instruction_follower
```

## Stack

uv, pytest, ruff. PyTorch for 01, JAX/Flax for 02. Runs on a single GPU or plain CPU.
