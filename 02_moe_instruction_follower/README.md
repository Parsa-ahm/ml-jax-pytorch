# 02 - Algorithmic Instruction-Follower (MoE)

A Mixture of Experts model built by JAX/FLAX (nnx), trained on a list of Operations that are deterministic, allowing for full synthetic training and testing sets.

This project was made to explore the characteristics of MoE models, and for reference I also have a dense GPT that has been trained on the exact same data for comparison.
For the results see [WRITEUP.md](./WRITEUP.md).

## Task

Given an operation token and a short list of digits emit the results:
"SORT 4 5 2 1 = 1 2 4 5".
Given the exploratory nature of this experiment I decided to maintain variables and avoid hard coding variables. This was done in a Config class inside core.py.
This allowed for a training run of a variety of models with different attributes within the same regime, for comparison.

## Model Attributes:

| Attributes    | GPT | MoE | Description                                                                        |
| ------------- | --- | --- | ---------------------------------------------------------------------------------- |
| n_ops         | ✓   | ✓   | Number of operations that the model have to learn out of the 21 pre made functions |
| n_dig         | ✓   | ✓   | Number of digits that are in the vocabulary                                        |
| min_input_len | ✓   | ✓   | Min length of list operated on                                                     |
| max_input_len | ✓   | ✓   | Max length of list operated on                                                     |
| d_model       | ✓   | ✓   | The size of the hidden layers of the models                                        |
| n_layers      | ✓   | ✓   | Number of the layers in the hidden layers                                          |
| n_head        | ✓   | ✓   | Number of attention heads per layer                                                |
| n_experts     | x   | ✓   | Number of experts that the MoE should have                                         |
| top_k         | x   | ✓   | How many of the experts should be active per run                                   |
| balance_coef  | x   | ✓   | The coefficient used to ballance the load between experts to avoid dead experts    |
| steps         | ✓   | ✓   | Number of steps used to train the models                                           |
| batch_size    | ✓   | ✓   | How many instances in each batch for each step during training                     |
| lr            | ✓   | ✓   | The adam learning rate                                                             |
| seed          | ✓   | ✓   | The seed used when randomizing for each batch                                      |

## Files

| File             | What it holds                                                              |
| ---------------- | -------------------------------------------------------------------------- |
| `ops.py`         | The 21 list operations and the `Op` registry (name, family, solve).        |
| `core.py`        | `Config`, `Tokenizer`, and data generation/grading.                        |
| `dense.py`       | The dense transformer: embeddings, causal attention, block, GPT.           |
| `moe.py`         | MoE model: router, experts, MoE block/GPT, load-balance loss.              |
| `train.py`       | `build_model`, the training loop, generation, checkpointing.               |
| `analysis.py`    | Per-model measurement: accuracy, routing, cost, records, plots.            |
| `sparse.py`      | Sparse top-k routing via `jax.lax.ragged_dot` - only selected experts run. |
| `experiments.py` | The study: config sweep, sparse-vs-naive benchmark, surfaces, demo.        |
| `app.py`         | Single CLI entry point (all commands below).                               |
| `test_*.py`      | 103 tests across ops, core, dense, moe, sparse, and the full pipeline.     |

## Usage

```bash
# train one config (dense if --experts omitted)
uv run python app.py train --d-model 64 --experts 4 --top-k 2 --steps 8000

# run the accuracy grid -> datasheet.jsonl
uv run python app.py sweep --seeds 1 2 3 --d-models 16 32 64 128 \
    --n-ops 4 8 16 --n-dig 10 20

# sparse vs naive MoE compute crossover
uv run python app.py bench --d-models 64 128 256 512 1024

# render accuracy surfaces from datasheet.jsonl -> figures/
uv run python app.py plot

# interactive: prompt models, compare outputs
uv run python app.py demo

# tests
uv run pytest

```

Notes:
Sweep appends to `datasheet.jsonl` (gitignored) and skips configs already recorded, so it's resumable. Plot and demo read what sweep and train produced.
