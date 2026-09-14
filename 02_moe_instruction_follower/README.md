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

## Next Steps: Generalization

Everything measured so far is in-distribution. `make_batch` draws fresh examples
from an infinite generator every step, so there is no fixed training set and no
train/test gap: held-out loss tracks training loss by construction. The ~90%
ceiling in [WRITEUP.md](./WRITEUP.md) is therefore not overfitting, it is a
failure to learn the underlying algorithm. Regularization and "more data" cannot
move it.

The open question is out-of-distribution generalization. Three axes, none
currently tested, each with a structural blocker in the code.

### 1. Length

Train on `min_input_len=3, max_input_len=8`, test on 9-16.

Blockers:

- `dense.py` `Embeddings` uses learned absolute positions,
  `nnx.Embed(seq_len, d_model)`, and `Config.seq_len = 2 * max_input_len + 4`.
  Positions past the trained range do not exist, and positions seen rarely are
  undertrained. Length extrapolation is structurally impossible as written.
- `generate(..., max_new=9)` in `train.py` caps output length independently.
- `n_layers=2` is fixed depth. SORT, MODE and DEDUP on a length-16 list need
  more sequential steps than on a length-4 list; constant depth cannot supply
  them.

Planned work:

1. Make `seq_len` an independent `Config` field and derive `max_new` from it, so
   the model can be trained at length 8 but allocated for 16.
2. Ablate the positional scheme: learned-absolute (current) vs RoPE vs NoPE (no
   positional embedding, causal mask only). Single-variable change, fits the
   existing sweep harness.
3. Scratchpad targets: emit intermediate states rather than only the final
   answer, e.g. `SORT 4 5 2 1 = [4] [4 5] [2 4 5] [1 2 4 5] <eos>`. This turns a
   superlinear-depth problem into a constant-depth per-step one. The imperative
   loops in `ops.py` (`runmax`, `runmin`, `dedup`, insertion-form `sort`) make
   trace emission straightforward.
4. Hold out an interior length (train 3, 4, 6, 7, 8; test 5) to separate
   interpolation from extrapolation.

### 2. Vocabulary

Train on digits 0-9, test on 10-19.

Blocker: digits are atomic tokens with independent embeddings. Nothing encodes
that `7 < 8`; the ordering relation is learned only from comparison examples, so
an unseen digit carries a random embedding and every compare or sort op fails
outright. This is the likely mechanism behind the ~3 pt cost of `n_dig=20` in
WRITEUP section 4: more of the order relation to relearn from fewer examples per
token, not a harder task.

Planned work: structured number representation, either tying the embedding to a
scalar magnitude feature or tokenizing numbers positionally so ordering becomes
compositional. Held-out digits then become a meaningful test.

### 3. Instruction compositionality

The operation is a single atomic token (`Tokenizer` in `core.py`), so an unseen
op token has an untrained embedding and scores zero by construction. "Instruction
following" currently means 21-way classification into memorized behaviors.

Planned work: multi-token compositional instructions (`SORT DESC`,
`FILTER EVEN`, `REVERSE` after `SORT`), trained on a subset of the cross product
with combinations held out. This is also the setting where expert specialization
has a reason to emerge: `Op.family` (compare / set / move / reduce / filter)
already exists, and WRITEUP section 7 found no specialization, plausibly because
atomic op tokens give the experts no shared substructure to carve along.

### Order of work

1. OOD evaluation suite in `analysis.py`: accuracy broken out by input length, by
   held-out digit set, and by held-out operation. Baseline the existing
   checkpoints first. Near-zero scores on all three are the expected result and
   are worth reporting on their own.
2. `seq_len` / `max_new` decoupling.
3. Positional-scheme ablation on length extrapolation.
4. Scratchpad targets for the traceable ops.
5. Compositional instructions, which reopens the expert-specialization question.

Steps 1-3 are the small ones and they change what the study answers: from "MoE
vs dense at matched budget" to "does either actually learn the algorithm."
