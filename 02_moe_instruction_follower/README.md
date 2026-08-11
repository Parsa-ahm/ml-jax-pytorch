# 02 — Algorithmic Instruction-Follower (MoE)

One tiny decoder transformer learns **eight** short list operations at once, picked
by a prefix token:

```
SORT    4 1 3      = 1 3 4
DEDUP   4 0 0 1 0  = 4 0 1
RUNMAX  6 9 5 6    = 6 9 9 9
PARITY  9 3 6 9    = 1
```

Data is generated on the fly (infinite, free) and **every answer is checked by real
code**, so accuracy is exact — per operation, no labels, no test set to curate.

## Why this task

This is the open-ended rung. Instead of one optimization, it tells three connected
stories at once, because the task was chosen to make them line up:

1. **Dense vs Mixture-of-Experts.** Each op is a natural "mode." A dense model crams
   all eight into one FFN; an MoE can route different ops to different experts. So the
   dense-vs-MoE comparison is the point, not a formality.
2. **Scaling curves.** Train several sizes, plot accuracy vs compute, dense vs MoE.
3. **A custom kernel.** The MoE experts reduce to a `ragged_dot` (grouped matmul); we
   write a fused Pallas/Triton kernel for it and benchmark it against the naive version.

The standout result: **8 ops but only 4 experts**, so experts must *share*. We measure
how the four experts partition the eight ops, whether that partition respects operation
*families* (compare / set / move / reduce), and how routing sharpens as the model scales.

## The eight ops

| Op | Family | Meaning |
|------|---------|---------|
| SORT | compare | ascending sort |
| RUNMAX | compare | running maximum |
| RUNMIN | compare | running minimum |
| DEDUP | set | drop duplicates, keep first-seen order |
| UNION | set | sorted unique values |
| ROTATE | move | rotate right by one |
| REVERSE | move | reverse the list |
| PARITY | reduce | parity (0/1) of the count of odd values |

Values are single digits 0–9; input length 3–8; sequence length 20. Vocab = 22 tokens.

## Stack (differs from rungs 1 & 3)

**JAX + Flax nnx + Pallas.** This is the JAX/kernel rung; rungs 1 and 3 stay PyTorch.

## Files

```
ops.py        the 8 operations as pure functions (the source of truth)   [done]
data.py       tokenizer, on-the-fly batch generator, answer grader        [done]
baseline.py   dense decoder transformer (~10M)                            [todo]
moe.py        same model, FFN swapped for top-2 router + 4 experts        [todo]
kernel.py     fused ragged_dot Pallas/Triton kernel for the experts       [todo]
scaling.py    train several sizes, dense vs MoE, save the curve           [todo]
analysis.py   router assignments vs op, family alignment, entropy vs scale [todo]
test_*.py     correctness first — a claim with no passing test is not a claim
```

## Results (filled in at the end)

| Model | Params | Overall acc | Weakest op | Notes |
|-------|--------|-------------|------------|-------|
| dense | — | — | — | — |
| MoE (4 experts) | — | — | — | — |

| Kernel | tokens/sec | vs naive | Notes |
|--------|-----------|----------|-------|
| naive grouped matmul | — | 1.0× | — |
| fused ragged_dot | — | — | — |

## Run

```bash
uv run python 02_moe_instruction_follower/data.py    # eyeball sample examples
uv run pytest 02_moe_instruction_follower            # tests
```
