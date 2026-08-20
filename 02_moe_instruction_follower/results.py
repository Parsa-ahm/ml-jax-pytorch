import time

import jax
import numpy as np
from analysis import route_counts
from baseline import GPT
from checkpoint import save_model
from data import SEQ_LEN, Tokenizer
from evaluate import evaluate, plot_accuracy, plot_routing
from flax import nnx
from moe import MoEGPT
from ops import OP_NAMES
from train import train

SEEDS = [1, 2, 3, 4, 5]
STEPS = 8000


def run(seeds=SEEDS, steps=STEPS):
    tok = Tokenizer()
    dense_runs = []
    moe_runs = []
    train_times = {"dense": [], "moe": []}
    routing_grids = []
    cfg_d = {
        "vocab_size": tok.vocab_size,
        "seq_len": SEQ_LEN,
        "d_model": 64,
        "n_layers": 2,
    }
    cfg_m = {**cfg_d, "n_experts": 4, "top_k": 2}
    for seed in seeds:
        dense = GPT(tok.vocab_size, SEQ_LEN, 64, 2, rngs=nnx.Rngs(seed))
        moe = MoEGPT(tok.vocab_size, SEQ_LEN, 64, 2, 4, 2, rngs=nnx.Rngs(seed))

        t0 = time.perf_counter()
        dense, _ = train(steps=steps, seed=seed, model=dense)
        train_times["dense"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        moe, _ = train(steps=steps, seed=seed, model=moe)
        train_times["moe"].append(time.perf_counter() - t0)

        dense_runs.append(evaluate(dense, tok))
        moe_runs.append(evaluate(moe, tok))
        routing_grids.append(route_counts(moe, tok))

        save_model(dense, cfg_d, f"checkpoints/dense_seeds{seed}.pkl")
        save_model(moe, cfg_m, f"checkpoints/moe_seeds{seed}.pkl")

    return tok, dense_runs, moe_runs, train_times, routing_grids


def aggregate(runs):
    out = {}
    for op in OP_NAMES:
        vals = [r[op] for r in runs]
        out[op] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def param_count(model):
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


if __name__ == "__main__":  # AI generated
    tok, dense_runs, moe_runs, train_times, routing_grids = run()
    d, m = aggregate(dense_runs), aggregate(moe_runs)

    # --- accuracy table (mean ± std) ---
    print(f"\n{'op':8} {'dense':>12} {'MoE':>12}")
    for op in OP_NAMES:
        print(
            f"{op:8}  {d[op][0] * 100:5.1f}±{d[op][1] * 100:4.1f}  "
            f"{m[op][0] * 100:5.1f}±{m[op][1] * 100:4.1f}"
        )

    # --- cost table (params + training time) ---
    p_dense = param_count(GPT(tok.vocab_size, SEQ_LEN, 64, 2, rngs=nnx.Rngs(0)))
    p_moe = param_count(MoEGPT(tok.vocab_size, SEQ_LEN, 64, 2, 4, 2, rngs=nnx.Rngs(0)))
    print(f"\n{'':6} {'params':>10} {'train (s)':>14}")
    print(
        f"{'dense':6} {p_dense:>10,} "
        f"{np.mean(train_times['dense']):>8.1f}±{np.std(train_times['dense']):.1f}"
    )
    print(
        f"{'moe':6} {p_moe:>10,} "
        f"{np.mean(train_times['moe']):>8.1f}±{np.std(train_times['moe']):.1f}"
    )

    # --- figures ---
    plot_accuracy({op: d[op][0] for op in OP_NAMES}, {op: m[op][0] for op in OP_NAMES})
    for i, grid in enumerate(routing_grids):
        plot_routing(grid, f"figures/routing_seed{i}.svg")
    plot_routing(np.sum(routing_grids, axis=0), "figures/routing_avg.svg")
