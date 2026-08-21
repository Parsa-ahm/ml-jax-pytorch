import time

import jax
import numpy as np
from analysis import evaluate, plot_accuracy, plot_routing, route_counts
from core import Config
from flax import nnx
from ops import OP_NAMES
from train import build_model, train

SEEDS = [1, 2, 3, 4, 5]
STEPS = 8000


def _ckpt_cfg(config: Config, tok) -> dict:
    d = {
        "vocab_size": tok.vocab_size,
        "seq_len": tok.seq_len,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_head": config.n_head,
    }
    if config.n_experts is not None:
        d.update(n_experts=config.n_experts, top_k=config.top_k)
    return d


def run(seeds=SEEDS, steps=STEPS):
    from train import save_model

    dense_runs, moe_runs, bal_runs = [], [], []
    train_times = {"dense": [], "moe": [], "balanced": []}
    routing_naive, routing_bal = [], []
    tok = None
    for seed in seeds:
        c_dense = Config(steps=steps, seed=seed)
        c_moe = Config(steps=steps, seed=seed, n_experts=4, top_k=2, balance_coef=0.0)
        c_bal = Config(steps=steps, seed=seed, n_experts=4, top_k=2, balance_coef=0.01)

        t0 = time.perf_counter()
        dense, tok = train(c_dense)
        train_times["dense"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        moe, _ = train(c_moe)
        train_times["moe"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        bal, _ = train(c_bal)
        train_times["balanced"].append(time.perf_counter() - t0)

        dense_runs.append(evaluate(dense, tok))
        moe_runs.append(evaluate(moe, tok))
        bal_runs.append(evaluate(bal, tok))
        routing_naive.append(route_counts(moe, tok))
        routing_bal.append(route_counts(bal, tok))

        save_model(dense, _ckpt_cfg(c_dense, tok), f"checkpoints/dense_seed{seed}.pkl")
        save_model(moe, _ckpt_cfg(c_moe, tok), f"checkpoints/moe_seed{seed}.pkl")
        save_model(bal, _ckpt_cfg(c_bal, tok), f"checkpoints/balmoe_seed{seed}.pkl")

    return tok, dense_runs, moe_runs, bal_runs, train_times, routing_naive, routing_bal


def aggregate(runs: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    out = {}
    for op in OP_NAMES:
        vals = [r[op] for r in runs]
        out[op] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def param_count(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


def utilization(grids: list[np.ndarray]) -> np.ndarray:
    # grids: list of (n_ops, n_experts) count matrices -> (n_seeds, n_experts) util
    return np.array([c.sum(axis=0) / c.sum() for c in grids])


if __name__ == "__main__":
    tok, dense_runs, moe_runs, bal_runs, train_times, r_naive, r_bal = run()
    d, m, b = aggregate(dense_runs), aggregate(moe_runs), aggregate(bal_runs)

    # --- accuracy table (3-way) ---
    print(f"\n{'op':8} {'dense':>11} {'MoE':>11} {'MoE+bal':>11}")
    for op in OP_NAMES:
        print(
            f"{op:8}  {d[op][0] * 100:4.1f}±{d[op][1] * 100:3.1f}  "
            f"{m[op][0] * 100:4.1f}±{m[op][1] * 100:3.1f}  "
            f"{b[op][0] * 100:4.1f}±{b[op][1] * 100:3.1f}"
        )

    # --- cost table ---
    p_dense = param_count(build_model(Config(), nnx.Rngs(0)))
    p_moe = param_count(build_model(Config(n_experts=4, top_k=2), nnx.Rngs(0)))
    print(f"\n{'':10} {'params':>10} {'train (s)':>14}")
    print(
        f"{'dense':10} {p_dense:>10,} "
        f"{np.mean(train_times['dense']):>8.1f}±{np.std(train_times['dense']):.1f}"
    )
    print(
        f"{'moe':10} {p_moe:>10,} "
        f"{np.mean(train_times['moe']):>8.1f}±{np.std(train_times['moe']):.1f}"
    )
    print(
        f"{'moe+bal':10} {p_moe:>10,} "
        f"{np.mean(train_times['balanced']):>8.1f}±{np.std(train_times['balanced']):.1f}"
    )

    # --- utilization: collapse before vs after balancing ---
    un, ub = utilization(r_naive), utilization(r_bal)
    print("\nmean expert utilization (ideal 0.25):")
    print(
        f"  naive    {np.array2string(un.mean(0), precision=3)}  "
        f"dead={int((un < 0.02).sum())}/{un.size}"
    )
    print(
        f"  balanced {np.array2string(ub.mean(0), precision=3)}  "
        f"dead={int((ub < 0.02).sum())}/{ub.size}"
    )

    # --- figures ---
    plot_accuracy({op: d[op][0] for op in OP_NAMES}, {op: b[op][0] for op in OP_NAMES})
    plot_routing(np.sum(r_naive, axis=0), "figures/routing_naive.svg")
    plot_routing(np.sum(r_bal, axis=0), "figures/routing_balanced.svg")
