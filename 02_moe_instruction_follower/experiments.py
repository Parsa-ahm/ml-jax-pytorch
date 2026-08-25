"""Study runs: config sweep, sparse-vs-naive benchmark, surface plots, demo."""

import json
import os
import time
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import questionary
from analysis import append_record, measure
from core import Config, Tokenizer, decode_answer
from flax import nnx
from matplotlib.lines import Line2D  # noqa: E402
from moe import MoELayer
from ops import OP_BY_NAME, OP_NAMES
from rich.console import Console
from rich.live import Live
from rich.table import Table
from sparse import sparse_moe
from train import build_prompt, generate, load_model, save_model, train

PATH = "datasheet.jsonl"
CKPT_DIR = "checkpoints"
STEP_COUNTS = [1500, 2500, 3500, 4500]
SWEPT = [
    "d_model",
    "n_ops",
    "n_dig",
    "max_input_len",
    "n_experts",
    "top_k",
    "balance_coef",
    "seed",
]


def arch(r):
    if r["n_experts"] is None:
        return "dense"
    return f"moe{r['n_experts']}" + ("-bal" if r["balance_coef"] > 0 else "")


def mean_acc(rows, a, xkey, xv, ykey, yv):
    vs = [
        r["overall_acc"]
        for r in rows
        if arch(r) == a and r[xkey] == xv and r[ykey] == yv
    ]
    return 100 * sum(vs) / len(vs) if vs else np.nan


def surface_fig(rows, ykey, yvals, title, path, archs=("dense", "moe4")):
    xvals = [16, 32, 64, 128]  # d_model
    X, Y = np.meshgrid(xvals, yvals)
    colors = {
        "dense": "steelblue",
        "moe4": "coral",
        "moe4-bal": "seagreen",
        "moe3": "purple",
    }

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for a in archs:
        Z = np.array(
            [
                [mean_acc(rows, a, "d_model", xv, ykey, yv) for xv in xvals]
                for yv in yvals
            ]
        )
        ax.plot_surface(
            X, Y, Z, color=colors[a], alpha=0.6, edgecolor="k", linewidth=0.3
        )

    ax.set_xlabel("d_model")
    ax.set_ylabel(ykey)
    ax.set_zlabel("accuracy (%)")
    ax.set_title(title)
    ax.view_init(elev=22, azim=-125)
    legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=12,
            markerfacecolor=colors[a],
            alpha=0.7,
            label=a,
        )
        for a in archs
    ]
    ax.legend(handles=legend, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")


def plot_surfaces():
    with open(PATH) as f:
        rows = [json.loads(line) for line in f]
    print(f"{len(rows)} rows")
    surface_fig(
        rows, "n_dig", [10, 20], "accuracy vs d_model x vocab", "figures/surf_vocab.svg"
    )
    surface_fig(
        rows, "n_ops", [4, 8, 16], "accuracy vs d_model x n_ops", "figures/surf_ops.svg"
    )


def already_done(config: Config) -> bool:
    if not os.path.exists(PATH):
        return False
    key = {k: getattr(config, k) for k in SWEPT}
    with open(PATH) as f:
        for line in f:
            r = json.loads(line)
            if all(r.get(k) == key[k] for k in SWEPT):
                return True
    return False


def run_config(config: Config) -> None:
    t0 = time.perf_counter()
    model, tok = train(config)
    dt = time.perf_counter() - t0
    append_record(measure(config, model, tok, dt), PATH)


def run_sweep(
    seeds=(1, 2, 3),
    steps=8000,
    d_models=(16, 32, 64, 128),
    n_ops=(4, 8, 16),
    n_dig=(10, 20),
    archs=None,
):
    if archs is None:
        archs = [
            dict(n_experts=None),
            dict(n_experts=4, top_k=2, balance_coef=0.0),
            dict(n_experts=4, top_k=2, balance_coef=0.02),
            dict(n_experts=3, top_k=2, balance_coef=0.0),
        ]

    base = Config(steps=steps)
    done = 0
    for seed in seeds:
        for d_model in d_models:
            for n_op in n_ops:
                for n_d in n_dig:
                    for arch in archs:
                        cfg = replace(
                            base,
                            seed=seed,
                            d_model=d_model,
                            n_ops=n_op,
                            n_dig=n_d,
                            **arch,
                        )
                        if already_done(cfg):
                            continue
                        print(
                            f"[run {done}] seed={seed} d={d_model} n_ops={n_op} "
                            f"n_dig={n_d} experts={arch.get('n_experts')} "
                            f"bal={arch.get('balance_coef', 0.0)}"
                        )
                        run_config(cfg)
                        done += 1
    print(f"done: {done} new runs")


def benchmark_crossover(d_models=(64, 128, 256, 512, 1024)):

    E = 4
    k = 2

    moe = MoELayer(64, E, k, rngs=nnx.Rngs(0))
    x = jnp.array(np.random.default_rng(0).normal(size=(8, 20, 64)), dtype=jnp.float32)
    naive, _ = moe(x)
    sp = sparse_moe(moe, x, k)
    print("max |naive - sparse|:", float(jnp.abs(naive - sp).max()), "(want < 1e-4)")

    def bench(fn, *a, iters=100):
        fn(*a).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*a).block_until_ready()
        return (time.perf_counter() - t0) / iters * 1e3

    print(f"{'d_model':>8} {'naive':>9} {'sparse':>9} {'speedup':>8}")

    for dm in d_models:
        m = MoELayer(dm, E, k, rngs=nnx.Rngs(0))
        xx = jnp.array(
            np.random.default_rng(0).normal(size=(64, 64, dm)), dtype=jnp.float32
        )
        nj = nnx.jit(lambda mm, z: mm(z)[0])
        sj = nnx.jit(lambda mm, z: sparse_moe(mm, z, k))
        mn, ms = bench(nj, m, xx), bench(sj, m, xx)
        print(f"{dm:>8} {mn:>9.3f} {ms:>9.3f} {mn / ms:>7.2f}x")


def demo():
    tok = Tokenizer()
    models = {}
    config = {
        "vocab_size": tok.vocab_size,
        "seq_len": tok.seq_len,
        "d_model": 64,
        "n_layers": 2,
        "n_head": 1,
    }
    os.makedirs(CKPT_DIR, exist_ok=True)
    for s in STEP_COUNTS:
        path = f"{CKPT_DIR}/dense_{s}.pkl"
        if os.path.exists(path):
            model = load_model(path)
        else:
            model, _ = train(Config(steps=s))
            save_model(model, config, path)
        models[f"{s} steps"] = model

    console = Console()
    warm_prompt = build_prompt(tok, "SORT", [1, 2, 3])
    for model in models.values():
        generate(model, tok, warm_prompt)
    while True:
        active_ops = OP_NAMES[: tok.config.n_ops]  # only ops the model was trained on
        op = questionary.select("Operations: ", choices=[*active_ops, "QUIT"]).ask()
        if op == "QUIT":
            break
        nums = [
            int(x)
            for x in questionary.text("Numbers (space separated upto 8): ")
            .ask()
            .split()
        ]

        prompt = build_prompt(tok, op, nums)
        truth = OP_BY_NAME[op].solve(nums)
        console.print(f"[bold]{op} {nums} [/bold] -> truth: [cyan]{truth}[/cyan]")

        table = Table()
        table.add_column("Model")
        table.add_column("Output")
        table.add_column("✓")
        table.add_column("Time (ms)", justify="right")

        with Live(table, console=console, refresh_per_second=10):
            for label, model in models.items():
                t0 = time.perf_counter()
                out = generate(model, tok, prompt)
                dt = (time.perf_counter() - t0) * 1000
                pred = decode_answer(jnp.array(out), tok)
                ok = pred == truth
                table.add_row(
                    label,
                    str(pred),
                    "✓" if ok else "✗",
                    f"{dt:.1f}",
                    style="green" if ok else "red",
                )
