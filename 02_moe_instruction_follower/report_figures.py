"""Regenerate every figure used in WRITEUP.md from datasheet.jsonl.

Pure matplotlib over the recorded sweep (no GPU, no training). The one
exception is the sparse-vs-naive crossover, whose wall-clock numbers are the
measured output of ``app.py bench`` on the dev GPU and are pinned here so the
figure is reproducible without re-running the benchmark.

    uv run python report_figures.py
"""

import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ops import OP_NAMES  # noqa: E402

PATH = "datasheet.jsonl"
OUT = "figures"
ARCHS = ["dense", "moe4", "moe4-bal", "moe3"]
COLORS = {
    "dense": "#4C72B0",
    "moe4": "#DD8452",
    "moe4-bal": "#55A868",
    "moe3": "#8172B3",
}
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load():
    return [json.loads(line) for line in open(PATH)]


def arch(r):
    if r["n_experts"] is None:
        return "dense"
    return f"moe{r['n_experts']}" + ("-bal" if r["balance_coef"] > 0 else "")


def marginal(rows, xkey, xvals, **fixed):
    """mean overall_acc (%) per arch at each x, averaging over everything else."""
    out = {a: [] for a in ARCHS}
    for a in ARCHS:
        for xv in xvals:
            vs = [
                r["overall_acc"]
                for r in rows
                if arch(r) == a
                and r[xkey] == xv
                and all(r[k] == v for k, v in fixed.items())
            ]
            out[a].append(100 * st.mean(vs) if vs else np.nan)
    return out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/{name}")


# --- 1. capacity (d_model) ------------------------------------------------
def fig_dmodel(rows):
    xs = [16, 32, 64, 128]
    data = marginal(rows, "d_model", xs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ARCHS:
        ax.plot(xs, data[a], "o-", color=COLORS[a], label=a, linewidth=2, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.set_xlabel("d_model (capacity)")
    ax.set_ylabel("overall accuracy (%)")
    ax.set_title("Accuracy vs capacity — saturates near d=64")
    ax.legend()
    save(fig, "fig_dmodel.svg")


# --- 2. task diversity (n_ops) & vocab (n_dig) ----------------------------
def fig_nops(rows):
    xs = [4, 8, 16]
    data = marginal(rows, "n_ops", xs, d_model=64)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ARCHS:
        ax.plot(xs, data[a], "o-", color=COLORS[a], label=a, linewidth=2, markersize=6)
    ax.set_xticks(xs)
    ax.set_xlabel("n_ops (number of operations to learn)")
    ax.set_ylabel("overall accuracy (%)")
    ax.set_title("Accuracy vs task diversity (d=64)")
    ax.legend()
    save(fig, "fig_nops.svg")


def fig_ndig(rows):
    xs = [10, 20]
    data = marginal(rows, "n_dig", xs, d_model=64)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = 0.2
    base = np.arange(len(xs))
    for i, a in enumerate(ARCHS):
        ax.bar(base + (i - 1.5) * w, data[a], w, color=COLORS[a], label=a)
    ax.set_xticks(base)
    ax.set_xticklabels([f"n_dig={x}" for x in xs])
    ax.set_ylabel("overall accuracy (%)")
    ax.set_ylim(80, 95)
    ax.set_title("Accuracy vs vocabulary size (d=64)")
    ax.legend()
    save(fig, "fig_ndig.svg")


# --- 3. accuracy vs parameters (the "same curve" finding) -----------------
def fig_params(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in ARCHS:
        pts = []
        for dm in [16, 32, 64, 128]:
            rs = [r for r in rows if arch(r) == a and r["d_model"] == dm]
            if rs:
                p = st.mean(r["params"] for r in rs)
                acc = 100 * st.mean(r["overall_acc"] for r in rs)
                pts.append((p, acc))
        pts.sort()
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, "o-", color=COLORS[a], label=a, linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("total parameters")
    ax.set_ylabel("overall accuracy (%)")
    ax.set_title("Accuracy vs total params — all archs share one curve")
    ax.legend()
    save(fig, "fig_params.svg")


# --- 4. seed variance (the noise floor) -----------------------------------
def fig_seed(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, a in enumerate(ARCHS):
        vs = [
            100 * r["overall_acc"]
            for r in rows
            if arch(r) == a
            and r["d_model"] == 64
            and r["n_ops"] == 8
            and r["n_dig"] == 10
        ]
        ax.errorbar(
            i,
            st.mean(vs),
            yerr=st.pstdev(vs) if len(vs) > 1 else 0,
            fmt="o",
            color=COLORS[a],
            capsize=6,
            markersize=8,
            linewidth=2,
        )
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("overall accuracy (%)")
    ax.set_title("Seed variance (d=64, n_ops=8, n_dig=10) — the noise floor")
    save(fig, "fig_seed.svg")


# --- 5. routing: collapse vs balanced -------------------------------------
def _grid(rows, a, dm, nops):
    rs = [
        r
        for r in rows
        if arch(r) == a
        and "routing_grid" in r
        and r["d_model"] == dm
        and r["n_ops"] == nops
    ]
    if not rs:
        return None
    g = sum(np.array(r["routing_grid"]) for r in rs)
    return g[: nops]  # only trained ops


def fig_routing(rows):
    nops = 16
    unbal = _grid(rows, "moe4", 128, nops)
    bal = _grid(rows, "moe4-bal", 128, nops)
    if unbal is None or bal is None:
        print("routing grids missing, skipping fig_routing")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, g, title in [
        (axes[0], unbal, "Unbalanced → collapse"),
        (axes[1], bal, "Balanced → uniform"),
    ]:
        frac = g / (g.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(frac.shape[1]))
        ax.set_xticklabels([f"E{j}" for j in range(frac.shape[1])])
        ax.set_yticks(range(nops))
        ax.set_yticklabels(OP_NAMES[:nops], fontsize=8)
        ax.set_title(title)
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="fraction of tokens", shrink=0.8)
    fig.suptitle("Routing: op → expert (d=128, n_ops=16) — no specialization", y=1.02)
    fig.savefig(f"{OUT}/fig_routing.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/fig_routing.svg")


# --- 6. sparse-vs-naive crossover (measured on dev GPU) -------------------
def fig_crossover():
    dm = [64, 128, 256, 512, 1024]
    speedup = [0.65, 1.95, 2.33, 9.76, 5.72]  # app.py bench, dev GPU
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dm, speedup, "o-", color="#C44E52", linewidth=2, markersize=7)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="break-even")
    ax.fill_between(
        dm, 1.0, speedup, where=[s >= 1 for s in speedup], alpha=0.15, color="#C44E52"
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(dm)
    ax.set_xticklabels(dm)
    ax.set_xlabel("d_model")
    ax.set_ylabel("sparse speedup over naive (×)")
    ax.set_title("Sparse routing: liability small, win at scale")
    ax.legend()
    save(fig, "fig_crossover.svg")


# --- 7. 3D surfaces: accuracy vs d_model × n_ops --------------------------
def fig_surface(rows):
    xs = [16, 32, 64, 128]
    ys = [4, 8, 16]
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for a in ["dense", "moe4"]:
        Z = np.array(
            [
                [
                    100
                    * st.mean(
                        [
                            r["overall_acc"]
                            for r in rows
                            if arch(r) == a and r["d_model"] == xv and r["n_ops"] == yv
                        ]
                        or [np.nan]
                    )
                    for xv in xs
                ]
                for yv in ys
            ]
        )
        ax.plot_surface(
            X, Y, Z, color=COLORS[a], alpha=0.6, edgecolor="k", linewidth=0.3
        )
    ax.set_xlabel("d_model")
    ax.set_ylabel("n_ops")
    ax.set_zlabel("accuracy (%)")
    ax.set_title("Accuracy surface: dense (blue) vs moe4 (orange)")
    ax.view_init(elev=22, azim=-125)
    save(fig, "fig_surface.svg")


def main():
    rows = load()
    print(f"{len(rows)} rows")
    fig_dmodel(rows)
    fig_nops(rows)
    fig_ndig(rows)
    fig_params(rows)
    fig_seed(rows)
    fig_routing(rows)
    fig_crossover()
    fig_surface(rows)


if __name__ == "__main__":
    main()
