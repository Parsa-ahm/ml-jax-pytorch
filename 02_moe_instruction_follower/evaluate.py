import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from data import Tokenizer, decode_answer, make_batch
from flax import nnx
from generate import build_prompt, generate
from ops import OP_NAMES


def evaluate(model: nnx.Module, tok: Tokenizer, n_examples: int = 500):
    rng = np.random.default_rng(8)
    batch = make_batch(rng, tok, n_examples)

    correct = {name: 0 for name in OP_NAMES}
    total = {name: 0 for name in OP_NAMES}

    for ex in batch["examples"]:
        name = OP_NAMES[ex.op_id]
        prompt = build_prompt(tok, name, ex.xs)
        out = generate(model, tok, prompt)
        pred = decode_answer(jnp.array(out), tok)
        total[name] += 1
        if pred == ex.ys:
            correct[name] += 1

    return {name: correct[name] / total[name] for name in OP_NAMES}


def plot_accuracy(dense, moe, path="figures/accuracy.svg"):
    os.makedirs("figures", exist_ok=True)
    ops = list(dense.keys())
    x = np.arange(len(ops))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, [dense[o] * 100 for o in ops], w, label="dense")
    ax.bar(x + w / 2, [moe[o] * 100 for o in ops], w, label="MoE")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("per_op accuracy: dense vs MoE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")


def plot_routing(counts, path="figures/routing.svg"):
    os.makedirs("figures", exist_ok=True)
    frac = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(frac.shape[1]))
    ax.set_xticklabels([f"E{j}" for j in range(frac.shape[1])])
    ax.set_yticks(range(len(OP_NAMES)))
    ax.set_yticklabels(OP_NAMES)
    ax.set_title("Routing: op -> expert")
    fig.colorbar(im, label="fraction of tokens")
    fig.tight_layout()
    fig.savefig(path)
    print(f"saved {path}")
