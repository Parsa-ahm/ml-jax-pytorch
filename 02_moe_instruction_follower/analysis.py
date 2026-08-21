import json
import os
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from core import Config, Tokenizer, decode_answer, make_batch
from flax import nnx
from moe import MoEGPT
from ops import OP_NAMES
from train import build_prompt, generate


# --- accuracy -------------------------------------------------------------
def evaluate(
    model: nnx.Module, tok: Tokenizer, n_examples: int = 500
) -> dict[str, float]:
    active = OP_NAMES[: tok.config.n_ops]  # only the ops this task actually uses
    rng = np.random.default_rng(8)
    batch = make_batch(rng, tok, n_examples)

    correct = {name: 0 for name in active}
    total = {name: 0 for name in active}

    for ex in batch["examples"]:
        name = OP_NAMES[ex.op_id]
        prompt = build_prompt(tok, name, ex.xs)
        out = generate(model, tok, prompt)
        pred = decode_answer(jnp.array(out), tok)
        total[name] += 1
        if pred == ex.ys:
            correct[name] += 1

    return {name: correct[name] / total[name] for name in active}


# --- routing analysis -----------------------------------------------------
def get_routing(model: MoEGPT, ids: jax.Array) -> jax.Array:
    x = model.emb(ids)
    block = model.blocks[0]
    h = x + block.attn(block.norm1(x))
    weights, _ = block.moe.router(block.norm2(h))
    top_expert = jnp.argmax(weights, axis=-1)
    return top_expert


def route_counts(
    model: MoEGPT, tok: Tokenizer, n_experts: int = 4, n_examples: int = 2000
) -> np.ndarray:
    rng = np.random.default_rng(0)
    batch = make_batch(rng, tok, n_examples)

    ids = jnp.array(batch["tokens"])
    op_ids = batch["op_ids"]
    mask = batch["answer_mask"]

    top_expert = np.array(get_routing(model, ids))

    counts = np.zeros((len(OP_NAMES), n_experts))

    B, T = ids.shape

    for b in range(B):
        for t in range(T):
            if mask[b, t]:
                counts[op_ids[b], top_expert[b, t]] += 1
    return counts


# --- cost / routing metrics ----------------------------------------------
def param_count(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


def inference_speed(model: nnx.Module, tok: Tokenizer, n: int = 100) -> float:
    rng = np.random.default_rng(0)
    generate(model, tok, build_prompt(tok, OP_NAMES[0], [1, 2, 3]))  # warmup (JIT)
    t0, total = time.perf_counter(), 0
    for _ in range(n):
        op = OP_NAMES[rng.integers(tok.config.n_ops)]
        xs = rng.integers(0, tok.config.n_dig, size=4).tolist()
        out = generate(model, tok, build_prompt(tok, op, xs))
        total += len(out)
    return total / (time.perf_counter() - t0)


def routing_metrics(model: nnx.Module, tok: Tokenizer) -> dict:
    counts = route_counts(model, tok)
    util = counts.sum(axis=0) / counts.sum()
    p = util[util > 0]  # drop zeros before log
    entropy = float(-(p * np.log(p)).sum())
    return {
        "utilization": util.tolist(),
        "routing_entropy": entropy,
        "dead_experts": int((util < 0.02).sum()),
    }


# --- data-sheet record ----------------------------------------------------
def measure(
    config: Config, model: nnx.Module, tok: Tokenizer, train_time_s: float
) -> dict:
    acc = evaluate(model, tok)
    rec = {
        **asdict(config),  # every config knob, logged automatically
        "seq_len": tok.seq_len,
        "vocab_size": tok.vocab_size,
        "params": param_count(model),
        "train_time_s": round(train_time_s, 2),
        "tokens_per_sec": round(inference_speed(model, tok), 1),
        "overall_acc": float(np.mean(list(acc.values()))),
        "per_op_acc": acc,
    }
    if config.n_experts is not None:
        rec.update(routing_metrics(model, tok))
    return rec


def append_record(rec: dict, path: str = "datasheet.jsonl") -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# --- plots ----------------------------------------------------------------
def plot_accuracy(
    dense: dict[str, float], moe: dict[str, float], path: str = "figures/accuracy.svg"
) -> None:
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


def plot_routing(counts: np.ndarray, path: str = "figures/routing.svg") -> None:
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
