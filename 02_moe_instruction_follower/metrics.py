import json
import time
from dataclasses import asdict

import jax
import numpy as np
from analysis import route_counts
from config import Config
from data import Tokenizer
from evaluate import evaluate
from flax import nnx
from generate import build_prompt, generate
from ops import OP_NAMES


def param_count(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


def inference_speed(model: nnx.Module, tok: Tokenizer, n: int = 100) -> float:
    rng = np.random.default_rng(0)
    generate(model, tok, build_prompt(tok, OP_NAMES[0], [1, 2, 3]))
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
    p = util[util > 0]
    entropy = float(-(p * np.log(p)).sum())
    return {
        "utilization": util.tolist(),
        "routing_entropy": entropy,
        "dead_experts": int((util < 0.02).sum()),
    }


def measure(
    config: Config, model: nnx.Module, tok: Tokenizer, train_time_s: float
) -> dict:
    acc = evaluate(model, tok)
    rec = {
        **asdict(config),
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
