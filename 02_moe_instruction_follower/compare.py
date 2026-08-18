import os
import time

import jax.numpy as jnp
from checkpoint import load_model, save_model
from data import SEQ_LEN, Tokenizer, decode_answer
from generate import generate
from ops import OP_BY_NAME, OP_NAMES
from train import train


def build_prompt(tok: Tokenizer, op_name: str, xs: list):
    return (
        [tok.bos_id, tok.stoi[op_name]] + [tok.stoi[str(x)] for x in xs] + [tok.eq_id]
    )


STEP_COUNTS = [1500, 2500, 3500, 4500]
tok = Tokenizer()
models = {}

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
config = {
    "vocab_size": tok.vocab_size,
    "seq_len": SEQ_LEN,
    "d_model": 64,
    "n_layers": 2,
}

if __name__ == "__main__":
    for s in STEP_COUNTS:
        path = f"{CKPT_DIR}/dense_{s}.pkl"
        if os.path.exists(path):
            model = load_model(path)
        else:
            model, _ = train(
                steps=s, d_mode=config["d_model"], n_layers=config["n_layers"]
            )
            save_model(model, config, path)
        models[f"{s} steps"] = model

    while True:
        print("Operations (op):")
        for op in OP_NAMES:
            print(op)
        print("or type Quit to exit")
        op = input("op:").strip().upper()
        if op == "QUIT":
            break
        nums = [int(x) for x in input("numbers: ").split()]

        prompt = build_prompt(tok, op, nums)
        truth = OP_BY_NAME[op].solve(nums)
        print(truth)

        for label, model in models.items():
            t0 = time.perf_counter()
            out = generate(model, tok, prompt)
            dt = time.perf_counter() - t0
            pred = decode_answer(jnp.array(out), tok)
            flag = "✓" if pred == truth else "✗"
            print(f"  {label:11} {pred}  {flag}  ({dt * 1000:.1f} ms)")
