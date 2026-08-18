import os
import time

import jax.numpy as jnp
import questionary
from checkpoint import load_model, save_model
from data import SEQ_LEN, Tokenizer, decode_answer
from generate import generate
from ops import OP_BY_NAME, OP_NAMES
from rich.console import Console
from rich.live import Live
from rich.table import Table
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
                steps=s, d_model=config["d_model"], n_layers=config["n_layers"]
            )
            save_model(model, config, path)
        models[f"{s} steps"] = model

    console = Console()
    warm_prompt = build_prompt(tok, "SORT", [1, 2, 3])
    for model in models.values():
        generate(model, tok, warm_prompt)
    while True:
        op = questionary.select("Operations: ", choices=[*OP_NAMES, "QUIT"]).ask()
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
