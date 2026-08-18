import pickle

from baseline import GPT
from flax import nnx


def save_model(model: GPT, config: dict, path: str) -> None:
    blob = {"config": config, "state": nnx.state(model)}
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def load_model(path: str) -> GPT:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    c = blob["config"]
    model = GPT(
        c["vocab_size"], c["seq_len"], c["d_model"], c["n_layers"], rngs=nnx.Rngs(0)
    )
    nnx.update(model, blob["state"])
    return model
