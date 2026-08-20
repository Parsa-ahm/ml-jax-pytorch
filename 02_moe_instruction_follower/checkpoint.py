import pickle

from baseline import GPT
from flax import nnx
from moe import MoEGPT


def save_model(model: nnx.Module, config: dict, path: str) -> None:
    blob = {
        "config": config,
        "state": nnx.state(model),
        "model_type": "GPT" if type(model) is GPT else "MoE",
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def load_model(path: str) -> nnx.Module:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    c = blob["config"]
    mt = blob["model_type"]
    if mt == "GPT":
        model = GPT(
            c["vocab_size"], c["seq_len"], c["d_model"], c["n_layers"], rngs=nnx.Rngs(0)
        )
    elif mt == "MoE":
        model = MoEGPT(
            c["vocab_size"],
            c["seq_len"],
            c["d_model"],
            c["n_layers"],
            c["n_experts"],
            c["top_k"],
            rngs=nnx.Rngs(0),
        )
    else:
        raise ValueError(f"Unknown model_type: {mt}")
    nnx.update(model, blob["state"])
    return model
