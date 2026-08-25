"""build_model, the training loop, generation, and checkpointing."""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from core import Config, Tokenizer, decode_answer, make_batch
from dense import GPT
from flax import nnx
from moe import MoEGPT


# --- model construction ---------------------------------------------------
def build_model(config: Config, rngs: nnx.Rngs) -> nnx.Module:
    # n_experts is None -> dense GPT; otherwise a Mixture-of-Experts model.
    if config.n_experts is None:
        return GPT(
            config.vocab_size,
            config.seq_len,
            config.d_model,
            config.n_layers,
            rngs,
            n_heads=config.n_head,
        )
    return MoEGPT(
        config.vocab_size,
        config.seq_len,
        config.d_model,
        config.n_layers,
        config.n_experts,
        config.top_k,
        rngs,
        n_heads=config.n_head,
    )


# --- training -------------------------------------------------------------
def compute_loss(
    logits: jax.Array, ids: jax.Array, answer_mask: jax.Array
) -> jax.Array:
    shift_logits = logits[:, :-1, :]
    targets = ids[:, 1:]
    mask = answer_mask[:, 1:] * 1.0
    per_pos = optax.softmax_cross_entropy_with_integer_labels(shift_logits, targets)
    masked_mean = (per_pos * (mask)).sum() / (mask).sum()
    return masked_mean


@nnx.jit
def training_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    ids: jax.Array,
    answer_mask: jax.Array,
    balance_coef: float,
) -> jax.Array:
    def loss_fn(model):
        if hasattr(model, "forward_with_aux"):
            logits, aux = model.forward_with_aux(ids)
        else:
            logits, aux = model(ids), 0.0
        return compute_loss(logits, ids, answer_mask) + balance_coef * aux

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(config: Config) -> tuple[nnx.Module, Tokenizer]:
    tok = Tokenizer(config)
    model = build_model(config, nnx.Rngs(config.seed))
    optimizer = nnx.Optimizer(model, optax.adamw(config.lr), wrt=nnx.Param)
    rng = np.random.default_rng(config.seed)
    for step in range(config.steps):
        batch = make_batch(rng, tok, config.batch_size)
        ids = jnp.array(batch["tokens"])
        mask = jnp.array(batch["answer_mask"])
        loss = training_step(
            model,
            optimizer=optimizer,
            ids=ids,
            answer_mask=mask,
            balance_coef=config.balance_coef,
        )
        if step % 100 == 0:
            print(step, float(loss))
    return model, tok


# --- inference / generation ----------------------------------------------
def build_prompt(tok: Tokenizer, op_name: str, xs: list[int]) -> list[int]:
    return (
        [tok.bos_id, tok.stoi[op_name]] + [tok.stoi[str(x)] for x in xs] + [tok.eq_id]
    )


def generate(
    model: nnx.Module, tok: Tokenizer, prompt_ids: list[int], max_new: int = 9
) -> list[int]:
    pos = len(prompt_ids)
    seq = list(prompt_ids) + [tok.pad_id] * (tok.seq_len - pos)
    for _ in range(max_new):
        ids = jnp.array([seq])
        logits = model(ids)
        next_id = int(jnp.argmax(logits[0, pos - 1]))
        seq[pos] = next_id
        pos += 1
        if next_id == tok.eos_id or pos >= tok.seq_len:
            break
    return seq[:pos]


# --- checkpointing (trusted local files only) -----------------------------
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
    n_head = c.get("n_head", 1)
    if mt == "GPT":
        model = GPT(
            c["vocab_size"],
            c["seq_len"],
            c["d_model"],
            c["n_layers"],
            nnx.Rngs(0),
            n_heads=n_head,
        )
    elif mt == "MoE":
        model = MoEGPT(
            c["vocab_size"],
            c["seq_len"],
            c["d_model"],
            c["n_layers"],
            c["n_experts"],
            c["top_k"],
            nnx.Rngs(0),
            n_heads=n_head,
        )
    else:
        raise ValueError(f"Unknown model_type: {mt}")
    nnx.update(model, blob["state"])
    return model


if __name__ == "__main__":
    model, tok = train(Config(steps=1500))
    prompt = build_prompt(tok, "SORT", [4, 5, 3, 1])
    print(decode_answer(jnp.array(generate(model, tok, prompt)), tok))
