import jax
import jax.numpy as jnp
import numpy as np
from data import Tokenizer, make_batch
from moe import MoEGPT
from ops import OP_NAMES


def get_routing(model: MoEGPT, ids: jax.Array) -> jax.Array:
    x = model.emb(ids)
    block = model.blocks[0]
    h = x + block.attn(block.norm1(x))
    weights = block.moe.router(block.norm2(h))
    top_expert = jnp.argmax(weights, axis=-1)
    return top_expert


def route_counts(
    model: MoEGPT, tok: Tokenizer, n_experts: int = 4, n_examples: int = 2000
) -> np.array:
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
