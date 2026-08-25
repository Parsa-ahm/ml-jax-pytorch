"""Sparse Mixture-of-Experts forward: only the top-k experts run per token.

The naive MoELayer computes all E experts on every token, then discards the
unused ones (E x work). Here we build only the k (token, expert) pairs each
token routes to, group them by expert, and use jax.lax.ragged_dot (a grouped
matmul) so each expert's weights touch only its own tokens -- realizing MoE's
k/E compute saving.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def _expert_weights(moe):
    # stack each expert's Linear weights into (E, ...) tensors for ragged_dot
    W1 = jnp.stack([e.fc1.kernel[...] for e in moe.experts])
    b1 = jnp.stack([e.fc1.bias[...] for e in moe.experts])
    W2 = jnp.stack([e.fc2.kernel[...] for e in moe.experts])
    b2 = jnp.stack([e.fc2.bias[...] for e in moe.experts])
    return W1, b1, W2, b2


def sparse_moe(moe: nnx.Module, x: jax.Array, top_k: int) -> jax.Array:
    B, T, d = x.shape
    N = B * T
    E = len(moe.experts)
    xf = x.reshape(N, d)

    weights, _ = moe.router(x)
    gates, eidx = jax.lax.top_k(weights, top_k)
    gates = gates.reshape(N * top_k)
    eidx = eidx.reshape(N * top_k).astype(jnp.int32)
    tok = jnp.repeat(jnp.arange(N), top_k)

    perm = jnp.argsort(eidx)
    eidx_s, tok_s, gates_s = eidx[perm], tok[perm], gates[perm]
    rows = xf[tok_s]
    group_size = jnp.bincount(eidx_s, length=E).astype(jnp.int32)

    W1, b1, W2, b2 = _expert_weights(moe)
    up = jax.lax.ragged_dot(rows, W1, group_size) + b1[eidx_s]
    up = jax.nn.relu(up)
    down = jax.lax.ragged_dot(up, W2, group_size) + b2[eidx_s]

    down = down * gates_s[:, None]
    out = jnp.zeros((N, d), dtype=x.dtype).at[tok_s].add(down)
    return out.reshape(B, T, d)
