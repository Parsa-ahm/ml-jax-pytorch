"""The dense transformer: embeddings, multi-head causal attention, block, GPT."""

import jax
import jax.numpy as jnp
from flax import nnx


class Embeddings(nnx.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, rngs):
        self.token = nnx.Embed(num_embeddings=vocab_size, features=d_model, rngs=rngs)
        self.pos = nnx.Embed(num_embeddings=seq_len, features=d_model, rngs=rngs)

    def __call__(self, ids):
        positions = jnp.arange(ids.shape[1])
        return self.token(ids) + self.pos(positions)


class CausalSelfAttention(nnx.Module):
    def __init__(self, d_model: int, rngs: nnx.Rngs, n_heads: int = 1):
        self.k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def __call__(self, x):
        B, T, D = x.shape
        H = self.n_heads
        dh = self.d_head

        def split(t):
            return t.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        V = split(self.v(x))
        K = split(self.k(x))
        Q = split(self.q(x))
        scores = (jnp.matmul(Q, jnp.swapaxes(K, -1, -2))) / jnp.sqrt(dh)
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask == 0, -jnp.inf, scores)
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(weights, V)

        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(out)


class Block(nnx.Module):
    def __init__(self, d_model: int, rngs: nnx.Rngs, n_heads: int = 1):
        self.attn = CausalSelfAttention(d_model=d_model, rngs=rngs, n_heads=n_heads)
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.fc1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        h = self.fc2(jax.nn.relu(self.fc1(h)))
        x = x + h
        return x


class GPT(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        rngs: nnx.Rngs,
        n_heads: int = 1,
    ):
        self.emb = Embeddings(
            vocab_size=vocab_size, seq_len=seq_len, d_model=d_model, rngs=rngs
        )
        self.blocks = nnx.List(
            [
                Block(d_model=d_model, rngs=rngs, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def __call__(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.head(x)
