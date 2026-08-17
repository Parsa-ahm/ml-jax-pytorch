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
    def __init__(self, d_model: int, rngs: nnx.Rngs):
        self.k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.q = nnx.Linear(d_model, d_model, rngs=rngs)
        self.proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.d_model = d_model

    def __call__(self, x):
        K = self.k(x)
        V = self.v(x)
        Q = self.q(x)
        T = x.shape[1]
        scores = (jnp.matmul(Q, jnp.swapaxes(K, -1, -2))) / jnp.sqrt(self.d_model)
        mask = jnp.tril(jnp.ones((T, T)))
        scores = jnp.where(mask == 0, -jnp.inf, scores)
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(weights, V)
        return self.proj(out)


class Block(nnx.Module):
    def __init__(self, d_model: int, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(d_model=d_model, rngs=rngs)
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
    ):
        self.emb = Embeddings(
            vocab_size=vocab_size, seq_len=seq_len, d_model=d_model, rngs=rngs
        )
        self.blocks = nnx.List(
            [Block(d_model=d_model, rngs=rngs) for _ in range(n_layers)]
        )
        self.norm_f = nnx.LayerNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

    def __call__(self, ids):
        x = self.emb(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.head(x)
