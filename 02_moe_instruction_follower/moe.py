import jax
import jax.numpy as jnp
from baseline import CausalSelfAttention, Embeddings
from flax import nnx


class Router(nnx.Module):
    def __init__(self, d_model, n_experts, top_k, rngs):
        self.gate = nnx.Linear(d_model, n_experts, rngs=rngs)
        self.top_k = top_k

    def __call__(self, x):
        scores = self.gate(x)
        top_vals, _ = jax.lax.top_k(scores, self.top_k)
        threshold = top_vals[..., -1:]
        masked = jnp.where(scores < threshold, -jnp.inf, scores)
        weights = jax.nn.softmax(masked, axis=-1)
        return weights


class Expert(nnx.Module):
    def __init__(self, d_model, rngs):
        self.fc1 = nnx.Linear(d_model, 4 * d_model, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d_model, d_model, rngs=rngs)

    def __call__(self, x):
        return self.fc2(jax.nn.relu(self.fc1(x)))


class MoELayer(nnx.Module):
    def __init__(self, d_model, n_experts, top_k, rngs):
        self.router = Router(
            d_model=d_model, n_experts=n_experts, top_k=top_k, rngs=rngs
        )
        self.experts = nnx.List(
            [Expert(d_model=d_model, rngs=rngs) for _ in range(n_experts)]
        )

    def __call__(self, x):
        weights = self.router(x)
        outs = jnp.stack([e(x) for e in self.experts], axis=2)
        return (outs * weights[..., None]).sum(axis=2)


class MoEBlock(nnx.Module):
    def __init__(self, d_model, n_experts, top_k, rngs):
        self.attn = CausalSelfAttention(d_model=d_model, rngs=rngs)
        self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.moe = MoELayer(d_model, n_experts, top_k, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class MoEGPT(nnx.Module):
    def __init__(self, vocab_size, seq_len, d_model, n_layers, n_experts, top_k, rngs):
        self.emb = Embeddings(vocab_size, seq_len, d_model, rngs=rngs)
        self.blocks = nnx.List(
            [
                MoEBlock(d_model=d_model, n_experts=n_experts, top_k=top_k, rngs=rngs)
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
