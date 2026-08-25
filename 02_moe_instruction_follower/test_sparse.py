import jax.numpy as jnp
import numpy as np
from flax import nnx
from moe import MoELayer
from sparse import sparse_moe


def test_sparse_matches_naive():
    E = 4
    k = 2

    moe = MoELayer(64, E, k, rngs=nnx.Rngs(0))
    x = jnp.array(np.random.default_rng(0).normal(size=(8, 20, 64)), dtype=jnp.float32)
    naive, _ = moe(x)
    sp = sparse_moe(moe, x, k)
    assert float(jnp.abs(naive - sp).max()) <= 1e-4
