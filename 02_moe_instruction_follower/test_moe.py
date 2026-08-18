import jax.numpy as jnp
import numpy as np
import pytest
from baseline import Embeddings
from data import SEQ_LEN, Tokenizer, make_batch
from flax import nnx
from moe import MoEGPT, MoELayer, Router

D_MODEL = 64
N_EXPERTS = 4
TOP_K = 2


@pytest.fixture
def tok():
    return Tokenizer()


@pytest.fixture
def ids(tok):
    return make_batch(np.random.default_rng(0), tok, 8)["tokens"]


@pytest.fixture
def x(tok, ids):
    return Embeddings(tok.vocab_size, SEQ_LEN, D_MODEL, rngs=nnx.Rngs(0))(ids)


def test_router_shape(x):
    w = Router(D_MODEL, N_EXPERTS, TOP_K, rngs=nnx.Rngs(0))(x)
    assert w.shape == (8, SEQ_LEN, N_EXPERTS)


def test_router_sums_to_one(x):
    w = Router(D_MODEL, N_EXPERTS, TOP_K, rngs=nnx.Rngs(0))(x)
    assert bool(jnp.allclose(w.sum(-1), 1.0))


def test_router_topk_sparsity(x):
    w = Router(D_MODEL, N_EXPERTS, TOP_K, rngs=nnx.Rngs(0))(x)
    assert bool(jnp.all((w > 0).sum(-1) == TOP_K))


def test_moelayer_shape(x):
    out = MoELayer(D_MODEL, N_EXPERTS, TOP_K, rngs=nnx.Rngs(0))(x)
    assert out.shape == (8, SEQ_LEN, D_MODEL)


def test_moegpt_logits(tok, ids):
    model = MoEGPT(
        tok.vocab_size, SEQ_LEN, D_MODEL, 2, N_EXPERTS, TOP_K, rngs=nnx.Rngs(0)
    )
    logits = model(ids)
    assert logits.shape == (8, SEQ_LEN, tok.vocab_size)
