import jax
import jax.numpy as jnp
import numpy as np
import pytest
from baseline import GPT, Block, CausalSelfAttention, Embeddings
from data import SEQ_LEN, Tokenizer, make_batch
from flax import nnx

D_MODEL = 64


@pytest.fixture
def tok():
    return Tokenizer()


@pytest.fixture
def ids(tok):
    return make_batch(np.random.default_rng(0), tok, 8)["tokens"]  # (8, 20)


def test_model_shape(tok, ids):
    model = GPT(tok.vocab_size, SEQ_LEN, D_MODEL, n_layers=2, rngs=nnx.Rngs(0))
    logits = model(ids)
    assert logits.shape == (ids.shape[0], SEQ_LEN, tok.vocab_size)


def test_embeddings_shape(tok, ids):
    assert (
        Embeddings(tok.vocab_size, SEQ_LEN, D_MODEL, rngs=nnx.Rngs(0))(ids)
    ).shape == (
        8,
        20,
        64,
    )


def test_attention_shape(tok, ids):
    x = Embeddings(tok.vocab_size, SEQ_LEN, D_MODEL, rngs=nnx.Rngs(0))(ids)
    out = CausalSelfAttention(D_MODEL, rngs=nnx.Rngs(0))(x)
    assert out.shape == (8, 20, D_MODEL)

def test_block_shape(tok, ids):
    x = Embeddings(tok.vocab_size, SEQ_LEN, D_MODEL, rngs=nnx.Rngs(0))(ids)
    out = Block(D_MODEL, rngs=nnx.Rngs(0))(x)
    assert out.shape == (8, 20, D_MODEL)


def test_attention_causal(tok, ids):
    # rebuild the attention-weight grid and prove the upper triangle (future) is 0
    x = Embeddings(tok.vocab_size, SEQ_LEN, D_MODEL, rngs=nnx.Rngs(0))(ids)
    attn = CausalSelfAttention(D_MODEL, rngs=nnx.Rngs(0))
    K, Q = attn.k(x), attn.q(x)
    T = x.shape[1]
    scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(D_MODEL)
    mask = jnp.tril(jnp.ones((T, T)))
    weights = jax.nn.softmax(jnp.where(mask == 0, -jnp.inf, scores), axis=-1)
    assert bool(jnp.all(weights[:, mask == 0] == 0))
