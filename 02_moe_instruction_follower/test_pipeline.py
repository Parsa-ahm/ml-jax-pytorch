import math

import jax.numpy as jnp
import numpy as np
import pytest
from analysis import get_routing, route_counts
from core import Tokenizer, make_batch
from dense import GPT
from flax import nnx
from moe import MoEGPT
from ops import OP_NAMES
from train import build_prompt, compute_loss, load_model, save_model

D_MODEL, N_EXPERTS, TOP_K = 64, 4, 2


@pytest.fixture
def tok():
    return Tokenizer()


@pytest.fixture
def ids(tok):
    return make_batch(np.random.default_rng(0), tok, 8)["tokens"]


# --- build_prompt ---------------------------------------------------------
def test_build_prompt_layout(tok):
    # prompt is [<bos>, OP, digits..., =]  — no answer, no <eos>
    p = build_prompt(tok, "SORT", [4, 1, 3])
    assert p[0] == tok.bos_id
    assert p[1] == tok.stoi["SORT"]
    assert p[-1] == tok.eq_id
    assert p[2:-1] == [tok.stoi[str(x)] for x in [4, 1, 3]]


# --- compute_loss ---------------------------------------------------------
def test_loss_untrained_near_uniform(tok, ids):
    # an untrained model bets ~uniformly over the vocab -> loss ~ log(vocab)
    model = GPT(tok.vocab_size, tok.seq_len, D_MODEL, 2, rngs=nnx.Rngs(0))
    batch = make_batch(np.random.default_rng(0), tok, 8)
    loss = compute_loss(model(ids), ids, jnp.array(batch["answer_mask"]))
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert abs(float(loss) - math.log(tok.vocab_size)) < 1.5


# --- checkpoint roundtrip (the save/load contract) ------------------------
def test_checkpoint_roundtrip_dense(tok, ids, tmp_path):
    model = GPT(tok.vocab_size, tok.seq_len, D_MODEL, 2, rngs=nnx.Rngs(1))
    cfg = {
        "vocab_size": tok.vocab_size,
        "seq_len": tok.seq_len,
        "d_model": D_MODEL,
        "n_layers": 2,
    }
    path = str(tmp_path / "dense.pkl")
    save_model(model, cfg, path)
    loaded = load_model(path)
    assert bool(jnp.allclose(model(ids), loaded(ids)))


def test_checkpoint_roundtrip_moe(tok, ids, tmp_path):
    model = MoEGPT(
        tok.vocab_size, tok.seq_len, D_MODEL, 2, N_EXPERTS, TOP_K, rngs=nnx.Rngs(2)
    )
    cfg = {
        "vocab_size": tok.vocab_size,
        "seq_len": tok.seq_len,
        "d_model": D_MODEL,
        "n_layers": 2,
        "n_experts": N_EXPERTS,
        "top_k": TOP_K,
    }
    path = str(tmp_path / "moe.pkl")
    save_model(model, cfg, path)
    loaded = load_model(path)
    assert bool(jnp.allclose(model(ids), loaded(ids)))


# --- routing analysis shapes ---------------------------------------------
def test_get_routing_shape(tok, ids):
    model = MoEGPT(
        tok.vocab_size, tok.seq_len, D_MODEL, 2, N_EXPERTS, TOP_K, rngs=nnx.Rngs(0)
    )
    te = get_routing(model, ids)
    assert te.shape == ids.shape
    assert int(te.min()) >= 0 and int(te.max()) < N_EXPERTS


def test_route_counts_shape(tok):
    model = MoEGPT(
        tok.vocab_size, tok.seq_len, D_MODEL, 2, N_EXPERTS, TOP_K, rngs=nnx.Rngs(0)
    )
    counts = route_counts(model, tok, n_experts=N_EXPERTS, n_examples=200)
    assert counts.shape == (len(OP_NAMES), N_EXPERTS)
    assert counts.sum() > 0
