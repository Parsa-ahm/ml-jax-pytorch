import numpy as np
import pytest
from core import Tokenizer, decode_answer, grade, make_batch


@pytest.fixture
def tok():
    return Tokenizer()


@pytest.fixture
def batch(tok):
    return make_batch(np.random.default_rng(), tok, 20)


def test_batch_shape(batch):
    assert batch["tokens"].shape == (20, 20)
    assert batch["answer_mask"].shape == (20, 20)


def test_roundtrip(tok, batch):
    for i in range(len(batch["examples"])):
        ex = batch["examples"][i]
        assert decode_answer(batch["tokens"][i], tok) == ex.ys


def test_vocab_roundtrip(tok):
    for token, idx in tok.stoi.items():
        assert tok.itos[idx] == token


def test_vocab(tok):
    assert tok.vocab_size == 22


def test_alignment(batch):
    for ex in batch["examples"]:
        assert sum(ex.answer_mask) == len(ex.ys) + 1


def test_grade(batch, tok):
    decoded_row = decode_answer(batch["tokens"][0], tok)
    grade(decoded_row, batch["examples"][0])
