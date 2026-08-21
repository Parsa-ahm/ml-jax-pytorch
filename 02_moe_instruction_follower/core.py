from dataclasses import dataclass

import numpy as np
from ops import OP_NAMES, OPS, Op


@dataclass(frozen=True)
class Config:
    # Operation and inputs
    n_ops: int = 8
    n_dig: int = 10
    min_input_len: int = 3
    max_input_len: int = 8

    # model
    d_model: int = 64
    n_layers: int = 2
    n_head: int = 1

    # MoE
    n_experts: int | None = None
    top_k: int = 2
    balance_coef: float = 0.0

    # training
    steps: int = 8000
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0

    @property
    def seq_len(self) -> int:
        return 2 * self.max_input_len + 4

    @property
    def vocab_size(self) -> int:
        return 4 + self.n_dig + self.n_ops


class Tokenizer:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        c = self.config
        specials = ["<pad>", "<bos>", "<eos>", "="]
        numbers = [str(i) for i in range(c.n_dig)]
        op_names = list(OP_NAMES[: c.n_ops])
        self.itos = specials + numbers + op_names
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.eq_id = self.stoi["="]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def seq_len(self) -> int:
        return self.config.seq_len


def sample_inputs(rng: np.random.Generator, config: Config) -> list[int]:
    n = rng.integers(config.min_input_len, config.max_input_len + 1)
    return rng.integers(0, config.n_dig, size=n).tolist()


@dataclass
class Example:
    xs: list[int]
    ys: list[int]
    op_id: int
    sample: list[str]
    token_ids: list[int]
    answer_mask: list[bool]


def make_example(rng: np.random.Generator, op: Op, tok: Tokenizer) -> Example:
    c = tok.config
    xs = sample_inputs(rng, c)
    ys = op.solve(xs)
    xs_string = [str(i) for i in xs]
    ys_string = [str(i) for i in ys]
    op_id = OP_NAMES.index(op.name)
    sample = ["<bos>", op.name] + xs_string + ["="] + ys_string + ["<eos>"]
    padding_length = c.seq_len - len(sample)
    sample = sample + ["<pad>"] * padding_length
    token_ids = [tok.stoi[s] for s in sample]
    answer_mask = (
        [False, False]
        + [False] * len(xs)
        + [False]
        + [True] * len(ys)
        + [True]
        + [False] * padding_length
    )
    return Example(
        xs,
        ys,
        op_id,
        sample,
        token_ids,
        answer_mask,
    )


def make_batch(rng: np.random.Generator, tok: Tokenizer, batch_size: int) -> dict:
    c = tok.config
    examples = []
    for _ in range(batch_size):
        op = OPS[rng.integers(c.n_ops)]
        examples.append(make_example(rng, op, tok))
    tokens = np.array([e.token_ids for e in examples], dtype=np.int32)
    answer_mask = np.array([e.answer_mask for e in examples], dtype=bool)
    op_ids = np.array([e.op_id for e in examples], dtype=np.int32)

    batch = {
        "tokens": tokens,
        "answer_mask": answer_mask,
        "op_ids": op_ids,
        "examples": examples,
    }

    return batch


def decode_answer(token_row: list[int], tok: Tokenizer) -> list[int]:
    decoded_row = []
    in_answer = False
    for tid in token_row:
        if in_answer and tok.itos[tid].isdigit():
            decoded_row.append(int(tok.itos[tid]))
        elif tid == tok.eq_id:
            in_answer = True
        if in_answer and tid == tok.eos_id:
            in_answer = False
            break
    return decoded_row


def grade(pred_answers: list[int], example: Example) -> bool:
    return pred_answers == example.ys
