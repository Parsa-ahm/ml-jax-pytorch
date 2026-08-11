from dataclasses import dataclass

import numpy as np
from ops import OP_NAMES, OPS, Op

MIN_INPUT_LEN = 3
MAX_INPUT_LEN = 8
SEQ_LEN = 20


class Tokenizer:
    def __init__(self):
        specials = ["<pad>", "<bos>", "<eos>", "="]
        numbers = [str(i) for i in range(10)]
        self.itos = specials + numbers + list(OP_NAMES)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.eq_id = self.stoi["="]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)


def sample_inputs(rng: np.random.Generator) -> list[int]:
    n = rng.integers(MIN_INPUT_LEN, MAX_INPUT_LEN + 1)
    return rng.integers(0, 10, size=n).tolist()


@dataclass
class Example:
    xs: list[int]
    ys: list[int]
    op_id: int
    sample: list[str]
    token_id: list[int]
    answer_mask: list[bool]


def make_example(rng: np.random.Generator, op: Op, tok: Tokenizer) -> Example:
    xs = sample_inputs(rng)
    ys = op.solve(xs)
    xs_string = [str(i) for i in xs]
    ys_string = [str(i) for i in ys]
    op_id = OP_NAMES.index(op.name)
    sample = ["<bos>", op.name] + xs_string + ["="] + ys_string + ["<eos>"]
    padding_length = SEQ_LEN - len(sample)
    sample = sample + ["<pad>"] * padding_length
    token_id = [tok.stoi[s] for s in sample]
    answer_mask = (
        [False, False]
        + [False] * len(xs)
        + [False]
        + [True] * len(ys)
        + [True]
        + [False] * padding_length
    )
    example = Example(
        xs,
        ys,
        op_id,
        sample,
        token_id,
        answer_mask,
    )
    return example


def make_batch(rng: np.random.Generator, tok: Tokenizer, batch_size: int) -> dict:
    examples = []
    for _ in range(batch_size):
        op = OPS[rng.integers(len(OPS))]
        examples.append(make_example(rng, op, tok))
    tokens = np.array([e.token_id for e in examples], dtype=np.int32)
    answer_mask = np.array([e.answer_mask for e in examples], dtype=bool)
    op_ids = np.array([e.op_id for e in examples], dtype=np.int32)

    batch = {
        "tokens": tokens,
        "answer_mask": answer_mask,
        "op_ids": op_ids,
        "examples": examples,
    }

    return batch


def decode_answer(token_row: list[str], tok: Tokenizer) -> list[int]:
    decoded_row = []
    in_answer = False
    for i in token_row:
        if in_answer and tok.itos[i].isdigit():
            decoded_row = decoded_row + [int(tok.itos[i])]
        elif i == tok.eq_id:
            in_answer = True
        if in_answer and i == tok.eos_id:
            in_answer = False
            break
    return decoded_row


def grade(pred_answers: list[int], example: list) -> bool:
    return pred_answers == example.ys


if __name__ == "__main__":
    rng = np.random.default_rng()
    batch_size = 20
    tok = Tokenizer()
    sample = make_batch(rng, tok, batch_size)
    print(sample["tokens"].shape)
    print(sample["answer_mask"].shape)
    print(len(sample["examples"]))
    decoded_row = decode_answer(sample["tokens"][0], tok)
    print(grade(decoded_row, sample["examples"][0]))
