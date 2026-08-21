import jax.numpy as jnp
from config import Config
from data import Tokenizer, decode_answer
from flax import nnx
from train import train


def build_prompt(tok: Tokenizer, op_name: str, xs: list[int]) -> list[int]:
    return (
        [tok.bos_id, tok.stoi[op_name]] + [tok.stoi[str(x)] for x in xs] + [tok.eq_id]
    )


def generate(
    model: nnx.Module, tok: Tokenizer, prompt_ids: list[int], max_new: int = 9
) -> list[int]:
    pos = len(prompt_ids)
    seq = list(prompt_ids) + [tok.pad_id] * (tok.seq_len - pos)
    for _ in range(max_new):
        ids = jnp.array([seq])
        logits = model(ids)
        next_id = int(jnp.argmax(logits[0, pos - 1]))
        seq[pos] = next_id
        pos += 1
        if next_id == tok.eos_id or pos >= tok.seq_len:
            break
    return seq[:pos]


if __name__ == "__main__":
    model, tok = train(Config(steps=1500))
    prompt = [
        tok.bos_id,
        tok.stoi["SORT"],
        tok.stoi["4"],
        tok.stoi["5"],
        tok.stoi["3"],
        tok.stoi["1"],
        tok.eq_id,
    ]
    out = generate(model, tok, prompt)
    print(decode_answer(jnp.array(out), tok))
