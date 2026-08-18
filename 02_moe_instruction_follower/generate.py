import jax
import jax.numpy as jnp
from baseline import GPT
from data import SEQ_LEN, Tokenizer, decode_answer
from train import train


def generate(model: GPT, tok: Tokenizer, prompt_ids: jax.Array, max_new=9):
    pos = len(prompt_ids)
    seq = list(prompt_ids) + [tok.pad_id] * (SEQ_LEN - pos)
    for _ in range(max_new):
        ids = jnp.array([seq])
        logits = model(ids)
        next_id = int(jnp.argmax(logits[0, pos - 1]))
        seq[pos] = next_id
        pos += 1
        if next_id == tok.eos_id or pos >= SEQ_LEN:
            break
    return seq[:pos]


if __name__ == "__main__":
    model, tok = train(steps=1500)
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
