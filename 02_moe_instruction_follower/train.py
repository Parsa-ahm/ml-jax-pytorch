import jax
import jax.numpy as jnp
import numpy as np
import optax
from baseline import GPT
from data import SEQ_LEN, Tokenizer, make_batch
from flax import nnx


def compute_loss(
    logits: jax.Array, ids: jax.Array, answer_mask: jax.Array
) -> jax.Array:
    shift_logits = logits[:, :-1, :]
    targets = ids[:, 1:]
    mask = answer_mask[:, 1:] * 1.0
    per_pos = optax.softmax_cross_entropy_with_integer_labels(shift_logits, targets)
    masked_mean = (per_pos * (mask)).sum() / (mask).sum()
    return masked_mean


@nnx.jit
def training_step(
    model: GPT, optimizer: nnx.Optimizer, ids: jax.Array, answer_mask: jax.Array
) -> jax.Array:
    def loss_fn(model):
        logits = model(ids)
        return compute_loss(logits, ids, answer_mask)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(
    steps: int = 2000,
    batch_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[GPT, Tokenizer]:
    tok = Tokenizer()
    model = GPT(tok.vocab_size, SEQ_LEN, d_model, n_layers, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)
    rng = np.random.default_rng()
    for step in range(steps):
        batch = make_batch(rng, tok, batch_size)
        ids = jnp.array(batch["tokens"])
        mask = jnp.array(batch["answer_mask"])
        loss = training_step(model, optimizer=optimizer, ids=ids, answer_mask=mask)
        if step % 100 == 0:
            print(step, float(loss))
    return model, tok


if __name__ == "__main__":
    train()
