import jax
import jax.numpy as jnp
import numpy as np
import optax
from config import Config
from data import Tokenizer, make_batch
from flax import nnx
from models import build_model


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
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    ids: jax.Array,
    answer_mask: jax.Array,
    balance_coef: float,
) -> jax.Array:

    def loss_fn(model):
        if hasattr(model, "forward_with_aux"):
            logits, aux = model.forward_with_aux(ids)
        else:
            logits, aux = model(ids), 0.0
        return compute_loss(logits, ids, answer_mask) + balance_coef * aux

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(config: Config) -> tuple[nnx.Module, Tokenizer]:
    tok = Tokenizer(config)
    model = build_model(config, nnx.Rngs(config.seed))
    optimizer = nnx.Optimizer(model, optax.adamw(config.lr), wrt=nnx.Param)
    rng = np.random.default_rng(config.seed)
    for step in range(config.steps):
        batch = make_batch(rng, tok, config.batch_size)
        ids = jnp.array(batch["tokens"])
        mask = jnp.array(batch["answer_mask"])
        loss = training_step(
            model,
            optimizer=optimizer,
            ids=ids,
            answer_mask=mask,
            balance_coef=config.balance_coef,
        )
        if step % 100 == 0:
            print(step, float(loss))
    return model, tok


if __name__ == "__main__":
    train(Config())
