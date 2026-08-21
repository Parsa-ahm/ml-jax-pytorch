import jax.numpy as jnp


def load_balance_loss(probs: jnp.ndarray) -> jnp.ndarray:
    E = probs.shape[-1]
    tokens = probs.reshape(-1, E)

    P = tokens.mean(axis=0)
    top1 = jnp.argmax(tokens, axis=-1)
    f = jnp.bincount(top1, length=E) / tokens.shape[0]

    return E * jnp.sum(f * P)
