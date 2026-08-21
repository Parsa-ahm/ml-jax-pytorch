from baseline import GPT
from config import Config
from flax import nnx
from moe import MoEGPT


def build_model(config: Config, rngs: nnx.Rngs) -> nnx.Module:
    # n_experts is None -> dense GPT; otherwise a Mixture-of-Experts model.
    if config.n_experts is None:
        return GPT(
            config.vocab_size,
            config.seq_len,
            config.d_model,
            config.n_layers,
            rngs,
            n_heads=config.n_head,
        )
    return MoEGPT(
        config.vocab_size,
        config.seq_len,
        config.d_model,
        config.n_layers,
        config.n_experts,
        config.top_k,
        rngs,
        n_heads=config.n_head,
    )
