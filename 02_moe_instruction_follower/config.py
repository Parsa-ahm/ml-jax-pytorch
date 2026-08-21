from dataclasses import dataclass


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
