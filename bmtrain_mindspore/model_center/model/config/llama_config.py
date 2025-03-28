import mindspore as ms

from .config import Config, DTYPE_MAPPING

class LlamaConfig(Config):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim_model: int,
        dim_head: int,
        dim_ff: int,
        num_heads: int,
        activate_fn: str = 'gated_silu',
        norm_eps: float = 1e-6,
        dtype: str = 'fp16'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.activate_fn = activate_fn
        self.norm_eps = norm_eps
        self.dtype = DTYPE_MAPPING[dtype]