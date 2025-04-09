import mindspore as ms

from .config import Config, DTYPE_MAPPING

class DeepseekV2Config(Config):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        qk_nope_head_dim: int,
        activate_fn: str = 'gated_silu',
        norm_eps: float = 1e-6,
        dtype: str = 'fp16',
        rope_factor: float = 1.0,
        rope_original_max_position_embeddings: int = 4096,
        rope_beta_fast: int = 32,
        rope_beta_slow: int = 1,
        rope_mscale: float = 1.0,
        rope_mscale_all_dim: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.activate_fn = activate_fn
        self.norm_eps = norm_eps
        self.dtype = DTYPE_MAPPING[dtype]
        self.rope_factor = rope_factor
        self.original_max_position_embeddings = rope_original_max_position_embeddings
        self.rope_beta_fast = rope_beta_fast
        self.rope_beta_slow = rope_beta_slow
        self.rope_mscale = rope_mscale
        self.rope_mscale_all_dim = rope_mscale_all_dim