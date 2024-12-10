import numpy as np
import mindspore as ms

from mindspore import Tensor, nn
from mindnlp.core import ops
from mindspore.nn import Cell


class RotaryEmbeddingESM(Cell):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self, 
        dim: int, 
        base: float = 10000,
        distance_scale: float = 1,
    ):
        super().__init__()
        self.distance_scale = distance_scale

        # Generate and save the inverse frequency buffer (non trainable)
        self.inv_freq: Tensor = base ** (- ops.arange(0, dim, 2, dtype=ms.float32) / dim) # (dim/2, )

        self.seq_len_cached = -1
        self.cos_cached = None
        self.sin_cached = None

    def rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, axis=-1)
        return ops.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x: Tensor, right_bound: int) -> Tensor:
        length = x.shape[-2]
        cos = self.cos_cached[None, None, right_bound-length:right_bound, :] # (1, 1, length, dim)
        sin = self.sin_cached[None, None, right_bound-length:right_bound, :] # (1, 1, length, dim)
        res = (x * cos) + (self.rotate_half(x) * sin) 
        return res

    def update_cos_sin_tables(self, new_seq_len: int):
        if new_seq_len <= self.seq_len_cached: return
        n = 1
        while n < new_seq_len:
            n = n * 2
        n=new_seq_len

        self.seq_len_cached = n
        t: Tensor = ops.arange(n).type_as(self.inv_freq) * self.distance_scale
        freqs = t.view(n, 1) @ self.inv_freq.view(1, -1) # (n, dim/2)
        emb = ops.cat((freqs, freqs), dim=-1) # (n, dim)
        
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def construct(self, q: Tensor, k: Tensor):
        """
        Args:
            q: A tensor of shape (batch, num_heads, len_q, dim_head).
            k: A tensor of shape (batch, num_heads, len_k, dim_head) which len_k >= len_q (caused by kv cache).
        Returns:
            rotated q and k.
        """
        len_k = k.shape[-2]
        self.update_cos_sin_tables(len_k)
        q = self.apply_rotary_pos_emb(q, len_k)
        k = self.apply_rotary_pos_emb(k, len_k)
        return q, k