import math
import numpy as np
import mindspore as ms

from typing import Union
from mindspore.common.initializer import initializer
from mindspore import ops, Tensor, nn

from ...distributed_module import DistributedModule
from ...distributed_parameter import DistributedParameter
from .linear import Linear

class Attention(DistributedModule):
    def __init__(
        self,
        dim_in: int, 
        dim_head: int,
        num_heads: int, 
        dim_out: int = None,
        bias: bool = False,
        mask_value: float = float("-inf"),
        pos_bias_type: str = "none",
        attn_scale: bool = False,
        dropout_p: float = None,
        num_heads_kv: int = -1,
    ):
        super().__init__()
        dim_out = dim_out if dim_out != None else dim_in
        num_heads_kv = num_heads_kv if num_heads_kv != -1 else num_heads 
        # save arguments
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.attn_scale = attn_scale
        assert num_heads % num_heads_kv == 0
        self.num_head_groups = num_heads // num_heads_kv

        self.project_q = Linear(
            dim_in=dim_in,
            dim_out=num_heads * dim_head,
            bias=False,
        )
        self.project_k = Linear(
            dim_in=dim_in,
            dim_out=num_heads_kv * dim_head,
            bias=False,
        )
        self.project_v = Linear(
            dim_in=dim_in,
            dim_out=num_heads_kv * dim_head,
            bias=False,
        )
        self.attention_out = Linear(
            dim_in=num_heads * dim_head,
            dim_out=dim_out,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p != None else None

    def construct(
        self,
        query: Tensor,
        key_value: Tensor,
        attention_mask: Tensor,
        position_bias = None,
        use_cache: bool = False,
        past_key_value = None,
    ):
        """
        Args:
            query: A tensor of shape (batch, len_q, dim_in).
            key_value: A tensor of shape (batch, len_k, dim_in).
            attention_mask: A tensor of shape (batch, len_q, len_k).
        Returns:
            A tensor of shape (batch, len_q, dim_out).
        """
        batch_size, len_q, _ = query.shape
        len_k = key_value.shape[1]

        h_q = self.project_q.construct(query)             # (batch, len_q, num_heads * dim_head)
        h_k = self.project_k.construct(key_value)         # (batch, len_k, num_heads_kv * dim_head)
        h_v = self.project_v.construct(key_value)         # (batch, len_k, num_heads_kv * dim_head)

        h_q: Tensor = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)    # (batch, num_heads, len_q, dim_head)
        h_k: Tensor = h_k.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3) # (batch, num_heads_kv, len_k, dim_head)
        h_v: Tensor = h_v.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3) # (batch, num_heads_kv, len_k, dim_head)

        # either num_heads_kv == num_heads or num_heads_kv == 1, otherwise replicate h_k and h_v to make num_heads_kv == num_heads 
        if self.num_head_groups != 1 and self.num_heads_kv != 1:
            h_k = h_k[:, :, None, :, :].broadcast_to(
                batch_size, self.num_heads_kv, self.num_head_groups, len_k, self.dim_head).reshape(
                batch_size, self.num_heads, len_k, self.dim_head
            ) # (batch, num_heads, len_k, dim_head)
            h_v = h_v[:, :, None, :, :].broadcast_to(
                batch_size, self.num_heads_kv, self.num_head_groups, len_k, self.dim_head).reshape(
                batch_size, self.num_heads, len_k, self.dim_head
            ) # (batch, num_heads, len_k, dim_head)

        h_q = h_q.contiguous()
        h_k = h_k.contiguous()
        h_v = h_v.contiguous()

        # concat past_key_value and update len_k
        if past_key_value is not None:
            h_k = ops.cat([past_key_value[0], h_k], dim=-2)
            h_v = ops.cat([past_key_value[1], h_v], dim=-2)
            len_k = h_k.size(-2)
        # for future calculation
        current_key_value = (h_k, h_v) if use_cache else None

        #TODO if self.pos_bias_type == "rotary":
        #    h_q, h_k = position_bias(h_q, h_k)

        # calculate attention scores
        score: Tensor = ops.matmul(h_q, h_k.permute(0, 1, 3, 2)) # (batch, num_heads, len_q, len_k)
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        #TODO if self.pos_bias_type == "relative":
        #    if position_bias is not None:
        #        score = score + position_bias
        
        score = ops.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            Tensor(float('-inf'), dtype=score.dtype)
        )
        score = ops.softmax(score, axis=-1)
        # avoid nan in softmax
        score = ops.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            Tensor(0, dtype=score.dtype)
        )
        if self.dropout is not None:
            score = self.dropout(score)

        # calculate output based on attetion scores
        output: Tensor = ops.matmul(score, h_v)                                     # (batch, num_heads, len_q, dim_head)
        output = output.permute(0, 2, 1, 3)                                         # (batch, len_q, num_heads, dim_head)
        output = output.reshape(batch_size, len_q, self.num_heads * self.dim_head)  # (batch, len_q, num_heads * dim_head)
        output = self.attention_out.construct(output)                                         # (batch, len_q, dim_out)

        if use_cache:
            return output, current_key_value
        else:
            return output
