import math
import numpy as np
import mindspore as ms

from typing import Callable, Tuple
from mindspore import Tensor, nn
from mindspore import ops
from mindspore.nn import Cell

from .linear import Linear
from .layer_norm import LayerNorm

class Attention(Cell):
    def __init__(
        self,
        dim_in: int, 
        dim_head: int,
        num_heads: int, 
        dim_out: int = None,
        pos_bias_type: str = "none",
        attn_scale: bool = False,
        dropout_p: float = None,
        num_heads_kv: int = -1,
        dtype = ms.float32,
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
        self.pos_bias_type = pos_bias_type

        self.project_q = Linear(
            dim_in=dim_in,
            dim_out=num_heads * dim_head,
            bias=False,
            dtype=dtype,
        )
        self.project_k = Linear(
            dim_in=dim_in,
            dim_out=num_heads_kv * dim_head,
            bias=False,
            dtype=dtype,
        )
        self.project_v = Linear(
            dim_in=dim_in,
            dim_out=num_heads_kv * dim_head,
            bias=False,
            dtype=dtype,
        )
        self.attention_out = Linear(
            dim_in=num_heads * dim_head,
            dim_out=dim_out,
            bias=False,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p != None else None

    def construct(
        self,
        query: Tensor,
        key_value: Tensor,
        attention_mask: Tensor,
        position_bias: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        past_key_value: Tuple[Tensor] = None,
    ):
        """
        Args:
            query: A tensor of shape (batch, len_q, dim_in).
            key_value: A tensor of shape (batch, len_k, dim_in).
            attention_mask: A tensor of shape (batch, len_q, len_k).
            position_bias: A callable object for rotary embedding.
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

        h_q = h_q.contiguous()
        h_k = h_k.contiguous()
        h_v = h_v.contiguous()

        # concat past_key_value and update len_k
        if past_key_value is not None:
            h_k = ops.cat([past_key_value[0], h_k], axis=-2)
            h_v = ops.cat([past_key_value[1], h_v], axis=-2)
            len_k = h_k.shape[-2]
        # for future calculation
        current_key_value = (h_k, h_v) if use_cache else None

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

        if self.pos_bias_type == "rotary":
            h_q, h_k = position_bias(h_q, h_k)

        # calculate attention scores
        score: Tensor = ops.matmul(h_q, h_k.permute(0, 1, 3, 2)) # (batch, num_heads, len_q, len_k)
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        #TODO if self.pos_bias_type == "relative":
        #    if position_bias is not None:
        #        score = score + position_bias

        def remove_nan(grad: Tensor):
            return ops.masked_fill(grad, grad.isnan(), 0)
        score.register_hook(remove_nan)

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
        output = self.attention_out.construct(output)                               # (batch, len_q, dim_out)

        return output, current_key_value


class AttentionBlock(Cell):
    def __init__(
        self,
        dim_model: int, 
        dim_head: int,
        num_heads: int, 
        dim_out: int = None,
        pos_bias_type: str = "none",
        attn_scale: bool = False,
        dropout_p: float = None,
        num_heads_kv: int = -1,
        norm_eps: float = 1e-5,
        post_layer_norm: bool = False,
        rms_layer_norm: bool = False,
        dtype = ms.float32,
    ):
        super().__init__()
        self.layernorm = LayerNorm(
            dim_norm=dim_model,
            eps=norm_eps,
            rms_layer_norm=rms_layer_norm,
            dtype=dtype,
        )
        self.attention = Attention(
            dim_in=dim_model,
            dim_head=dim_head,
            num_heads=num_heads,
            dim_out=dim_out,
            pos_bias_type=pos_bias_type,
            attn_scale=attn_scale,
            dropout_p=dropout_p,
            num_heads_kv=num_heads_kv,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p != None else None
        self.post_layer_norm = post_layer_norm

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_bias = None,
        use_cache: bool = False,
        past_key_value = None,
    ):
        """
        Args:
            hidden_states: A tensor of shape (batch, seq_len, dim_model).
            attention_mask: A tensor of shape (batch, seq_len, seq_len+pkv_len).
            position_bias: A callable object for rotary embedding.
        Returns:
            A tensor of shape (batch, seq_len, dim_model).
        """
        x = self.layernorm.construct(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x, current_key_value = self.attention.construct(
            x, x,
            attention_mask=attention_mask,
            position_bias=position_bias,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        if self.dropout != None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states, current_key_value