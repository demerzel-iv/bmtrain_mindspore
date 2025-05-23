import numpy as np
import mindspore as ms

from typing import Tuple
from mindspore import Tensor, nn
from mindspore import ops
from mindspore.nn import Cell

from .attention import AttentionBlock
from .feedforward import FFNBlock
from .layer_norm import LayerNorm

class TransformerBlock(Cell):
    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        pos_bias_type: str = 'none',
        attn_scale: bool = False,
        activate_fn: str = 'gated_silu',
        norm_eps: float = 1e-5,
        dropout_p: float = None,
        post_layer_norm: bool = False,
        rms_layer_norm: bool = False,
        layer_id: int = None,
        dtype = ms.float32,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.self_att = AttentionBlock(
            dim_model=dim_model,
            dim_head=dim_head,
            num_heads=num_heads,
            pos_bias_type=pos_bias_type,
            attn_scale=attn_scale,
            dropout_p=dropout_p,
            norm_eps=norm_eps,
            post_layer_norm=post_layer_norm,
            rms_layer_norm=rms_layer_norm,
            dtype=dtype,
        )
        self.ffn = FFNBlock(
            dim_model=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            bias=False,
            dropout_p=dropout_p,
            norm_eps=norm_eps,
            post_layer_norm=post_layer_norm,
            rms_layer_norm=rms_layer_norm,
            dtype=dtype,
        )
    
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
            - A tensor of shape (batch, seq_len, dim_model).
            - current_key_value if use_cache is True else None.
        """
        hidden_states, current_key_value = self.self_att.construct(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        hidden_states = self.ffn.construct(hidden_states)
        return hidden_states, current_key_value


class Encoder(Cell):
    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        pos_bias_type: str = 'none',
        attn_scale: bool = False,
        activate_fn: str = 'gated_silu',
        norm_eps: float = 1e-5,
        dropout_p: float = None,
        post_layer_norm: bool = False,
        rms_layer_norm: bool = False,
        dtype = ms.float32,
    ):
        super().__init__()
        self.layers = nn.CellList([
            TransformerBlock(
                dim_model=dim_model,
                dim_ff=dim_ff,
                num_heads=num_heads,
                dim_head=dim_head,
                pos_bias_type=pos_bias_type,
                attn_scale=attn_scale,
                activate_fn=activate_fn,
                norm_eps=norm_eps,
                dropout_p=dropout_p,
                post_layer_norm=post_layer_norm,
                rms_layer_norm=rms_layer_norm,
                layer_id=i,
                dtype=dtype,
            ) for i in range(num_layers)
        ])
        self.output_layer_norm = LayerNorm(
            dim_norm=dim_model,
            eps=norm_eps,
            rms_layer_norm=rms_layer_norm,
            dtype=dtype,
        )

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_bias = None,
        use_cache: bool = False,
        past_key_values: Tuple[Tuple[Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor]]]:
        """
        Args:
            hidden_states: A tensor of shape (batch, seq_len, dim_model).
            attention_mask: A tensor of shape (batch, seq_len, seq_len+pkv_len).
            position_bias: A callable object for rotary embedding.
        Returns:
            - A tensor of shape (batch, seq_len, dim_model).
            - A tulple consists of current_key_value if use_cache is True else None.
        """
        current_key_values = ()
        for i, module in enumerate(self.layers):
            module: TransformerBlock 
            #hidden_states, current_key_value = module.construct(
            #    hidden_states=hidden_states,
            #    attention_mask=attention_mask,
            #    position_bias=position_bias,
            #    use_cache=use_cache,
            #    past_key_value=past_key_values[i] if past_key_values != None else None,
            #)
            hidden_states, current_key_value = ms.recompute(
                module,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                use_cache=use_cache,
                past_key_value=past_key_values[i] if past_key_values != None else None,
            )
            current_key_values += (current_key_value,)
        hidden_states = self.output_layer_norm.construct(hidden_states)
        return hidden_states, current_key_values
