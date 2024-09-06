import mindspore as ms

from typing import List, Tuple
from mindspore import ops, Tensor, nn

from .base_model import BaseModel
from .config import LlamaConfig
from ..layer import Embedding, RotaryEmbeddingESM, Encoder, Linear

class Llama(BaseModel):
    _CONFIG_TYPE = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
        )
        self.position_bias = RotaryEmbeddingESM(
            dim=config.dim_head,
        )
        self.encoder = Encoder(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            dim_ff=config.dim_ff,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            pos_bias_type='rotary',
            attn_scale=True,
            activate_fn=config.activate_fn,
            eps=config.eps,
            dropout_p=None,
            post_layer_norm=False,
            rms_layer_norm=True,
        )
        self.output_projection = Linear(
            dim_in=config.dim_model,
            dim_out=config.vocab_size,
            bias=False,
        )

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        inputs_embeds: Tensor = None,
        use_cache: bool = False,
        past_key_values: Tuple[Tuple[Tensor]] = None,
        output_logits: bool = False,
    ):
        if input_ids != None:
            batch_size, input_len = input_ids.shape
        elif inputs_embeds != None:
            batch_size, input_len, _ = inputs_embeds.shape
        else:
            assert False
        total_len = input_len
        if past_key_values != None:
            total_len += past_key_values[0][0].shape[-2]

        if attention_mask == None:
            attention_mask = ops.fill(shape=(batch_size,), value=total_len, type=ms.int32)
        
        if len(attention_mask.shape) == 3:
            attention_mask_2d = attention_mask
        else:
            if len(attention_mask.shape) == 1:
                attention_mask = ops.greater( # (batch_size, total_len)
                    attention_mask.view(-1, 1),
                    ops.arange(total_len).view(1, -1).repeat(batch_size, axis=0)
                 )
                # ops.bitwise_and do not support bool type, convert to int
                attention_mask = attention_mask.astype(ms.int8)
            assert len(attention_mask.shape) == 2
            attention_mask = attention_mask.view(batch_size, total_len, 1) & attention_mask.view(batch_size, 1, total_len) # (batch_size, total_len, total_len)
            directional_mask_2d: Tensor = ops.arange(total_len).view(-1, 1) >= ops.arange(total_len).view(1, -1) # (total_len, total_len)
            attention_mask_2d: Tensor = attention_mask & directional_mask_2d.view(1, total_len, total_len).astype(ms.int8)
        attention_mask_2d = attention_mask_2d[:, -input_len:, :].astype(ms.bool_)

        if inputs_embeds is None:
            inputs_embeds = self.input_embedding.construct(input_ids)

        hidden_states, current_key_values = self.encoder.construct(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask_2d,
            position_bias=self.position_bias,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        logits = self.output_projection(hidden_states) if output_logits else None
        if not use_cache:
            current_key_values = None
        return hidden_states, current_key_values, logits