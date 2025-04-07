import mindspore as ms

from typing import Tuple
from mindspore import Tensor

from .base_model import BaseModel, _prepare_attention_mask
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
            dtype=config.dtype,
        )
        self.position_bias = RotaryEmbeddingESM(
            dim=config.dim_head,
            dtype=config.dtype,
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
            norm_eps=config.norm_eps,
            dropout_p=None,
            post_layer_norm=False,
            rms_layer_norm=True,
            dtype=config.dtype,
        )
        self.output_projection = Linear(
            dim_in=config.dim_model,
            dim_out=config.vocab_size,
            bias=False,
            dtype=config.dtype,
        )

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        input_embeds: Tensor = None,
        use_cache: bool = False,
        past_key_values: Tuple[Tuple[Tensor]] = None,
        output_logits: bool = False,
    ):
        attention_mask_2d = _prepare_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            past_key_values=past_key_values
        )

        if input_embeds is None:
            input_embeds = self.input_embedding.construct(input_ids)

        hidden_states, current_key_values = self.encoder.construct(
            hidden_states=input_embeds,
            attention_mask=attention_mask_2d,
            position_bias=self.position_bias,
            use_cache=use_cache,
            past_key_values=past_key_values
        )
        logits = self.output_projection(hidden_states) if output_logits else None
        if not use_cache:
            current_key_values = None
        return hidden_states, current_key_values, logits