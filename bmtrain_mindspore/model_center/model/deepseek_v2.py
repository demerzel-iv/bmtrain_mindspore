import mindspore as ms

from typing import List, Tuple
from mindspore import Tensor, nn
from mindspore import ops

from .base_model import BaseModel, _prepare_attention_mask
from .config import DeepseekConfig
from ..layer import Embedding, RotaryEmbeddingESM, Encoder, Linear

class DeepseekV2(BaseModel):
    _CONFIG_TYPE = DeepseekConfig

    def __init__(self, config: DeepseekConfig):
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

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        inputs_embeds: Tensor = None,
        use_cache: bool = False,
        past_key_values: Tuple[Tuple[Tensor]] = None,
        output_logits: bool = False,
    ):
        attention_mask_2d = _prepare_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )

        if inputs_embeds is None:
            inputs_embeds = self.input_embedding.construct(input_ids)

        print(inputs_embeds.value())

        #hidden_states, current_key_values = self.encoder.construct(
        #    hidden_states=inputs_embeds,
        #    attention_mask=attention_mask_2d,
        #    position_bias=self.position_bias,
        #    use_cache=use_cache,
        #    past_key_values=past_key_values
        #)
        #logits = self.output_projection(hidden_states) if output_logits else None
        #if not use_cache:
        #    current_key_values = None
        #return hidden_states, current_key_values, logits