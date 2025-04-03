import os
import mindspore as ms

from typing import Type, TypeVar, List, Tuple
from mindspore.nn import Cell

from mindspore import Tensor
from mindspore import ops

from .config import Config
from ...store import load

ModelType = TypeVar('ModelType', bound='BaseModel')

def _prepare_attention_mask(
    input_ids: Tensor = None,
    attention_mask: Tensor = None,
    inputs_embeds: Tensor = None,
    past_key_values: Tuple[Tuple[Tensor]] = None,
):
    """
    Prepare the attention mask for the model.
    Args:
        input_ids (Tensor of shape (batch_size, input_len)): The input ids.
        attention_mask (Tensor of shape (batch_size, input_len)): The attention mask.
        inputs_embeds (Tensor of shape (batch_size, input_len, dim_model)): The input embeddings.
        past_key_values (Tuple[Tuple[Tensor]]): The past key values.

    Returns:
        attention_mask (Tensor of shape (batch_size, input_len, total_len)): The attention mask.
    """
    if input_ids != None:
        batch_size, input_len = input_ids.shape
    elif inputs_embeds != None:
        batch_size, input_len, _ = inputs_embeds.shape
    else:
        assert False, "input_ids and inputs_embeds cannot be None at the same time"
    total_len = input_len
    if past_key_values != None:
        total_len += past_key_values[0][0].shape[-2]

    if attention_mask == None:
        attention_mask = ops.full(size=(batch_size,), fill_value=total_len, dtype=ms.int32)
    
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
    return attention_mask_2d


class BaseModel(Cell):
    _CONFIG_TYPE: Type[Config]

    @classmethod
    def from_pretrained(cls: Type[ModelType], pretrained_model_path: str, config=None, **kwargs) -> ModelType:
        if config == None:
            config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_path, **kwargs)
        model = cls(config)
        load(model, os.path.join(pretrained_model_path, 'mindspore_model.safetensors'), strict=True)
        return model