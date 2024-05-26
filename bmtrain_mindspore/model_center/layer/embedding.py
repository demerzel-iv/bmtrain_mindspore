import mindspore as ms
import mindspore.nn as nn

from typing import Union
from mindspore.common.initializer import initializer
from mindspore import ops, Tensor

from ...distributed_module import DistributedModule
from ...distributed_parameter import DistributedParameter

class Embedding(DistributedModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        padding_idx: int = None,
        init: Union[str, Tensor] = 'norm',
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = embedding_size
        self.padding_idx = padding_idx

        self.weight = DistributedParameter(initializer(
            init=init,
            shape=(vocab_size, embedding_size)
        ))

    def construct(self, ids: Tensor) -> Tensor:
        """
        Args:
            ids: A tensor of shape (batch_size, seq_len).
        Returns:
            A tensor of shape (batch_size, seq_len, embedding_size) with values set to 0 where `ids` are equal to `padding_idx`.
        """
        embed = ops.gather(self.weight, ids, axis=0)
        if self.padding_idx != None:
            embed[ids == self.padding_idx] = 0.
        return embed