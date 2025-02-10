import mindspore as ms
import mindspore.nn as nn

from typing import Union
from mindspore.common.initializer import initializer
from mindspore import Tensor
from mindnlp.core import ops
from mindnlp.core.nn import functional as F

from ...distributed_module import DistributedModule
from ...distributed_parameter import DistributedParameter

class Embedding(DistributedModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        padding_idx: int = None,
        init = 'normal',
        dtype = ms.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = embedding_size
        self.padding_idx = padding_idx
        # initialize the weight
        init_tensor = initializer(init=init, shape=(vocab_size, embedding_size), dtype=dtype)
        self.weight = DistributedParameter(init_tensor)

    def construct(self, ids: Tensor) -> Tensor:
        """
        Args:
            ids: A tensor of shape (batch_size, seq_len).
        Returns:
            A tensor of shape (batch_size, seq_len, embedding_size) with values set to 0 where `ids` are equal to `padding_idx`.
        """
        embed = F.embedding(ids, self.weight, self.padding_idx)
        return embed