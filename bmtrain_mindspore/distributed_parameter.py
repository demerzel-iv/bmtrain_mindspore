from mindspore import ops
from mindspore import Tensor, Parameter
from typing import Tuple

from .utils import print_rank
from .global_var import config

import math
import mindspore as ms

PAD_VALUE = 0

class DistributedParameter(Parameter):
    _original_size: int
    _original_shape : Tuple

    def __new__(cls, x: Tensor, *args, **kwargs):
        num_tot = x.numel()
        num_per_device = int(math.ceil(num_tot / config['world_size']))
        rank = config['rank']
        original_shape = x.shape

        x = x.reshape(-1)
        left_bound = rank * num_per_device
        right_bound = (rank + 1) * num_per_device
        x = x[left_bound: right_bound]
        # ops.AllGather requires tensors on each device to have the same size
        if x.numel() < num_per_device:
            num_to_pad = num_per_device - x.numel()
            x = ops.cat((x, Tensor([PAD_VALUE] * num_to_pad, dtype=x.dtype)))

        obj: DistributedParameter = super().__new__(cls, x, *args, **kwargs)

        obj._original_size = num_tot
        obj._original_shape = original_shape

        return obj

    def gather(self) -> Tensor:
        all_gather = ops.AllGather()
        x = self.value()
        x: Tensor = all_gather(x)
        x = x[:self._original_size]
        x = x.reshape(self._original_shape)
        return x