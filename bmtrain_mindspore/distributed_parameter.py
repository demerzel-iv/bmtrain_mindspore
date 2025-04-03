import math
import mindspore as ms

from typing import Tuple
from mindspore import ops
from mindspore import Tensor, Parameter

from .utils import print_rank
from .global_var import config

PAD_VALUE = 0

class DistributedParameter(Parameter):
    _original_size: int
    _original_shape : Tuple
    all_gather_ops: ops.AllGather

    def __new__(cls, x: Tensor, *args, **kwargs):
        num_tot = x.numel()
        num_per_device = int(math.ceil(num_tot / config['world_size']))
        rank = config['rank']
        original_shape = x.shape

        x = x.reshape(-1)
        left_bound = rank * num_per_device
        right_bound = (rank + 1) * num_per_device
        x = x[left_bound: right_bound]#.copy()
        # ops.AllGather requires tensors on each device to have the same size
        if x.numel() < num_per_device:
            num_to_pad = num_per_device - x.numel()
            x = ops.cat((x, Tensor([PAD_VALUE] * num_to_pad, dtype=x.dtype)))

        # 不能删掉，但是不知道为什么
        x = x.numpy()

        obj: DistributedParameter = super().__new__(cls, x, *args, **kwargs)

        obj._original_size = num_tot
        obj._original_shape = original_shape
        obj.all_gather_ops = ops.AllGather()

        return obj

    def gather(self) -> Tensor:
        x: Tensor = self.all_gather_ops(self)
        x = ms.mint.narrow(x, 0, 0, self._original_size)
        x = x.reshape(self._original_shape)
        return x

    def __str__(self):
        return 'Distributed ' + super().__str__()
