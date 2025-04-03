import numpy as np
import mindspore as ms

from typing import Union
from mindspore.common.initializer import initializer
from mindspore import Tensor, nn
from mindspore import ops

from ...distributed_module import DistributedModule
from ...distributed_parameter import DistributedParameter

class Linear(DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = False,
        init = 'normal',
        dtype = ms.float32,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        # initialize the weight
        init_tensor = initializer(init=init, shape=(dim_in, dim_out), dtype=dtype)
        self.weight = DistributedParameter(init_tensor)
        self.bias = DistributedParameter(
            Tensor(np.zeros(shape=(dim_out,)), dtype=dtype)
        ) if bias else None

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            ids: A tensor of shape (..., dim_in).
        Returns:
            A tensor of shape (..., dim_out).
        """
        x = ops.matmul(x, self.weight)
        if self.bias != None:
            x = x + self.bias
        return x