import numpy as np
import mindspore as ms

from typing import Union
from mindspore.common.initializer import initializer
from mindspore import ops, Tensor, nn

from ...distributed_module import DistributedModule
from ...distributed_parameter import DistributedParameter

class LayerNorm(DistributedModule):
    def __init__(
        self,
        dim_norm: int,
        eps: float = 1e-5,
        rms_layer_norm: bool = False,
        init = 1,
    ):
        super().__init__()
        self.dim_norm = dim_norm
        self.eps = eps
        self.rms_layer_norm = rms_layer_norm
        # operator
        self.layer_norm_ops = ops.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=eps)

        # initialize the weight
        init_tensor = initializer(init=init, shape=(dim_norm,))
        self.weight = DistributedParameter(init_tensor)
        self.bias = DistributedParameter(
            Tensor(np.zeros(shape=(dim_norm,)), dtype=ms.float32)
        ) if not rms_layer_norm else None

    def construct(self, x: Tensor):
        """
        Args:
            ids: A tensor of shape (..., dim_norm).
        Returns:
            A tensor of shape (..., dim_norm).
        """
        if not self.rms_layer_norm:
            x, _, _ = self.layer_norm_ops(x, self.weight, self.bias)
            return x
        else:
            # RMS layer norm
            old_dtype = x.dtype
            variance = x.to(ms.float32).pow(2).mean(axis=-1, keep_dims=True)
            x = x * ops.rsqrt(variance + self.eps)
            return x.to(old_dtype) * self.weight
            