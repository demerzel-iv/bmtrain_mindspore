import numpy as np
import mindspore as ms

from typing import Union
from mindspore.common.initializer import initializer
from mindspore import Tensor, nn
from mindspore import ops
from mindnlp.core.nn import functional as F

from ...distributed_module import DistributedModule
from ...distributed_parameter import DistributedParameter

class LayerNorm(DistributedModule):
    def __init__(
        self,
        dim_norm: int,
        eps: float = 1e-5,
        rms_layer_norm: bool = False,
        dtype = ms.float32,
    ):
        super().__init__()
        self.dim_norm = dim_norm
        self.eps = eps
        self.rms_layer_norm = rms_layer_norm

        # initialize the weight
        self.weight = DistributedParameter(
            Tensor(np.ones(shape=(dim_norm,)), dtype=dtype)
        )
        self.bias = DistributedParameter(
            Tensor(np.zeros(shape=(dim_norm,)), dtype=dtype)
        ) if not rms_layer_norm else None

    def construct(self, x: Tensor):
        """
        Args:
            ids: A tensor of shape (..., dim_norm).
        Returns:
            A tensor of shape (..., dim_norm).
        """
        if not self.rms_layer_norm:
            #x, _, _ = self.layer_norm_ops(x, self.weight, self.bias)
            x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
            return x
        else:
            # RMS layer norm
            old_dtype = x.dtype
            variance = x.to(ms.float32).pow(2).mean(axis=-1, keep_dims=True)
            x = x * ops.rsqrt(variance + self.eps)
            return x.to(old_dtype) * self.weight
            