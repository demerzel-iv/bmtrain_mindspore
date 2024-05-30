import math
import numpy as np
import mindspore as ms

from typing import Union
from mindspore.common.initializer import initializer
from mindspore import ops, Tensor, nn

from ...distributed_module import DistributedModule
from ...distributed_parameter import DistributedParameter
from .linear import Linear

class DenseACT(DistributedModule):
    def __init__(
        self,
        dim_in : int,
        dim_ff : int,
        activate_fn : str = "silu",
        bias = False,
    ):
        super().__init__()
        self.gated = 'gated_' in activate_fn
        activate_fn = activate_fn.replace('gated_', '')

        self.w0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            bias=bias,
        )
        self.w1 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            bias=bias,
        ) if self.gated else None

        if activate_fn == "relu":
            self.act = ops.relu
        elif activate_fn == "gelu":
            self.act = ops.gelu
        elif activate_fn == "silu":
            self.act = ops.silu
        else:
            raise ValueError("Unsupported activation function: {}".format(activate_fn))

    def construct(self, x: Tensor):
        """
        Args:
            x: A tensor of shape (..., dim_in).
        Returns:
            A tensor of shape (..., dim_ff).
        """
        y = self.act(self.w0.construct(x))
        if self.gated:
            z = self.w1.construct(x)
            y = y * z
        return y

class FeedForward(DistributedModule):
    def __init__(
        self,
        dim_in : int, 
        dim_ff : int,
        dim_out : int = None,
        activate_fn: str = "gated_gelu",
        bias: bool = False,
        length_scale : bool = False,
        dropout_p: float = None,
    ):
        super().__init__()
        dim_out = dim_out if dim_out != None else dim_in

        self.w_in = DenseACT(
            dim_in=dim_in,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            bias=bias,
        )
        self.w_out = Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            bias = bias,
        )
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p != None else None

    def construct(self, x: Tensor):
        """
        Args:
            x: A tensor of shape (..., dim_in).
        Returns:
            A tensor of shape (..., dim_out).
        """
        x = self.w_in(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.w_out(x)
        return x