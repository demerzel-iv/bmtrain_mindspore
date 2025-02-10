import numpy as np
import mindspore as ms

from mindspore import Tensor, nn
from mindnlp.core import ops
from mindspore.nn import Cell
from mindnlp.core.nn import functional as F

from .linear import Linear
from .layer_norm import LayerNorm

class DenseACT(Cell):
    def __init__(
        self,
        dim_in : int,
        dim_ff : int,
        activate_fn : str = "silu",
        bias = False,
        dtype = ms.float32,
    ):
        super().__init__()
        self.gated = 'gated_' in activate_fn
        activate_fn = activate_fn.replace('gated_', '')

        self.w0 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            bias=bias,
            dtype=dtype,
        )
        self.w1 = Linear(
            dim_in=dim_in,
            dim_out=dim_ff,
            bias=bias,
            dtype=dtype,
        ) if self.gated else None

        if activate_fn == "relu":
            self.act = F.relu
        elif activate_fn == "gelu":
            self.act = F.gelu
        elif activate_fn == "silu":
            self.act = F.silu
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

class FeedForward(Cell):
    def __init__(
        self,
        dim_in : int, 
        dim_ff : int,
        dim_out : int = None,
        activate_fn: str = "gated_gelu",
        bias: bool = False,
        dropout_p: float = None,
        dtype = ms.float32,
    ):
        super().__init__()
        dim_out = dim_out if dim_out != None else dim_in

        self.w_in = DenseACT(
            dim_in=dim_in,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            bias=bias,
            dtype=dtype,
        )
        self.w_out = Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            bias = bias,
            dtype=dtype,
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


class FFNBlock(Cell):
    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        activate_fn: str = 'gated_gelu',
        bias: bool = False,
        dropout_p: float = None,
        norm_eps: float = 1e-5,
        post_layer_norm: bool = False,
        rms_layer_norm: bool = False,
        dtype = ms.float32,
    ):
        super().__init__()
        self.layernorm = LayerNorm(
            dim_norm=dim_model,
            eps=norm_eps,
            rms_layer_norm=rms_layer_norm,
            dtype=dtype,
        )
        self.ffn = FeedForward(
            dim_in=dim_model,
            dim_ff=dim_ff,
            activate_fn=activate_fn,
            bias=bias,
            dropout_p=dropout_p,
            dtype=dtype,
        )
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p != None else None
        self.post_layer_norm = post_layer_norm

    def construct(self, hidden_states: Tensor):
        """
        Args:
            hidden_states: A tensor of shape (..., dim_model).
        Returns:
            A tensor of shape (..., dim_model).
        """
        x = self.layernorm(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.ffn(x)
        if self.dropout != None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states