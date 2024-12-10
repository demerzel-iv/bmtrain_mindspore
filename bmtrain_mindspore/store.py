import gc
import numpy as np
import mindspore as ms

from typing import Dict, Tuple
from collections import OrderedDict
from mindspore import Tensor, Parameter, load_checkpoint, load_param_into_net
from mindspore import ops as raw_ops
from mindnlp.core import ops
from mindspore.nn import Cell
from mindspore.train.serialization import _exec_save

from .distributed_parameter import DistributedParameter
from .utils import serialize_to_numpy, deserialize_from_numpy
from .global_var import rank

def convert_model_to_param_dict(model: Cell) -> Dict[str, Tuple[Tuple[int], str, Tensor]]:
    """
    Returns:
        Dict[str, Tuple[Tuple[int], str, Tensor]]: A dictionary where:

            - The keys are the names of the parameters.
            - The values are tuples containing:
                - shape (Tuple[int]): The shape of the parameter tensor.
                - dtype (str): The data type of the parameter tensor.
                - data (Tensor): The parameter tensor itself.
    """
    param_dict = OrderedDict()
    for name, param in model.parameters_and_names():
        if isinstance(param, DistributedParameter):
            value = param.gather()
        else:
            assert isinstance(param, Parameter)
            value = param.value()
        param_dict[name] = (
            value.shape,
            str(value.dtype),
            value
        )
    return param_dict

def save(model: Cell, save_path: str):
    param_dict = convert_model_to_param_dict(model)
    _exec_save(save_path, param_dict)

def load(model: Cell, load_path: str, strict: bool = False):
    broad_cast = raw_ops.Broadcast(root_rank=0)
    if rank() == 0:
        from time import time
        print('begin loading - rank {}'.format(rank()))
        start_time = time()
        param_dict: Dict[str, Parameter] = load_checkpoint(load_path)
        print('finish loading, time used: {:.1f}s - rank {}'.format(time() - start_time, rank()))

        meta_data: Dict[str, Tuple] = {}
        for name, param in param_dict.items():
            meta_data[name] = (param.value().shape, param.value().dtype)
        meta_data_np = serialize_to_numpy(meta_data)
        meta_data_size = Tensor(meta_data_np.size, dtype=ms.int32)
        meta_data_size, = broad_cast((meta_data_size,))
        meta_data_ms = Tensor.from_numpy(meta_data_np.astype(np.int8))
        meta_data_ms, = broad_cast((meta_data_ms,))

        for name in meta_data:
            value = param_dict[name].value().copy()
            value, = broad_cast((value,))
            param_dict[name] = value

    else:
        print('wait for loading - rank {}'.format(rank()))
        meta_data_size = Tensor(0, dtype=ms.int32)
        meta_data_size, = broad_cast((meta_data_size,))
        meta_data_ms: Tensor = ops.zeros(int(meta_data_size), dtype=ms.int8)
        meta_data_ms, = broad_cast((meta_data_ms,))
        meta_data = deserialize_from_numpy(meta_data_ms.numpy())

        param_dict = {}
        for name in meta_data:
            shape, dtype = meta_data[name]
            value = ops.zeros(shape, dtype=dtype)
            value, = broad_cast((value,))
            param_dict[name] = value
         
    for key in param_dict:
        param_dict[key] = DistributedParameter(param_dict[key].value())
    load_param_into_net(model, param_dict, strict)

    # force collection
    del param_dict
    gc.collect()