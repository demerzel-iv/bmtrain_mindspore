import gc
import numpy as np
import mindspore as ms

from typing import Dict, Tuple
from collections import OrderedDict
from mindspore import Tensor, Parameter, load_checkpoint, load_param_into_net
from mindspore import ops as raw_ops
from mindnlp.core import ops
from mindnlp.core.serialization import safe_load_file, safe_save_file
from mindspore.nn import Cell
from mindspore.train.serialization import _exec_save

from .distributed_parameter import DistributedParameter
from .utils import serialize_to_numpy, deserialize_from_numpy, Timer
from .global_var import rank

def save(model: Cell, save_path: str):
    param_dict = OrderedDict()
    for name, param in model.parameters_and_names():
        if isinstance(param, DistributedParameter):
            value = param.gather()
        else:
            assert isinstance(param, Parameter)
            value = param.value()
        param_dict[name] = value

    if rank() == 0:
        safe_save_file(param_dict, save_path)

def load(model: Cell, load_path: str, strict: bool = False):
    broad_cast = raw_ops.Broadcast(root_rank=0)
    if rank() == 0:
        with Timer('load file'):
            param_dict: Dict[str, Parameter] = safe_load_file(load_path)

        meta_data: Dict[str, Tuple] = {}
        for name, param in param_dict.items():
            meta_data[name] = (param.value().shape, param.value().dtype)
        # send meta data to other device
        meta_data_np = serialize_to_numpy(meta_data)
        meta_data_size = Tensor(meta_data_np.size, dtype=ms.int32)
        meta_data_size, = broad_cast((meta_data_size,))
        meta_data_ms = Tensor.from_numpy(meta_data_np.astype(np.int8))
        meta_data_ms, = broad_cast((meta_data_ms,))

        with Timer('boradcast'):
            for name in meta_data:
                value = param_dict[name].value().copy()
                value, = broad_cast((value,))
                param_dict[name] = value

    else:
        # recieve meta data from rank 0
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
         
    with Timer('load to net'):
        for key in param_dict:
            param_dict[key] = DistributedParameter(param_dict[key].value())
        load_param_into_net(model, param_dict, strict)

    # force collection
    del param_dict
    gc.collect()