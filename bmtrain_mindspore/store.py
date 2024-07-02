from typing import Dict, Tuple
from collections import OrderedDict
from mindspore import Tensor, Parameter, load_checkpoint, load_param_into_net
from mindspore.nn import Cell
from mindspore.train.serialization import _exec_save

from .distributed_parameter import DistributedParameter
from .utils import print_rank, synchronize

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
    from .global_var import rank
    print('begin - {}'.format(rank()))
    param_dict: Dict[str, Parameter] = load_checkpoint(load_path)
    print('finload - {}'.format(rank()))
    for key in param_dict:
        print('distribute - {} - {}'.format(key, rank()))
        param_dict[key] = DistributedParameter(param_dict[key].value())
    print('loadintonet - {}'.format(rank()))
    load_param_into_net(model, param_dict, strict)
    print('end - {}'.format(rank()))