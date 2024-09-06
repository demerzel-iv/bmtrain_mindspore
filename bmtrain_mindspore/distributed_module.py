import mindspore as ms

from mindspore.nn import Cell
from time import time

from .distributed_parameter import DistributedParameter

class DistributedModule(Cell):
    def __getattr__(self, name):
        ret = super().__getattr__(name)
        if isinstance(ret, DistributedParameter): 
            return ret.gather()
        return ret