from mindspore import ops
from mindspore import Tensor
from .global_var import config

def print_rank(*args, rank=0, **kwargs):
    if config['rank'] == rank:
        print(*args, **kwargs)

def synchronize():
    barrier = Tensor(1.)
    all_gather = ops.AllReduce()
    x = all_gather(barrier).item()