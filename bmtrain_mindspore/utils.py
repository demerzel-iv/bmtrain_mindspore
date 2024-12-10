import pickle
import numpy as np

from mindspore import ops as raw_ops
from mindnlp.core import ops
from mindspore import Tensor
from .global_var import config

def print_rank(*args, rank=0, **kwargs):
    if config['rank'] == rank:
        print(*args, **kwargs)

def synchronize():
    barrier = Tensor(1.)
    all_gather = raw_ops.AllReduce()
    x = all_gather(barrier).item()

def serialize_to_numpy(obj) -> np.ndarray:
    serialized_obj = pickle.dumps(obj)
    numpy_array = np.frombuffer(serialized_obj, dtype=np.uint8)
    return numpy_array

def deserialize_from_numpy(numpy_array: np.ndarray):
    serialized_obj = numpy_array.tobytes()
    obj = pickle.loads(serialized_obj)
    return obj