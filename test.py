import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn
from mindspore import Tensor, Parameter
from bmtrain_mindspore import DistributedParameter, DistributedModule

class Test(DistributedModule):
    def __init__(self, a, b):
        super().__init__()
        arr = np.ones((a, b))
        self.w = DistributedParameter(Tensor(arr, dtype=ms.float32))

    def construct(self, x):
        return (self.w * x).sum()

def test():
    bms.init_distributed()
    m = Test(3, 4)

    x = Tensor(bms.rank(), dtype=ms.float32)
    grad_fn = ms.grad(m.construct, weights=m.trainable_params(), grad_position=0)
    out = grad_fn(x)

    print('-----rank-{}, [{}]------'.format(bms.rank(), out))

def main():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    test()

if __name__ == '__main__':
    main()