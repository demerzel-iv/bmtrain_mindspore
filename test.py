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
        arr = np.random.rand(a, b)
        self.w = DistributedParameter(Tensor(arr, dtype=ms.float32))
        #self.w = Parameter(Tensor(arr, dtype=ms.float32))

    def construct(self, x):
        return (self.w * x).sum()

def ouo():
    bms.init_distributed()
    m = Test(3, 4)

    #for k in m.parameters_and_names():
    #    bms.print_rank(k)

    x = Tensor(bms.rank(), dtype=ms.float32)
    grad_fn = ms.grad(m.construct, weights=m.trainable_params(), grad_position=0)
    out = grad_fn(x)

    bms.print_rank(len(m.trainable_params()), len(out))

    print('\n-----\nrank-{}, [{}]\n------\n'.format(bms.rank(), out))

def test2():
    print("=== mindspore.grad 多个output ===")
    x = mindspore.Tensor(1.0)
    y = mindspore.Tensor(2.0)
    def net(x, y):
        return x**2+y, x
    out = mindspore.grad(net, grad_position=(0,1))(x, y)
    print("out", out)


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ouo()

if __name__ == '__main__':
    main()