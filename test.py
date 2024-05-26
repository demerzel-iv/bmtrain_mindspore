import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn, ops
from mindspore import Tensor, Parameter

from bmtrain_mindspore import DistributedParameter, DistributedModule
from bmtrain_mindspore.model_center.layer import Embedding

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

def test2():
    vs, dim = 7, 3
    arr = Tensor(np.arange(vs*dim).reshape(vs, dim)) * 1.

    input_embedding = Embedding(
        vocab_size=vs,
        embedding_size=dim,
        padding_idx=4,
        init=arr,
    )

    def func(ids):
        embed = input_embedding.construct(ids) # 1, n, dim
        bms.print_rank(embed)
        return ops.sum(embed[0, :, -1] * embed[0, :, -2])

    #ids = Tensor([[6, 2, 4, 3, 3, 1, 6]])
    ids = Tensor([[(2*vs - bms.rank()) % vs,]])
    grad_fn = ms.grad(func, weights=input_embedding.trainable_params(), grad_position=None)

    out = grad_fn(ids)
    print('-----rank-{}, [{}]------'.format(bms.rank(), out))


def main():
    bms.init_distributed()
    ms.set_context(mode=ms.PYNATIVE_MODE)
    test2()

if __name__ == '__main__':
    main()