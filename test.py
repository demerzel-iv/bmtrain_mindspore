import os
import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn, ops
from mindspore import Tensor, Parameter

from bmtrain_mindspore import DistributedParameter, DistributedModule
from bmtrain_mindspore.model_center.layer import Embedding, Linear

OUTPUT_PATH = '/root/output'

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

def test3():
    vs, dim = 7, 3
    arr = Tensor(np.arange(vs*dim).reshape(vs, dim)) * 1.

    input_embedding = Embedding(
        vocab_size=vs,
        embedding_size=dim,
        init=arr,
    )
    ms.save_checkpoint(input_embedding, os.path.join(OUTPUT_PATH, 'rank-{}.ckpt'.format(bms.rank())))

def test4():
    vs, dim = 7, 3
    input_embedding = Embedding(
        vocab_size=vs,
        embedding_size=dim,
    )

    param_dict = ms.load_checkpoint(os.path.join(OUTPUT_PATH, 'rank-{}.ckpt'.format(bms.rank())))
    print('\n-----rank-{}, [{}]------\n'.format(bms.rank(), param_dict['weight'].value()))

    bms.synchronize()

    bms.print_rank('weight = ', input_embedding.weight * 100)
    ms.load_param_into_net(input_embedding, param_dict)
    bms.print_rank('loaded weight = ', input_embedding.weight)

class Test(DistributedModule):
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)

        #arr1 = Tensor(np.arange(28).reshape(7, 4)) * 1.
        #arr2 = Tensor(np.arange(12).reshape(4, 3)) * 1.
        arr1 = Tensor(np.random.normal(size=(7, 4)), dtype=ms.float32)
        arr2 = Tensor(np.random.normal(size=(4, 3)), dtype=ms.float32)

        self.emb = Embedding(7, 4, init=arr1)
        self.output = DistributedParameter(arr2)

def test5():
    t = Test()
    save_path = os.path.join(OUTPUT_PATH,'test.ckpt')
    bms.save(t, save_path)

def test6():
    from bmtrain_mindspore.store import convert_model_to_param_dict
    t = Test()
    bms.print_rank(convert_model_to_param_dict(t))

    save_path = os.path.join(OUTPUT_PATH,'test.ckpt')
    bms.load(t, save_path)

    bms.print_rank(convert_model_to_param_dict(t))

def test_linear():
    np.random.seed(1926)
    n, m = 3, 5
    a = np.random.normal(size=(n,m))
    b = np.random.normal(size=(2, 3, 4, n))
    print(a, b)

    lin = Linear(n,m,init=Tensor(a, dtype=ms.float32))
    b = Tensor(b, dtype=ms.float32)

    def func(x):
        return ops.sum(lin.construct(x))
    grad_fn = ms.value_and_grad(func, weights=lin.trainable_params(), grad_position=None)

    out, grad = grad_fn(b)
    print('-----rank-{}, [{}], [{}]------'.format(bms.rank(), out, grad))

def main():
    try:
        bms.init_distributed()
    except Exception:
        print("init_distributed failed")

    ms.set_context(mode=ms.PYNATIVE_MODE)
    test_linear()

if __name__ == '__main__':
    main()