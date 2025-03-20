import mindspore as ms
from mindspore import Tensor

#from mindspore import ops
#from mindnlp.core import ops
from mindspore.mint import ops

def forward_fn(u, mask):
    print('u is\n', u)
    print('mask is\n', mask)
    res: Tensor = ops.masked_fill(u, mask, 0)
    print('res is\n', res)

    def modify_grad(grad: Tensor):
        _grad = grad.copy()
        _grad[1, 1] = float('nan')
        print('grad of res is\n', _grad)
        return _grad
    res.register_hook(modify_grad)

    return ops.sum(res)

grad_fn = ms.value_and_grad(forward_fn, 0)

u = ops.rand(2, 2)
mask = Tensor([[0, 0], [0, 1]], dtype=ms.bool_)
res, grad = grad_fn(u, mask)
print('grad of u is')
print(grad)