import os
import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn, ops
from mindspore import Tensor, Parameter

from bmtrain_mindspore import DistributedParameter, DistributedModule
from bmtrain_mindspore.model_center.layer import Embedding, Linear
from bmtrain_mindspore.model_center import layer

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

def test_layer_norm():
    from bmtrain_mindspore.model_center.layer import LayerNorm
    n, m = 3, 5
    ln = LayerNorm(m, 1e-12, True, init=Tensor(np.ones(5), dtype=ms.float32))
    a = np.random.normal(size=(n,m))
    a = Tensor(a, dtype=ms.float32)
    bms.print_rank(ln.construct(a))

def test_attention():
    from bmtrain_mindspore.model_center.layer import Attention

    dim=24
    dim_head=3
    att = Attention(
        dim_in=dim,
        dim_head=dim_head,
        num_heads=dim // dim_head,
    )
    lq = 12
    lk = 7

    q = Tensor(np.random.normal(size=(1, lq, dim)), dtype=ms.float32)
    kv = Tensor(np.random.normal(size=(1, lk, dim)), dtype=ms.float32)
    mask = Tensor(np.random.randint(0, 2, size=(1, lq, lk)), dtype=ms.bool_)

    res = att.construct(q, kv, mask)
    bms.print_rank(res)

def test_ffn():
    from bmtrain_mindspore.model_center.layer import FeedForward

    dim=13
    dim_ff = 129
    ffn = FeedForward(
        dim_in=dim,
        dim_ff=dim_ff,
        activate_fn='gated_silu'
    )
    n = 7

    x = Tensor(np.random.normal(size=(3, n, dim)), dtype=ms.float32)

    bms.print_rank(x)
    res = ffn.construct(x)
    bms.print_rank(res)

def test_att_block():
    from bmtrain_mindspore.model_center.layer import AttentionBlock

    dim=24
    dim_head=3
    att = AttentionBlock(
        dim_model=dim,
        dim_head=dim_head,
        num_heads=dim // dim_head,
    )
    n = 12

    hs = Tensor(np.random.normal(size=(1, n, dim)), dtype=ms.float32)
    mask = Tensor(np.random.randint(0, 2, size=(1, n, n)), dtype=ms.bool_)

    res, _ = att.construct(hs, mask) 
    bms.print_rank(res)

def test_transformer():
    dim=24
    dim_ff=55
    dim_head=3
    tsm = layer.Encoder(
        num_layers=12,
        dim_model=dim,
        dim_ff=dim_ff,
        num_heads=dim // dim_head,
        dim_head=dim_head,
    )

    n = 12
    hs = Tensor(np.random.normal(size=(1, n, dim)), dtype=ms.float32)
    mask = Tensor(np.random.randint(0, 2, size=(1, n, n)), dtype=ms.bool_)
    res = tsm.construct(
        hidden_states=hs,
        attention_mask=mask
    )
    bms.print_rank(res)

def test_rotate():
    from mindnlp.transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
    dim=1024
    rp = layer.RotaryEmbeddingESM(dim=dim,)
    n = 1231
    q = Tensor(np.random.normal(size=(1, 1, n, dim)), dtype=ms.float32)
    k = Tensor(np.random.normal(size=(1, 1, n, dim)), dtype=ms.float32)
    rq, rk = rp.construct(q, k)

    rrp = LlamaRotaryEmbedding(dim, max_position_embeddings=1)
    cos, sin = rrp(k, seq_len=n)
    query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, ops.arange(n), unsqueeze_dim=0)

    res = rq - query_states
    #res = rq - sin
    thr=1e-9
    res[res < thr] = 0
    bms.print_rank(res)
    bms.print_rank(ops.sum(res>thr))
    bms.print_rank(res.size)
    bms.print_rank(ops.sum(res>thr)*1.0 / res.size)

def test_llama():
    from bmtrain_mindspore.model_center.model.llama import Llama, LlamaConfig

    config = LlamaConfig(
        vocab_size=44,
        num_layers=12,
        dim_model=128,
        dim_head=16,
        dim_ff=512,
        num_heads=8,
    )
    model = Llama(config)

    n = 123
    input_ids = ops.randint(0, config.vocab_size, size=(2, n,))
    attention_mask = Tensor([45, 89])
    res, _, _ = model.construct(input_ids=input_ids, attention_mask=attention_mask)
    print(res)

def main():
    try:
        bms.init_distributed()
    except Exception:
        print("init_distributed failed")

    ms.set_context(mode=ms.PYNATIVE_MODE)
    test_llama()

if __name__ == '__main__':
    main()