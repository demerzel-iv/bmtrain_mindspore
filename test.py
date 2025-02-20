import os
import argparse
import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn
from mindspore import mint
from mindnlp.core import ops
from mindspore import Tensor, Parameter
from mindspore.train.summary import SummaryRecord

from mindnlp.engine import Trainer

from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end

from bmtrain_mindspore.utils import Timer
from bmtrain_mindspore import DistributedParameter, DistributedModule
from bmtrain_mindspore.model_center.layer import Embedding, Linear
from bmtrain_mindspore.model_center import layer

OUTPUT_PATH = '/root/outputs'

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

    print('-----rank-{}, [{}]------'.format(bms.rank(), m.w))
    print('-----rank-{}, [{}]------'.format(bms.rank(), x))
    print('-----rank-{}, [{}]------'.format(bms.rank(), out))

class MyMatmul(DistributedModule):
    def __init__(self, a, b):
        super().__init__()
        arr = np.random.rand(a, b)
        self.w = DistributedParameter(Tensor(arr, dtype=ms.float32))

    def construct(self, x):
        return ops.matmul(x, self.w)

def test_train():
    bms.init_distributed()

    u, v = 5, 5

    np.random.seed(0)
    m = MyMatmul(u, v)
    ans = Tensor(np.random.rand(u, v), dtype=ms.float32)

    np.random.seed(bms.rank())

    optimizer = mint.optim.Adam(m.trainable_params(), lr=1e-3)
    print('--rank-{}-, [{}]'.format(bms.rank(), optimizer.parameters))

    def forward_fn(x, y):
        y_pred = m.construct(x)
        loss = ops.mean((y_pred - y) ** 2)
        return loss, y_pred

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for i in range(1000):
        x = Tensor(np.random.rand(1000//bms.world_size(), u), dtype=ms.float32)
        y = ops.matmul(x, ans)

        (loss, _), grads = grad_fn(x, y)
        grads = tuple(gd / bms.world_size() for gd in grads)
        optimizer.construct(grads)
        bms.print_rank(i, loss)

    bms.print_rank(m.w, '\n', ans)
    #bms.save(m, '/home/hanxu/lyq/data/test_model/test.ckpt')

def test_load():
    m = MyMatmul(33, 51)
    bms.load(m, '/home/hanxu/lyq/data/test_model/test.ckpt')
    bms.print_rank(m.w)

def test_load_time():
    from bmtrain_mindspore.model_center.model.llama import Llama, LlamaConfig
    from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer
    model_path = '/root/PLMs/llama2-7b-ms'
    model = Llama.from_pretrained(model_path)
    print('memory occupation after loading - {} - {} -'.format(ms.hal.memory_allocated(), bms.rank()))

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
    bms.print_rank(res)
    bms.save(model, '/root/output/test.ckpt')

def test_generate():
    from bmtrain_mindspore.model_center.model.llama import Llama, LlamaConfig
    from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer
    from tqdm import tqdm

    import bmtrain_mindspore as bms
    from bmtrain_mindspore.model_center.model.llama import Llama

    model_path = '/home/hanxu/lyq/data/Llama-2-7b-ms'

    model = Llama.from_pretrained(model_path)

    print('device rank {} - memory occupation - {}'.format(bms.rank(), ms.runtime.memory_allocated()))
    
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path)
    s = "The sun sets in the west, painting the sky with"
    x = tokenizer.encode(s, return_tensors='ms') # bs, n

    print('begin forwarding - rank {} - memory occupation - {}'.format(bms.rank(), ms.runtime.memory_allocated()))

    #_framework_profiler_step_start()
    with Timer('prefill'):
        hidden_states, past_key_values, logits = model.construct(x, use_cache=True, output_logits=True)
    #_framework_profiler_step_end()

    print('after prefilling - rank {} - memory occupation - {}'.format(bms.rank(), ms.runtime.memory_allocated()))

    next_tok = logits[0, -1].argmax().reshape(1,1)
    tok_list = (x, next_tok)
    print(next_tok)

    for i in range(20):
        with Timer('decoder {}'.format(i)):
            hidden_states, past_key_values, logits = model.construct(
                next_tok,
                use_cache=True,
                past_key_values=past_key_values,
                output_logits=True
            )
        next_tok = logits[0, -1].argmax().reshape(1,1)
        tok_list += (next_tok,)

        prob, ids = logits[0, -1].softmax(axis=-1).topk(k=6)
        
        res = ops.cat(tok_list, dim=-1)

    print(tokenizer.convert_ids_to_tokens(res[0]))

def main():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    bms.init_distributed(
        synchronous_execution=False,
        device_list=[4, 5, 6, 7],
    )
    
    #test_generate()
    test_train()
    #test_load()

if __name__ == '__main__':
    main()