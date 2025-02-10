import os
import argparse
import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn
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

class MM(nn.Cell):
    def __init__(self, u):
        super().__init__()
        self.w = Parameter(u)

def more_test():
    #ms.set_context(mode=ms.PYNATIVE_MODE, save_graphs=3, save_graphs_path="ir")
    ag = ms.ops.AllGather()
    rank = bms.rank()

    #def fwd(x, y):
    #    return (ag(x) * y).sum()
    #x = Tensor(np.ones((1,6)).astype(np.float32))
    #y = Tensor(bms.rank(), dtype=ms.float32)
    #grad_fn = ms.value_and_grad(fwd, grad_position=(0, 1))
    #out = grad_fn(x, y)

    def initm():
        param = Tensor(np.ones((4, 6)), dtype=ms.float32)
        m = MM(param.reshape(-1)[rank*6: (rank+1)*6])
        return m
    m = initm()

    def fwd(y):
        return (ag(m.w) * y).sum()
    y = Tensor(rank, dtype=ms.float32)
    grad_fn = ms.value_and_grad(fwd, grad_position=0, weights=m.trainable_params())
    out = grad_fn(y)

    #m = Test(4, 6)
    #def fwd():
    #    return m.w.sum()

    #grad_fn = ms.value_and_grad(fwd, weights=m.trainable_params(), grad_position=None)

    print('-----rank-{}, [{}]------'.format(bms.rank(), out))

def test():
    bms.init_distributed()

    m = Test(3, 4)

    x = Tensor(bms.rank(), dtype=ms.float32)
    grad_fn = ms.grad(m.construct, weights=m.trainable_params(), grad_position=0)
    out = grad_fn(x)

    print('-----rank-{}, [{}]------'.format(bms.rank(), m.w))
    print('-----rank-{}, [{}]------'.format(bms.rank(), x))
    print('-----rank-{}, [{}]------'.format(bms.rank(), out))

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

    #model_path = '/root/data/Llama-2-7b-ms'
    model_path = '/root/lyq/data/Llama-2-7b-ms'

    model = Llama.from_pretrained(model_path)
    #print('device rank {} - memory occupation - {}'.format(bms.rank(), ms.hal.memory_allocated()))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path)
    s = "The sun sets in the west, painting the sky with"
    x = tokenizer.encode(s, return_tensors='ms') # bs, n

    #print('begin forwarding - {} - {} -'.format(ms.hal.memory_allocated(), bms.rank()))

    #_framework_profiler_step_start()
    with Timer('prefill'):
        hidden_states, past_key_values, logits = model.construct(x, use_cache=True, output_logits=True)
    #_framework_profiler_step_end()

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

def test_train():
    pass

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--sync', action='store_true')
    #sync = parser.parse_args().sync
    sync = True

    bms.init_distributed()

    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_context(pynative_synchronize=sync)
    #ms.set_context(pynative_synchronize=False)
    
    test_generate()
    #test()

if __name__ == '__main__':
    main()