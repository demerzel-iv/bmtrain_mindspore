import os
import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn, ops
from mindspore import Tensor, Parameter

from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end

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

    print('-----rank-{}, [{}]------'.format(bms.rank(), out))

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

def test_pretrained_llama():
    from bmtrain_mindspore.model_center.model.llama import Llama, LlamaConfig
    from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer

    model_path = '/root/outputs/llama2-7b-bms'
    model = Llama.from_pretrained(model_path)

    print('memory occupation after loading - {} - {} -'.format(ms.hal.memory_allocated(), bms.rank()))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path)
    s = "The sun sets in the west, painting the sky with"
    x = tokenizer.encode(s, return_tensors='ms') # bs, n

    hidden_states, past_key_values, logits = model.construct(input_ids=x, attention_mask=None, output_logits=True)

    print('memory occupation after forwarding - {} - {} -'.format(ms.hal.memory_allocated(), bms.rank()))
    #next_tok = logits[0, -1].argmax().reshape(1,1)

def test_generate():
    from bmtrain_mindspore.model_center.model.llama import Llama, LlamaConfig
    from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer
    from tqdm import tqdm

    import bmtrain_mindspore as bms
    from bmtrain_mindspore.model_center.model.llama import Llama
    model_path = '/root/outputs/llama2-7b-bms'
    model = Llama.from_pretrained(model_path)
    print('device rank {} - memory occupation - {}'.format(bms.rank(), ms.hal.memory_allocated()))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path)
    s = "The sun sets in the west, painting the sky with"
    x = tokenizer.encode(s, return_tensors='ms') # bs, n

    #hidden_states, past_key_values, logits = model.construct(input_ids=x, attention_mask=None, output_logits=True)
    print('begin forwarding - {} - {} -'.format(ms.hal.memory_allocated(), bms.rank()))
    _framework_profiler_step_start()
    hidden_states, past_key_values, logits = model.construct(x, use_cache=True, output_logits=True)
    _framework_profiler_step_end()
    next_tok = logits[0, -1].argmax().reshape(1,1)
    tok_list = (x, next_tok)
    print(next_tok)

    return

    with ms.SummaryRecord('/root/test_llama/test_sum/', network=model) as summary_record:
        for i in (range(20)):
            hidden_states, past_key_values, logits = model.construct(next_tok, use_cache=True, past_key_values=past_key_values)
            next_tok = logits[0, -1].argmax().reshape(1,1)
            tok_list += (next_tok,)

            prob, ids = logits[0, -1].softmax(axis=-1).topk(k=6)
            if bms.rank() == 0:
                summary_record.add_value('tensor', 'probability', prob)
                summary_record.add_value('tensor', 'ids', ids)
                summary_record.record(i)
            
            res = ops.cat(tok_list, axis=-1)
            print('i=', i)
    print(tokenizer.convert_ids_to_tokens(res[0]))

def main():
    try:
        bms.init_distributed()
    except Exception:
        print("init_distributed failed")

    ms.set_context(mode=ms.PYNATIVE_MODE)
    #test_pretrained_llama()
    test_generate()

if __name__ == '__main__':
    main()