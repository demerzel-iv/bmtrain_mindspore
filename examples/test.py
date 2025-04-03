import os
import argparse
import mindspore
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import nn
from mindspore import mint
from mindspore import ops
from mindspore import Tensor, Parameter

from bmtrain_mindspore.utils import Timer
from bmtrain_mindspore import DistributedParameter, DistributedModule
from bmtrain_mindspore.model_center.layer import Embedding, Linear
from bmtrain_mindspore.model_center import layer

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

    with Timer('prefill'):
        hidden_states, past_key_values, logits = model.construct(x, use_cache=True, output_logits=True)

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
        res = ops.cat(tok_list, axis=-1)

    print(tokenizer.convert_ids_to_tokens(res[0]))

def tmptest():
    from bmtrain_mindspore.model_center.model.deepseek_v2 import DeepseekV2
    from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer

    path = '/root/thunlp/data/DeepSeek-V2-Lite-ms'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path)

    text = 'Generate training samples with reasoning prefix from the Countdown Game'
    x = tokenizer.encode(text, return_tensors='ms') # bs, n

    model = DeepseekV2.from_pretrained('/root/thunlp/data/DeepSeek-V2-Lite-ms')
    model.construct(x)

def main():
    bms.init_distributed(
        synchronous_execution=False,
    )
    
    #test_generate()
    tmptest()

if __name__ == '__main__':
    main()