import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import ops

from bmtrain_mindspore.utils import Timer

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
    
    tmptest()

if __name__ == '__main__':
    main()