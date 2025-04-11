import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import ops

from bmtrain_mindspore.utils import Timer

def show_param_name():
    from bmtrain_mindspore.model_center.model.deepseek_v2 import DeepseekV2, DeepseekV2Config
    config = DeepseekV2Config.from_json_file('/mnt/paas/kubernetes/kubelet/thunlp_data/DeepSeek-V2-Lite-ms/config.json')
    model = DeepseekV2(config)
    for name, param in model.parameters_and_names():
        param: bms.DistributedParameter
        print(name, param._original_shape)

def tmptest():
    from bmtrain_mindspore.model_center.model.deepseek_v2 import DeepseekV2
    from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer

    path = '/root/thunlp/data/DeepSeek-V2-Lite-ms'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path)

    text = 'Generate training samples with reasoning prefix from the Countdown Game'
    x = tokenizer.encode(text, return_tensors='ms') # bs, n

    model = DeepseekV2.from_pretrained('/root/thunlp/data/DeepSeek-V2-Lite-ms')
    model.set_train()
    model.construct(x)

def generate():
    from bmtrain_mindspore.model_center.model.deepseek_v2 import DeepseekV2
    from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer

    path = '/root/thunlp/data/DeepSeek-V2-Lite-ms'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token

    model = DeepseekV2.from_pretrained(path)
    model.set_train(False)

    print('mem_loc', ms.runtime.max_memory_allocated() / (2**30))

    input_text = 'Tsinghua University is'
    input_ids = tokenizer.encode(input_text, return_tensors='ms')
    tok_list = (input_ids,)
    key_values = None
    stream = ''

    temperature = 0.4

    for _ in range(100):
        _, key_values, logits = model.construct(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=key_values,
        )
        logits = logits[:, -1, :]  # 取最后一个token的logits
        print(logits.shape)
        prob = ops.softmax(logits.reshape(-1).astype(ms.float32) / temperature).numpy()
        next_tok_id = np.random.choice(logits.shape[-1], p=prob)

        input_ids = ms.Tensor([[next_tok_id]])
        tok_list += (input_ids,)

        res = ops.cat(tok_list, axis=-1)
        new_stream = tokenizer.decode(res[0].asnumpy().tolist())
        #print(new_stream[len(stream):], end='', flush=True)
        print(new_stream, flush=True)
        stream = new_stream

        if next_tok_id == tokenizer.eos_token_id:
            break

    print('')

def main():
    bms.init_distributed(
        synchronous_execution=False,
    )
    
    #show_param_name()
    #tmptest()
    generate()

if __name__ == '__main__':
    main()