import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import ops

from bmtrain_mindspore.utils import Timer

def show_param_name():
    from bmtrain_mindspore.model_center.model.deepseek_v2 import DeepseekV2, DeepseekV2Config
    config = DeepseekV2Config(
        vocab_size=1000,
        num_layers=2,
        dim_model=128,
        dim_ff=256,
        num_heads=4,
        max_position_embeddings=128,
        rope_theta=1.0,
        qk_rope_head_dim=32,
        kv_lora_rank=4,
        v_head_dim=32,
        qk_nope_head_dim=32,
        activate_fn='relu',
        norm_eps=1e-5,
        dtype='fp32',
        rope_factor=1.0,
        rope_original_max_position_embeddings=128,
        rope_beta_fast=16,
        rope_beta_slow=1,
        rope_mscale=1.0,
        rope_mscale_all_dim=0,
    )
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
    model.construct(x)

def main():
    bms.init_distributed(
        synchronous_execution=False,
    )
    
    tmptest()
    #show_param_name()

if __name__ == '__main__':
    main()