import os
import json
import shutil
import argparse

from collections import OrderedDict
from mindnlp.core.serialization import safe_save_file
from mindnlp.transformers.modeling_utils import load_state_dict
from mindspore.train.serialization import _exec_save

def copy_tokenizer(source_path: str, target_path: str):
    file_list = ['special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json']
    for file in file_list:
        file_path = os.path.join(source_path, file)
        if not os.path.exists(file_path):
            print('file `{}` not found!'.format(file))
        else:
            shutil.copy(file_path, os.path.join(target_path, file))

def convert_model(source_path: str, target_path: str):
    # convert the config file
    config_path = os.path.join(source_path, 'config.json')
    assert os.path.exists(config_path)
    with open(config_path) as f:
        old_config = json.load(f)
    config = {
        'vocab_size' : old_config['vocab_size'],
        'num_layers' : old_config['num_hidden_layers'],
        'dim_model' : old_config['hidden_size'],
        'dim_head' : old_config['hidden_size'] // old_config['num_attention_heads'],
        'dim_ff' : old_config['intermediate_size'],
        'num_heads' : old_config['num_attention_heads'],
        'norm_eps' : old_config['rms_norm_eps'],
    }

    os.makedirs(target_path, exist_ok=True)
    with open(os.path.join(target_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    # convert weight checkpoints
    old_ckpt = {}
    for file in os.listdir(source_path):
        if file.endswith('.bin'):
            old_ckpt |= load_state_dict(os.path.join(source_path, file))

    new_ckpt = {}

    new_ckpt['input_embedding.weight'] = old_ckpt['model.embed_tokens.weight']
    new_ckpt['encoder.output_layer_norm.weight'] = old_ckpt['model.norm.weight']
    new_ckpt['output_projection.weight'] = old_ckpt['lm_head.weight']
    for i in range(old_config['num_hidden_layers']):
        new_ckpt[f'encoder.layers.{i}.ffn.layernorm.weight']                    = old_ckpt[f'model.layers.{i}.post_attention_layernorm.weight']
        new_ckpt[f'encoder.layers.{i}.ffn.ffn.w_in.w0.weight']                  = old_ckpt[f'model.layers.{i}.mlp.gate_proj.weight']
        new_ckpt[f'encoder.layers.{i}.ffn.ffn.w_in.w1.weight']                  = old_ckpt[f'model.layers.{i}.mlp.up_proj.weight']
        new_ckpt[f'encoder.layers.{i}.ffn.ffn.w_out.weight']                    = old_ckpt[f'model.layers.{i}.mlp.down_proj.weight']
        new_ckpt[f'encoder.layers.{i}.self_att.layernorm.weight']               = old_ckpt[f'model.layers.{i}.input_layernorm.weight']
        new_ckpt[f'encoder.layers.{i}.self_att.attention.project_k.weight']     = old_ckpt[f'model.layers.{i}.self_attn.k_proj.weight']
        new_ckpt[f'encoder.layers.{i}.self_att.attention.project_q.weight']     = old_ckpt[f'model.layers.{i}.self_attn.q_proj.weight']
        new_ckpt[f'encoder.layers.{i}.self_att.attention.project_v.weight']     = old_ckpt[f'model.layers.{i}.self_attn.v_proj.weight']
        new_ckpt[f'encoder.layers.{i}.self_att.attention.attention_out.weight'] = old_ckpt[f'model.layers.{i}.self_attn.o_proj.weight']

    os.makedirs(target_path, exist_ok=True)
    safe_save_file(new_ckpt, os.path.join(target_path, 'mindspore_model.safetensors'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', type=str, required=True, help='Path to checkpoint files of huggingface')
    parser.add_argument('--target-path', type=str, required=True, help='Path to save converted checkpoint files')
    args = parser.parse_args()
    convert_model(**vars(args))
    copy_tokenizer(**vars(args))
