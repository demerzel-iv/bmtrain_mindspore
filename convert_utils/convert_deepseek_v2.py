import os
import json
import shutil
import argparse

from mindnlp.core.serialization import safe_save_file, safe_load_file
from mindnlp.transformers.modeling_utils import load_state_dict

DTYPE_MAPPING = {
    'bfloat16' : 'bf16'
}

def copy_tokenizer(source_path: str, target_path: str):
    file_list = ['tokenizer_config.json', 'tokenizer.json']
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
        'vocab_size': old_config['vocab_size'],
        'num_layers': old_config['num_hidden_layers'],
        'dim_model': old_config['hidden_size'],
        'num_heads': old_config['num_attention_heads'],
        'max_position_embeddings': old_config['max_position_embeddings'],
        'rope_theta': old_config['rope_theta'],
        'qk_rope_head_dim': old_config['qk_rope_head_dim'],
        'kv_lora_rank': old_config['kv_lora_rank'],
        'v_head_dim': old_config['v_head_dim'],
        'qk_nope_head_dim': old_config['qk_nope_head_dim'],
        'activate_fn': 'gated_' + old_config['hidden_act'],
        'norm_eps': old_config['rms_norm_eps'],
        'dtype': DTYPE_MAPPING[old_config['torch_dtype']],
        'rope_factor': old_config['rope_scaling']['factor'],
        'rope_original_max_position_embeddings': old_config['rope_scaling']['original_max_position_embeddings'],
        'rope_beta_fast': old_config['rope_scaling']['beta_fast'],
        'rope_beta_slow': old_config['rope_scaling']['beta_slow'],
        'rope_mscale': old_config['rope_scaling']['mscale'],
        'rope_mscale_all_dim': old_config['rope_scaling']['mscale_all_dim'],
        'first_k_dense_replace': old_config['first_k_dense_replace'],
        'moe_layer_freq': old_config['moe_layer_freq'],
        'num_experts_per_tok': old_config['num_experts_per_tok'],
        'n_routed_experts': old_config['n_routed_experts'],
        'moe_intermediate_size': old_config['moe_intermediate_size'],
        'n_shared_experts': old_config['n_shared_experts'],
        'intermediate_size': old_config['intermediate_size'],
        'norm_topk_prob': old_config['norm_topk_prob'],
        'aux_loss_alpha': old_config['aux_loss_alpha'],
        'routed_scaling_factor': old_config['routed_scaling_factor'],
        'seq_aux': old_config['seq_aux'],
    }

    os.makedirs(target_path, exist_ok=True)
    with open(os.path.join(target_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    # convert weight checkpoints
    old_ckpt = {}
    for file in os.listdir(source_path):
        if file.endswith('.bin'):
            print(f'loading file : {file}')
            old_ckpt |= load_state_dict(os.path.join(source_path, file))
        elif file.endswith('.safetensors'):
            print(f'loading file : {file}')
            old_ckpt |= safe_load_file(os.path.join(source_path, file))

    new_ckpt = {}

    new_ckpt['input_embedding.weight'] = old_ckpt['model.embed_tokens.weight']
    new_ckpt['output_norm.weight'] = old_ckpt['model.norm.weight']
    new_ckpt['lm_head.weight'] = old_ckpt['lm_head.weight']

    for i in range(old_config['num_hidden_layers']):
        new_ckpt[f'layers.{i}.input_layernorm.weight'] = old_ckpt[f'model.layers.{i}.input_layernorm.weight']
        new_ckpt[f'layers.{i}.post_attention_layernorm.weight'] = old_ckpt[f'model.layers.{i}.post_attention_layernorm.weight']
        new_ckpt[f'layers.{i}.self_attn.q_proj.weight'] = old_ckpt[f'model.layers.{i}.self_attn.q_proj.weight']
        new_ckpt[f'layers.{i}.self_attn.kv_a_proj_with_mqa.weight'] = old_ckpt[f'model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight']
        new_ckpt[f'layers.{i}.self_attn.kv_a_layernorm.weight'] = old_ckpt[f'model.layers.{i}.self_attn.kv_a_layernorm.weight']
        new_ckpt[f'layers.{i}.self_attn.kv_b_proj.weight'] = old_ckpt[f'model.layers.{i}.self_attn.kv_b_proj.weight']
        new_ckpt[f'layers.{i}.self_attn.o_proj.weight'] = old_ckpt[f'model.layers.{i}.self_attn.o_proj.weight']

        if i < old_config['first_k_dense_replace']:
            new_ckpt[f'layers.{i}.mlp.w_in.w0.weight'] = old_ckpt[f'model.layers.{i}.mlp.gate_proj.weight']
            new_ckpt[f'layers.{i}.mlp.w_in.w1.weight'] = old_ckpt[f'model.layers.{i}.mlp.up_proj.weight']
            new_ckpt[f'layers.{i}.mlp.w_out.weight'] = old_ckpt[f'model.layers.{i}.mlp.down_proj.weight']
        else:
            for j in range(old_config['n_routed_experts']):
                new_ckpt[f'layers.{i}.mlp.experts.{j}.w_in.w0.weight'] = \
                    old_ckpt[f'model.layers.{i}.mlp.experts.{j}.gate_proj.weight']
                new_ckpt[f'layers.{i}.mlp.experts.{j}.w_in.w1.weight'] = \
                    old_ckpt[f'model.layers.{i}.mlp.experts.{j}.up_proj.weight']
                new_ckpt[f'layers.{i}.mlp.experts.{j}.w_out.weight'] = \
                    old_ckpt[f'model.layers.{i}.mlp.experts.{j}.down_proj.weight']

            new_ckpt[f'layers.{i}.mlp.gate.weight'] = \
                old_ckpt[f'model.layers.{i}.mlp.gate.weight']
            new_ckpt[f'layers.{i}.mlp.shared_experts.w_in.w0.weight'] = \
                old_ckpt[f'model.layers.{i}.mlp.shared_experts.gate_proj.weight']
            new_ckpt[f'layers.{i}.mlp.shared_experts.w_in.w1.weight'] = \
                old_ckpt[f'model.layers.{i}.mlp.shared_experts.up_proj.weight']
            new_ckpt[f'layers.{i}.mlp.shared_experts.w_out.weight'] = \
                old_ckpt[f'model.layers.{i}.mlp.shared_experts.down_proj.weight']

    os.makedirs(target_path, exist_ok=True)

    safe_save_file(new_ckpt, os.path.join(target_path, 'mindspore_model.safetensors'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', type=str, required=True, help='Path to checkpoint files of huggingface')
    parser.add_argument('--target-path', type=str, required=True, help='Path to save converted checkpoint files')
    args = parser.parse_args()
    convert_model(**vars(args))
    copy_tokenizer(**vars(args))
