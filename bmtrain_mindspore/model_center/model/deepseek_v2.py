"""
Copied from https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite with modifications
"""

import math
import numpy as np
import mindspore as ms

from typing import Tuple, Any
from mindspore import Tensor, nn, ops
from mindspore.mint.nn import functional as F
from bmtrain_mindspore import DistributedParameter, DistributedModule

from .base_model import BaseModel, _prepare_attention_mask
from .config import DeepseekV2Config
from ..layer import Embedding, RotaryEmbedding, Linear, LayerNorm, FeedForward

class AuxiliaryLoss:
    @staticmethod
    def grad_one(grad: Tensor):
        return ops.ones_like(grad)

    @staticmethod
    def apply(x: Tensor, loss: Tensor):
        if loss is None:
            return x
        assert loss.size == 1
        loss.register_hook(AuxiliaryLoss.grad_one)
        return x + 0 * loss

class YarnRotaryEmbedding(RotaryEmbedding):
    # Inverse dim formula to find dim based on number of rotations
    @staticmethod
    def find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
    ):
        return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    # Find dim range bounds based on rotations
    @staticmethod
    def find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
    ):
        low = math.floor(
            YarnRotaryEmbedding.find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            YarnRotaryEmbedding.find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    @staticmethod
    def get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def linear_ramp_mask(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001  # Prevent singularity

        linear_func = (np.arange(dim, dtype=np.float32) - min_val) / (max_val - min_val)
        ramp_func = np.clip(linear_func, 0, 1)
        return Tensor(ramp_func, dtype=ms.float32)

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale_factor: float = 1.0,
        dtype: ms.dtype = ms.float32,
        max_position_embeddings: int = 2048,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: int = 0,
    ):
        super().__init__(dim, base, scale_factor, dtype)

        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

    def update_cos_sin_tables(self, new_seq_len: int):
        if new_seq_len <= self.seq_len_cached:
            return

        seq_len = 1
        while seq_len < new_seq_len:
            seq_len = seq_len * 2
        seq_len = new_seq_len
        self.seq_len_cached = seq_len

        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (ops.arange(0, dim, 2, dtype=ms.float32) / dim)
        )
        freq_inter = 1.0 / (
            self.scale_factor
            * self.base
            ** (ops.arange(0, dim, 2, dtype=ms.float32) / dim)
        )

        low, high = self.find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self.linear_ramp_mask(low, high, dim // 2)
        self.inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = ops.arange(seq_len, dtype=ms.float32)
        freqs = t.view(seq_len, 1) @ self.inv_freq.view(1, -1) # (seq_len, dim/2)

        _mscale = float(
            self.get_mscale(self.scale_factor, self.mscale)
            / self.get_mscale(self.scale_factor, self.mscale_all_dim)
        )

        emb: Tensor = ops.cat((freqs, freqs), axis=-1) # (seq_len, dim)

        self.cos_cached = (emb.cos() * _mscale).astype(self.dtype)
        self.sin_cached = (emb.sin() * _mscale).astype(self.dtype)
        

class DeepseekV2Attention(nn.Cell):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.dim_model = config.dim_model
        self.num_heads = config.num_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.q_proj = Linear(
            self.dim_model,
            self.num_heads * self.q_head_dim,
            bias=False,
            dtype=config.dtype,
        )
        self.kv_a_proj_with_mqa = Linear(
            self.dim_model,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
            dtype=config.dtype,
        )
        self.kv_a_layernorm = LayerNorm(
            config.kv_lora_rank,
            eps=config.norm_eps,
            rms_layer_norm=True,
            dtype=config.dtype,
        )
        self.kv_b_proj = Linear(
            config.kv_lora_rank,
            self.num_heads * (
                self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
            dtype=config.dtype,
        )
        self.o_proj = Linear(
            self.num_heads * self.v_head_dim,
            self.dim_model,
            bias=False,
            dtype=config.dtype,
        )

        self.rotary_emb = YarnRotaryEmbedding(
            dim=self.qk_rope_head_dim,
            base=self.rope_theta,
            scale_factor=config.rope_factor,
            dtype=config.dtype,
            max_position_embeddings=self.max_position_embeddings,
            original_max_position_embeddings=config.original_max_position_embeddings,
            beta_fast=config.rope_beta_fast,
            beta_slow=config.rope_beta_slow,
            mscale=config.rope_mscale,
            mscale_all_dim=config.rope_mscale_all_dim,
        )

        self.softmax_scale = self.q_head_dim ** (-0.5)
        
        scaling_factor = self.config.rope_factor
        mscale_all_dim = self.config.rope_mscale_all_dim
        if mscale_all_dim > 0:
            mscale = YarnRotaryEmbedding.get_mscale(scaling_factor, mscale_all_dim)
            self.softmax_scale *= mscale * mscale

    @staticmethod
    def permute_pe(x: Tensor):
        prefix_shape = x.shape[:-1]
        dim = x.shape[-1]
        rank = len(prefix_shape)
        x = ops.reshape(x, prefix_shape + (dim // 2, 2))
        x = ops.transpose(x, tuple(range(rank)) + (rank + 1, rank))
        x = ops.reshape(x, prefix_shape + (dim,))
        return x

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        position_ids: Tensor = None,
        past_key_value: Tuple[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        """
        Args:
            hidden_states: A tensor of shape (batch, q_len, dim_model).
            attention_mask: A tensor of shape (batch, q_len, kv_seq_len).
            position_ids: A tensor of shape (batch, q_len).
            past_key_value: Tuple of tensors for past key and value states.
            use_cache: Boolean indicating whether to use cache.
        """
        bsz, q_len, _ = hidden_states.shape  # (bsz, q_len, dim_model)

        q = self.q_proj(hidden_states)  # (bsz, q_len, num_heads * q_head_dim)
        q = ops.transpose(ops.reshape(q, (bsz, q_len, self.num_heads, self.q_head_dim)), (0, 2, 1, 3))  # (bsz, num_heads, q_len, q_head_dim)

        q_nope, q_pe = ops.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # (bsz, q_len, kv_lora_rank + qk_rope_head_dim)
        compressed_kv, k_pe = ops.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
        k_pe = ops.transpose(ops.reshape(k_pe, (bsz, q_len, 1, self.qk_rope_head_dim)), (0, 2, 1, 3))  # (bsz, 1, q_len, qk_rope_head_dim)
        kv = ops.transpose(
            ops.reshape(
                self.kv_b_proj(self.kv_a_layernorm(compressed_kv)),  # (bsz, q_len, num_heads * (qk_nope_head_dim + v_head_dim))
                (bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
            ),
            (0, 2, 1, 3),
        )  # (bsz, num_heads, q_len, qk_nope_head_dim + v_head_dim)
        k_nope, value_states = ops.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)  # (bsz, num_heads, q_len, qk_nope_head_dim), (bsz, num_heads, q_len, v_head_dim)

        # Apply RoPE
        len_k = q_len
        if past_key_value is not None:
            len_k += past_key_value[0].shape[-2]

        q_pe = self.permute_pe(q_pe)  # (bsz, num_heads, q_len, qk_rope_head_dim)
        k_pe = self.permute_pe(k_pe)  # (bsz, num_heads, q_len, qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb.construct(q_pe, k_pe, len_k=len_k)

        query_states = ops.zeros((bsz, self.num_heads, q_len, self.q_head_dim), q_pe.dtype)  # (bsz, num_heads, q_len, q_head_dim)
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope  # Fill q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe  # Fill q_pe

        key_states = ops.zeros((bsz, self.num_heads, q_len, self.q_head_dim), k_pe.dtype)  # (bsz, num_heads, q_len, q_head_dim)
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope  # Fill k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe  # Fill k_pe

        # Handle past_key_value
        if past_key_value is not None:
            key_states = ops.cat((past_key_value[0], key_states), axis=-2)  # (bsz, num_heads, kv_seq_len, q_head_dim)
            value_states = ops.cat((past_key_value[1], value_states), axis=-2)  # (bsz, num_heads, kv_seq_len, v_head_dim)
        current_key_value = (key_states, value_states) if use_cache else None

        # Compute attention scores
        attn_weights = ops.matmul(query_states, ops.transpose(key_states, (0, 1, 3, 2))) * self.softmax_scale  # (bsz, num_heads, q_len, kv_seq_len)

        if attention_mask is not None:
            attn_weights = ops.masked_fill(
                attn_weights,
                attention_mask.view(bsz, 1, q_len, key_states.shape[-2])==False,
                Tensor(-1e9, dtype=self.config.dtype)
            )

        attn_weights = ops.softmax(
            attn_weights.to(ms.float32), axis=-1
        ).to(self.config.dtype)  # (bsz, num_heads, q_len, kv_seq_len)

        attn_output = ops.matmul(attn_weights, value_states)  # (bsz, num_heads, q_len, v_head_dim)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))  # (bsz, q_len, num_heads, v_head_dim)
        attn_output = ops.reshape(attn_output, (bsz, q_len, -1))  # (bsz, q_len, num_heads * v_head_dim)

        attn_output = self.o_proj(attn_output)  # (bsz, q_len, dim_model)

        return attn_output, current_key_value


class MoEGate(DistributedModule):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.weight = DistributedParameter(
            Tensor(
                np.empty((config.n_routed_experts, config.dim_model)),
                config.dtype
            ), 
        )

    def construct(self, hidden_states: Tensor):
        bsz, seq_len, h = hidden_states.shape

        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.to(ms.float32), self.weight.to(ms.float32))
        scores = ops.softmax(logits, axis=-1)

        # greedly select topk experts, ops.topk only support float
        topk_weight, topk_idx = ops.topk(scores, self.config.num_experts_per_tok, sorted=True)

        if self.config.num_experts_per_tok > 1 and self.config.norm_topk_prob:
            denominator = ops.sum(topk_weight, axis=-1, keep_dims=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.config.routed_scaling_factor

        aux_loss = None
        # calculate aux loss if in training
        if self.training and self.config.aux_loss_alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.config.num_experts_per_tok
            topk_idx_for_aux_loss = topk_idx.view(bsz, seq_len * aux_topk)
            if self.config.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = ops.zeros((bsz, self.config.n_routed_experts), ms.float32)
                ce = ms.mint.scatter_add(
                    ce, dim=-1,
                    index = topk_idx_for_aux_loss,
                    src = ops.ones_like(topk_idx_for_aux_loss, dtype=ce.dtype),
                )
                ce = ce.div(seq_len * aux_topk / self.config.n_routed_experts)
                aux_loss = (
                    ce * ops.mean(scores_for_seq_aux, axis=1)
                ).sum(axis=1).mean() * self.config.aux_loss_alpha
            else:
                mask_ce = ops.one_hot(topk_idx_for_aux_loss.view(-1), self.config.n_routed_experts, 1.0, 0.0)
                ce = ops.mean(mask_ce, axis=0)
                Pi = ops.mean(scores_for_aux, axis=0)
                fi = ce * self.config.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.config.aux_loss_alpha

        return topk_idx, topk_weight.to(self.config.dtype), aux_loss


class DeepseekV2MoE(nn.Cell):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.CellList([
            FeedForward(
                config.dim_model,
                config.moe_intermediate_size,
                activate_fn=config.activate_fn,
                dtype=config.dtype,
            )
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = FeedForward(
                config.dim_model,
                intermediate_size,
                activate_fn=config.activate_fn,
                dtype=config.dtype,
            )

        self.aux_loss = 0.

    def construct(self, hidden_states: Tensor) -> Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        hidden_states = ops.repeat_interleave(
            hidden_states, self.num_experts_per_tok, axis=0
        )
        y = ops.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            mask = (flat_topk_idx == i)
            y[mask] = expert(hidden_states[mask])
            
        y = (y.view(*topk_weight.shape, -1) * topk_weight.expand_dims(-1)).sum(axis=1)
        y = y.to(hidden_states.dtype).view(*orig_shape)
        
        # apply aux loss to the backward graph 
        AuxiliaryLoss.apply(y, aux_loss)
        self.aux_loss = float(aux_loss) if aux_loss is not None else 0.

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class DeepseekV2DecoderLayer(nn.Cell):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm = LayerNorm(
            config.dim_model,
            eps=config.norm_eps,
            rms_layer_norm=True,
            dtype=config.dtype,
        )
        self.self_attn = DeepseekV2Attention(config, layer_idx)
        self.post_attention_layernorm = LayerNorm(
            config.dim_model,
            eps=config.norm_eps,
            rms_layer_norm=True,
            dtype=config.dtype,
        )
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(config)
        else:
            self.mlp = FeedForward(
                config.dim_model,
                config.intermediate_size,
                activate_fn=config.activate_fn,
                dtype=config.dtype,
            )

        self.aux_loss = 0.

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        position_ids: Tensor = None,
        past_key_value: Tuple[Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor]]:

        # Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, current_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + residual  # Residual connection

        # MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        self.aux_loss = self.mlp.aux_loss if hasattr(self.mlp, 'aux_loss') else 0.

        return hidden_states, current_key_value


class DeepseekV2(BaseModel):
    _CONFIG_TYPE = DeepseekV2Config

    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config

        self.input_embedding = Embedding(
            vocab_size=config.vocab_size,
            embedding_size=config.dim_model,
            dtype=config.dtype,
        )
        self.layers = nn.CellList([
            DeepseekV2DecoderLayer(
                config, layer_idx=i
            ) for i in range(config.num_layers)
        ])
        self.output_norm = LayerNorm(
            config.dim_model,
            eps=config.norm_eps,
            rms_layer_norm=True,
            dtype=config.dtype,
        )
        self.lm_head = Linear(
            config.dim_model,
            config.vocab_size,
            bias=False,
            dtype=config.dtype,
        )

        self.aux_loss = 0.

    def construct(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        input_embeds: Tensor = None,
        use_cache: bool = False,
        past_key_values: Tuple[Tuple[Tensor]] = None,
        output_logits: bool = False,
        enable_checkpointing: bool = True,
    ) -> Tuple[Tensor, Any, Tensor]:
        """
        Args:
            input_ids: A tensor of shape (batch, seq_len).
            attention_mask: A tensor of shape (batch, seq_len, kv_seq_len).
            input_embeds: A tensor of shape (batch, seq_len, dim_model).
            use_cache: Boolean indicating whether to use cache.
            past_key_values: Tuple of tensors for past key and value states.
            output_logits: Boolean indicating whether to output logits.

        Returns:
            hidden_states: A tensor of shape (batch, seq_len, dim_model).
            current_key_values: Tuple of tensors for current key and value states.
            logits: A tensor of shape (batch, seq_len, vocab_size).
        """
        attention_mask_2d = _prepare_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            past_key_values=past_key_values,
        )

        if input_embeds is None:
            input_embeds = self.input_embedding.construct(input_ids)

        hidden_states = input_embeds
        current_key_values = ()
        aux_loss_sum = 0.
        for i, layer in enumerate(self.layers):
            layer: DeepseekV2DecoderLayer

            kargs = {
                'hidden_states': hidden_states,
                'attention_mask': attention_mask_2d,
                'use_cache': use_cache,
                'past_key_value': past_key_values[i] if past_key_values is not None else None,
            }
            if enable_checkpointing:
                if not self.training:
                    raise ValueError('recompute is not used because self.training is False')
                hidden_states, current_key_value = ms.recompute(layer, **kargs)
            else:
                hidden_states, current_key_value = layer.construct(**kargs)

            current_key_values += (current_key_value,)
            aux_loss_sum += layer.aux_loss

        # store aux loss for logging
        self.aux_loss = aux_loss_sum

        hidden_states = self.output_norm(hidden_states)
        current_key_values = current_key_values if use_cache else None
        logits = self.lm_head(hidden_states) if output_logits else None

        return hidden_states, current_key_values, logits
