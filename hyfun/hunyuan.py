# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .switch_layers import SwitchGLU
import logging


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_bias: bool
    moe_topk: int
    num_experts: int
    num_shared_expert: int
    use_mixed_mlp_moe: bool
    use_qk_norm: bool
    rms_norm_eps: float
    rope_theta: float
    use_cla: bool
    cla_share_factor: 2
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")


class DynamicNTKAlphaRoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000,
        scaling_alpha: float = 1.0,
    ):
        super().__init__()
        self.dims = dims
        base = base * scaling_alpha ** (dims / (dims - 2))
        self._freqs = base ** (mx.arange(0, self.dims, 2) / self.dims)

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class Attention(nn.Module):
    def __init__(self, kv_proj: bool, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        if kv_proj:
            self.k_proj = nn.Linear(
                dim, n_kv_heads * head_dim, bias=args.attention_bias
            )
            self.v_proj = nn.Linear(
                dim, n_kv_heads * head_dim, bias=args.attention_bias
            )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)
        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.query_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)
            self.key_layernorm = nn.RMSNorm(head_dim, args.rms_norm_eps)

        self.rope = DynamicNTKAlphaRoPE(
            head_dim,
            base=args.rope_theta,
            scaling_alpha=args.rope_scaling["alpha"],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        kv_states=None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)

        if kv_states is None:
            keys, values = self.k_proj(x), self.v_proj(x)
            kv_states = keys, values
        else:
            keys, values = kv_states

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache.offset if cache else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if self.use_qk_norm:
            queries = self.query_layernorm(queries)
            keys = self.key_layernorm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), kv_states


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Gate(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.wg = nn.Linear(dim, num_experts, bias=False)

    def __call__(self, x) -> mx.array:
        return self.wg(x)


class MoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Add debug logging
        logging.info(f"Initializing MoeBlock with args: {vars(args)}")
        
        dim = args.hidden_size
        intermediate_size = args.intermediate_size
        try:
            self.use_shared_mlp = args.use_mixed_mlp_moe
            logging.info(f"use_shared_mlp: {self.use_shared_mlp}")

            if args.use_mixed_mlp_moe:
                logging.info("Creating shared MLP")
                self.shared_mlp = MLP(dim, intermediate_size * args.num_shared_expert)

            self.num_experts = num_experts = args.num_experts
            logging.info(f"num_experts: {num_experts}")
            self.top_k = args.moe_topk
            logging.info(f"top_k: {self.top_k}")

            logging.info("Creating gate")
            self.gate = Gate(dim, num_experts)
            logging.info("Creating switch_mlp")
            self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)
            logging.info("MoeBlock initialization complete")
        except Exception as e:
            logging.error(f"Error initializing MoeBlock: {e}")
            raise

    def __call__(self, x: mx.array):
        logging.info("Starting MoeBlock forward pass")
        logging.info(f"Input shape: {x.shape}")
        
        gates = self.gate(x)
        logging.info(f"Gate output shape: {gates.shape}")
        
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k-1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)

        logging.info(f"Expert indices shape: {inds.shape}")
        
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        if self.use_shared_mlp:
            logging.info("Running shared MLP path")
            shared_expert_output = self.shared_mlp(x)
            y = y + shared_expert_output
            
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, kv_proj: bool):
        super().__init__()
        logging.info(f"Initializing DecoderLayer with kv_proj={kv_proj}")
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(kv_proj, args)
        try:
            logging.info("Creating MLP/MoE block")
            self.mlp = MoeBlock(args)
            if self.mlp is None:
                logging.error("MLP is None after initialization!")
        except Exception as e:
            logging.error(f"Error creating MLP: {e}")
            raise
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv_states: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        # Add logging for debugging
        logging.info(f"DecoderLayer call - mlp type: {type(self.mlp)}")
        
        r, shared_kv_states = self.self_attn(
            self.input_layernorm(x), mask, cache, shared_kv_states
        )
        h = x + r

        # Add more detailed logging
        if self.mlp is None:
            logging.error("MLP is None during forward pass!")
        else:
            logging.info(f"MLP attributes: {dir(self.mlp)}")
            
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, shared_kv_states


class HunYuanModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, kv_proj=(i % args.cla_share_factor) == 0)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if i % self.args.cla_share_factor == 0:
                shared_kv_states = None
            h, shared_kv_states = layer(h, mask, c, shared_kv_states)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HunYuanModel(args)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        return self.model.embed_tokens.as_linear(out)

    # Fixed 1-16-2025 to handle stacked weights from conversion
    def sanitize(self, weights):
        """Maps weights to the expected HunYuan MoE structure"""
        # Debug print existing weight structure
        moe_keys = [k for k in weights.keys() if 'mlp' in k]
        logging.info(f"Found {len(moe_keys)} MoE-related keys")

        # Categorize keys to understand structure
        key_categories = {
            'switch_mlp': [k for k in moe_keys if 'switch_mlp' in k],
            'shared_mlp': [k for k in moe_keys if 'shared_mlp' in k],
            'gate': [k for k in moe_keys if '.gate.' in k]
        }
        
        for category, keys in key_categories.items():
            logging.info(f"\n{category} keys found: {len(keys)}")
            # Sample a few keys to see structure
            for k in keys[:3]:
                shape = weights[k].shape if k in weights else None
                logging.info(f"  {k}: {shape}")

        # Check if we already have the stacked format (switch_mlp)
        if any('switch_mlp' in k for k in weights.keys()):
            # Sample check - let's look at layer 0's weights
            logging.info("\nChecking layer 0 weight structure:")
            for comp in ['switch_mlp', 'shared_mlp', 'gate']:
                prefix = f"model.layers.0.mlp.{comp}"
                found_keys = [k for k in weights.keys() if k.startswith(prefix)]
                logging.info(f"\n{comp} component keys:")
                for k in found_keys:
                    shape = weights[k].shape if k in weights else None
                    logging.info(f"  {k}: {shape}")

            return weights

        # If we get here, we have the old PyTorch format that needs conversion
        if "model.layers.0.mlp.experts.0.up_proj.weight" in weights:
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}"
                # Convert experts to stacked format
                for n in ["up_proj", "down_proj", "gate_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        expert_key = f"{prefix}.mlp.experts.0.{n}.{k}"
                        if expert_key in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(to_join)

        return weights

    @property
    def layers(self):
        return self.model.layers
