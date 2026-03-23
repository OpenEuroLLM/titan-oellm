# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field, fields

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs


@dataclass
class RoPEScalingArgs:
    scaling_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 50432  # default vocab size, can be overridden in config
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: int = 4
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    rope_scaling_args: RoPEScalingArgs = field(default_factory=RoPEScalingArgs)

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    use_flex_attn: bool = False
    use_flash_attn: bool = False  # Enable direct Flash Attention 2/3
    attn_mask_type: str = "causal"
    eos_id: int = 0

    qk_norm_type: str = "QKNormPlus"  # "QKNormPlus" or "RMSNorm"

    mlp_layers: int = 2  # Number of layers in MLP feedforward (2, 3, or 4)
    mlp_activation: str = "swiglu"  # MLP activation type: 'swiglu' (gated) or 'silu' (ungated)

    tie_embedding: bool = False  # Tie embedding and output head weights

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        self.max_seq_len = job_config.training.seq_len

        # Update vocab_size if specified in config
        if hasattr(job_config.model, 'vocab_size'):
            self.vocab_size = job_config.model.vocab_size

        # Update QK norm settings if provided in config
        if hasattr(job_config.model, 'qk_norm_type'):
            self.qk_norm_type = job_config.model.qk_norm_type

        # Update MLP settings if provided in config
        if hasattr(job_config.model, 'mlp_layers'):
            self.mlp_layers = job_config.model.mlp_layers
        if hasattr(job_config.model, 'mlp_activation'):
            self.mlp_activation = job_config.model.mlp_activation

        # Update tie_embedding setting if provided in config
        if hasattr(job_config.model, 'tie_embedding'):
            self.tie_embedding = job_config.model.tie_embedding

        # Update RoPE scaling settings if provided in config
        if hasattr(job_config.model, 'rope_scaling_factor'):
            self.rope_scaling_args.scaling_factor = job_config.model.rope_scaling_factor
        if hasattr(job_config.model, 'rope_low_freq_factor'):
            self.rope_scaling_args.low_freq_factor = job_config.model.rope_low_freq_factor
        if hasattr(job_config.model, 'rope_high_freq_factor'):
            self.rope_scaling_args.high_freq_factor = job_config.model.rope_high_freq_factor
        if hasattr(job_config.model, 'rope_original_max_position_embeddings'):
            self.rope_scaling_args.original_max_position_embeddings = job_config.model.rope_original_max_position_embeddings

        # FlexAttention config
        if hasattr(job_config.model, 'use_flex_attn'):
            self.use_flex_attn = job_config.model.use_flex_attn
        if hasattr(job_config.model, 'attn_mask_type'):
            self.attn_mask_type = job_config.model.attn_mask_type

        # Flash Attention config
        if hasattr(job_config.model, 'use_flash_attn'):
            self.use_flash_attn = job_config.model.use_flash_attn

        if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with selective AC yet. "
                "See https://github.com/pytorch/pytorch/issues/147879"
            )

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with CP yet. "
                "We are still working on this."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

        return nparams, num_flops_per_token