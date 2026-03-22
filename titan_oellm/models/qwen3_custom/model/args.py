# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs

from torchtitan.tools.logging import logger


@dataclass
class Qwen3CustomModelArgs(BaseModelArgs):
    """
    Qwen3 model arguments with titan-sci integration.

    This extends the base Qwen3 model with:
    - Integration with oellm_job_config for TOML configuration
    - Support for sci_dataloader and learning rate schedulers
    - HuggingFace checkpoint loading via state_dict_adapter
    """

    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    vocab_size: int = 151936
    head_dim: int = 128
    hidden_dim: int = 3072

    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    qk_norm: bool = True
    max_seq_len: int = 4096
    depth_init: bool = True

    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"
    eos_id: int = 151645

    qkv_bias: bool = False
    mlp_bias: bool = False

    enable_weight_tying: bool = False
    use_complex_rope: bool = False  # Complex-mul RoPE (interleaved pairing, fewer intermediates); incompatible with HF checkpoints
    use_flex_attn: bool = False  # Not supported; field exists to guard compatibility checks

    # MoE params
    moe_enabled: bool = False
    moe_inter_dim: int = 768
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        if hasattr(job_config.model, "max_seq_len") and job_config.model.max_seq_len:
            self.max_seq_len = job_config.model.max_seq_len
        elif hasattr(job_config, "max_seq_len"):
            self.max_seq_len = job_config.max_seq_len

        if hasattr(job_config.model, 'dim'):
            self.dim = job_config.model.dim
        if hasattr(job_config.model, 'n_layers'):
            self.n_layers = job_config.model.n_layers
        if hasattr(job_config.model, "n_heads"):
            self.n_heads = job_config.model.n_heads
        if hasattr(job_config.model, "qkv_bias"):
            self.qkv_bias = job_config.model.qkv_bias
        if hasattr(job_config.model, "mlp_bias"):
            self.mlp_bias = job_config.model.mlp_bias
        if hasattr(job_config.model, "n_kv_heads"):
            self.n_kv_heads = job_config.model.n_kv_heads
        if hasattr(job_config.model, "vocab_size"):
            self.vocab_size = job_config.model.vocab_size
        if hasattr(job_config.model, "head_dim"):
            self.head_dim = job_config.model.head_dim
        if hasattr(job_config.model, "hidden_dim"):
            self.hidden_dim = job_config.model.hidden_dim
        
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )

        # Update vocab_size if specified in config
        if hasattr(job_config.model, 'vocab_size'):
            self.vocab_size = job_config.model.vocab_size

        # Update Qwen3-specific parameters from config.
        # Fields typed as Optional (int | None, bool | None) in sci_job_config
        # default to None → skip to preserve the flavor value.
        cfg = job_config.model
        if hasattr(cfg, 'qk_norm'):
            self.qk_norm = cfg.qk_norm
        if hasattr(cfg, 'rope_theta'):
            self.rope_theta = cfg.rope_theta
        if hasattr(cfg, 'head_dim') and cfg.head_dim is not None:
            self.head_dim = cfg.head_dim
        if hasattr(cfg, 'hidden_dim') and cfg.hidden_dim is not None:
            self.hidden_dim = cfg.hidden_dim
        if hasattr(cfg, 'norm_eps'):
            self.norm_eps = cfg.norm_eps
        if hasattr(cfg, 'depth_init'):
            self.depth_init = cfg.depth_init
        if hasattr(cfg, 'enable_weight_tying') and cfg.enable_weight_tying is not None:
            self.enable_weight_tying = cfg.enable_weight_tying
        if hasattr(cfg, 'use_complex_rope'):
            self.use_complex_rope = cfg.use_complex_rope

        # MoE configuration
        if hasattr(cfg, 'moe_enabled') and cfg.moe_enabled is not None:
            self.moe_enabled = cfg.moe_enabled
        if hasattr(cfg, 'moe_inter_dim') and cfg.moe_inter_dim is not None:
            self.moe_inter_dim = cfg.moe_inter_dim

        # MoE debug force load balance
        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

        # Compatibility checks
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
        return get_moe_model_nparams_and_flops(self, model, 2 * self.head_dim, seq_len)