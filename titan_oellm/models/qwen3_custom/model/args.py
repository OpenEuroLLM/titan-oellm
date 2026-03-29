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

        if hasattr(job_config.model, 'dim') and job_config.model.dim is not None:
            self.dim = job_config.model.dim
        if hasattr(job_config.model, 'n_layers')  and job_config.model.n_layers is not None:
            self.n_layers = job_config.model.n_layers
        if hasattr(job_config.model, "n_heads")  and job_config.model.n_heads is not None:
            self.n_heads = job_config.model.n_heads
        if hasattr(job_config.model, "qkv_bias") and job_config.model.qkv_bias is not None:
            self.qkv_bias = job_config.model.qkv_bias
        if hasattr(job_config.model, "mlp_bias")  and job_config.model.mlp_bias is not None:
            self.mlp_bias = job_config.model.mlp_bias
        if hasattr(job_config.model, "n_kv_heads")  and job_config.model.n_kv_heads is not None:
            self.n_kv_heads = job_config.model.n_kv_heads
        if hasattr(job_config.model, "vocab_size")  and job_config.model.vocab_size is not None:
            self.vocab_size = job_config.model.vocab_size
        if hasattr(job_config.model, "head_dim")  and job_config.model.head_dim is not None:
            self.head_dim = job_config.model.head_dim
        if hasattr(job_config.model, "hidden_dim")  and job_config.model.hidden_dim is not None:
            self.hidden_dim = job_config.model.hidden_dim
        
        if hasattr(job_config.model, "moe_num_experts"):
            self.moe_args.moe_num_experts = job_config.model.moe_num_experts
        if hasattr(job_config.model, "moe_top_k"):
            self.moe_args.top_k = job_config.model.moe_top_k
        if hasattr(job_config.model, "moe_score_func"):
            self.moe_args.score_func = job_config.model.moe_score_func
        if hasattr(job_config.model, "moe_route_norm"):
            self.moe_args.route_norm = job_config.model.moe_route_norm
        if hasattr(job_config.model, "moe_num_shared_experts"):
            self.moe_args.num_shared_experts = job_config.model.moe_num_shared_experts
        if hasattr(job_config.model, "moe_route_scale"):
            self.moe_args.route_scale = job_config.model.moe_route_scale
        if hasattr(job_config.model, "moe_score_before_experts"):
            self.moe_args.scale_before_experts = job_config.model.moe_score_before_experts



        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )

        # Override flavor values only for fields explicitly set in config
        # (typed as X | None = None in oellm_job_config; None means "use flavor").
        cfg = job_config.model
        for field in (
            'qk_norm', 'rope_theta', 'head_dim', 'hidden_dim', 'norm_eps',
            'depth_init', 'enable_weight_tying', 'use_complex_rope',
            'moe_enabled', 'moe_inter_dim',
        ):
            if hasattr(cfg, field):
                val = getattr(cfg, field)
                if val is not None:
                    setattr(self, field, val)

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