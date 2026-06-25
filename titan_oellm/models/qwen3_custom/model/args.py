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

    enable_weight_tying: bool = False
    use_complex_rope: bool = False  # Complex-mul RoPE (interleaved pairing, fewer intermediates); incompatible with HF checkpoints
    use_flex_attn: bool = False  # Not supported; field exists to guard compatibility checks

    # MoE params
    moe_enabled: bool = False
    moe_inter_dim: int = 768
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        # Update vocab_size if specified in config
        if hasattr(job_config.model, 'vocab_size'):
            self.vocab_size = job_config.model.vocab_size

        # Wire attention dispatch from job config. Without this, the model stays
        # on its "sdpa" default and asserts when SFT passes VarlenMetadata.
        if hasattr(job_config.model, "attn_mask_type"):
            self.attn_mask_type = job_config.model.attn_mask_type

        if hasattr(job_config.model, "attn_type") and job_config.model.attn_type:
            self.attn_type = job_config.model.attn_type
        elif hasattr(job_config.model, "use_flex_attn"):
            self.attn_type = "flex" if job_config.model.use_flex_attn else "sdpa"

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