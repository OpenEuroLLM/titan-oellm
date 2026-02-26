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
    - Integration with sci_job_config for TOML configuration
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

        # Update Qwen3-specific parameters from config
        if hasattr(job_config.model, 'qk_norm'):
            self.qk_norm = job_config.model.qk_norm
        if hasattr(job_config.model, 'rope_theta'):
            self.rope_theta = job_config.model.rope_theta
        if hasattr(job_config.model, 'head_dim'):
            self.head_dim = job_config.model.head_dim
        if hasattr(job_config.model, 'hidden_dim'):
            self.hidden_dim = job_config.model.hidden_dim
        if hasattr(job_config.model, 'norm_eps'):
            self.norm_eps = job_config.model.norm_eps
        if hasattr(job_config.model, 'depth_init'):
            self.depth_init = job_config.model.depth_init
        if hasattr(job_config.model, 'enable_weight_tying'):
            self.enable_weight_tying = job_config.model.enable_weight_tying

        # MoE configuration
        if hasattr(job_config.model, 'moe_enabled'):
            self.moe_enabled = job_config.model.moe_enabled
        if hasattr(job_config.model, 'moe_inter_dim'):
            self.moe_inter_dim = job_config.model.moe_inter_dim

        # MoE debug force load balance
        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return get_moe_model_nparams_and_flops(self, model, 2 * self.head_dim, seq_len)
