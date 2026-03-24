# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


_SCI_MODEL_DEFAULTS = {
    "qk_norm": True,
    "rope_theta": 1000000,
    "head_dim": 128,
    "hidden_dim": 3072,
    "norm_eps": 1e-6,
    "depth_init": True,
    "enable_weight_tying": False,
    "moe_enabled": False,
    "moe_inter_dim": 768,
}


@dataclass
class Olmo3CustomModelArgs(BaseModelArgs):
    """OLMo-3 model arguments for titan_oellm custom training stack."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    vocab_size: int = 100278
    head_dim: int = 128
    hidden_dim: int = 11008

    norm_eps: float = 1e-6
    rope_theta: float = 500000
    rope_scaling_factor: float = 8.0
    qk_norm: bool = True
    max_seq_len: int = 65536
    depth_init: bool = True

    # OLMo-3 attention metadata from HF config.
    sliding_window: int = 4096
    layer_types: list[str] = field(default_factory=list)

    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"
    eos_id: int = 100257

    enable_weight_tying: bool = False

    # Keep MoE fields for compatibility with shared utilities; OLMo-3 is dense.
    moe_enabled: bool = False
    moe_inter_dim: int = 768
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    def _maybe_override(self, name: str, value):
        default = _SCI_MODEL_DEFAULTS.get(name)
        if default is not None and value == default and getattr(self, name) != default:
            # Avoid silently replacing OLMo flavor values with generic config defaults.
            return
        setattr(self, name, value)

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if hasattr(job_config.model, "vocab_size"):
            self.vocab_size = job_config.model.vocab_size
        if hasattr(job_config.model, "qk_norm"):
            self._maybe_override("qk_norm", job_config.model.qk_norm)
        if hasattr(job_config.model, "rope_theta"):
            self._maybe_override("rope_theta", job_config.model.rope_theta)
        if hasattr(job_config.model, "head_dim"):
            self._maybe_override("head_dim", job_config.model.head_dim)
        if hasattr(job_config.model, "hidden_dim"):
            self._maybe_override("hidden_dim", job_config.model.hidden_dim)
        if hasattr(job_config.model, "norm_eps"):
            self._maybe_override("norm_eps", job_config.model.norm_eps)
        if hasattr(job_config.model, "depth_init"):
            self._maybe_override("depth_init", job_config.model.depth_init)
        if hasattr(job_config.model, "enable_weight_tying"):
            self._maybe_override(
                "enable_weight_tying", job_config.model.enable_weight_tying
            )

        if hasattr(job_config.model, "moe_enabled"):
            self._maybe_override("moe_enabled", job_config.model.moe_enabled)
        if hasattr(job_config.model, "moe_inter_dim"):
            self._maybe_override("moe_inter_dim", job_config.model.moe_inter_dim)

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return get_moe_model_nparams_and_flops(self, model, 2 * self.head_dim, seq_len)
