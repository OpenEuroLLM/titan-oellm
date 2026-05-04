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
    "rope_scaling_factor": 8.0,
    "rope_old_context_len": 8192,
    "rope_beta_fast": 32,
    "rope_beta_slow": 1,
    "head_dim": 128,
    "hidden_dim": 3072,
    "norm_eps": 1e-6,
    "depth_init": True,
    "enable_weight_tying": False,
    "moe_enabled": False,
    "moe_inter_dim": 768,
    "sliding_window": 4096,
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
    rope_old_context_len: int = 8192
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
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

    # When True, swap in Liger fused Triton kernels:
    #   - LM head: fused linear + softmax + cross-entropy (skips materializing
    #     the (B*T, V) logits, ~13 GB at vocab=100278, seq=32k).
    #   - All RMSNorm modules (attention_norm, ffn_norm, q_norm, k_norm, final).
    #   - SwiGLU fused silu*mul in the FeedForward MLP.
    # Requires labels to be passed into model.forward (trainer wires this for CE).
    use_liger_kernels: bool = False

    # Keep MoE fields for compatibility with shared utilities; OLMo-3 is dense.
    moe_enabled: bool = False
    moe_inter_dim: int = 768
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    def default_uses_sliding(self) -> list[bool]:
        """OLMo-3 layer pattern: [sliding × 3, full] repeated, last layer forced full.

        Mirrors OLMo-core's SlidingWindowAttentionConfig used by olmo3_7B/13B/32B
        (pattern=[4096,4096,4096,-1], force_full_attention_on_last_layer=True).
        Used as the default when `layer_types` is not provided. Olmo3Model and
        get_nparams_and_flops both consume this so MFU stays in sync with the
        attention pattern that actually executes.
        """
        n = self.n_layers
        result = [((i % 4) != 3) for i in range(n)]
        if n > 0:
            result[-1] = False
        return result

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
        if hasattr(job_config.model, "attn_mask_type"):
            self.attn_mask_type = job_config.model.attn_mask_type
        if hasattr(job_config.model, "eos_id"):
            self.eos_id = job_config.model.eos_id
        if hasattr(job_config.model, "qk_norm"):
            self._maybe_override("qk_norm", job_config.model.qk_norm)
        if hasattr(job_config.model, "rope_theta"):
            self._maybe_override("rope_theta", job_config.model.rope_theta)
        if hasattr(job_config.model, "rope_scaling_factor"):
            self._maybe_override(
                "rope_scaling_factor", job_config.model.rope_scaling_factor
            )
        if hasattr(job_config.model, "rope_old_context_len"):
            self._maybe_override(
                "rope_old_context_len", job_config.model.rope_old_context_len
            )
        if hasattr(job_config.model, "rope_beta_fast"):
            self._maybe_override("rope_beta_fast", job_config.model.rope_beta_fast)
        if hasattr(job_config.model, "rope_beta_slow"):
            self._maybe_override("rope_beta_slow", job_config.model.rope_beta_slow)
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
        if hasattr(job_config.model, "sliding_window"):
            self._maybe_override("sliding_window", job_config.model.sliding_window)
        if hasattr(job_config.model, "layer_types"):
            self.layer_types = list(job_config.model.layer_types)

        if hasattr(job_config.model, "attn_type") and job_config.model.attn_type:
            self.attn_type = job_config.model.attn_type
        else:
            if hasattr(job_config.model, "use_flex_attn"):
                self.attn_type = "flex" if job_config.model.use_flex_attn else "sdpa"
            if hasattr(job_config.model, "use_flash_attn") and job_config.model.use_flash_attn:
                # OLMo3 custom attention uses SDPA path; PyTorch dispatches flash kernels
                # underneath SDPA when available.
                if self.attn_type != "flex":
                    self.attn_type = "sdpa"

        if hasattr(job_config.model, "use_liger_kernels"):
            self.use_liger_kernels = bool(job_config.model.use_liger_kernels)

        if hasattr(job_config.model, "moe_enabled"):
            self._maybe_override("moe_enabled", job_config.model.moe_enabled)
        if hasattr(job_config.model, "moe_inter_dim"):
            self._maybe_override("moe_inter_dim", job_config.model.moe_inter_dim)

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # torchtitan's helper assumes every layer attends over the full `seq_len`,
        # which overstates FLOPs (and therefore MFU) whenever any layer uses a
        # sliding window smaller than seq_len. We call it to get `nparams` and
        # the non-attention FLOP contribution, then overwrite the attention term
        # using the per-layer effective context length.
        head_dims = 2 * self.head_dim
        nparams, _ = get_moe_model_nparams_and_flops(self, model, head_dims, seq_len)

        # Per-layer sliding/full selection. Must match Olmo3Model's resolution
        # of layer types: explicit `layer_types` if given, else the OLMo-3
        # default pattern (sliding × 3, full × 1, last layer full).
        if self.layer_types:
            uses_sliding = [lt == "sliding_attention" for lt in self.layer_types]
        else:
            uses_sliding = self.default_uses_sliding()

        window = self.sliding_window
        eff_len_sum = sum(
            min(seq_len, window) if (sw and window > 0) else seq_len
            for sw in uses_sliding
        )

        # Recompute the non-attention term the same way torchtitan does so we
        # can swap in the corrected attention term.
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )
        if self.moe_enabled:
            # Mirror torchtitan's active-param accounting for MoE; dense olmo3
            # never hits this branch.
            nparams_moe_router = nparams_shared_experts = nparams_experts = 0
            nparams_dense = 0
            for name, p in model.named_parameters():
                if "embedding" in name:
                    nparams_dense += p.numel()
                elif "moe.shared_experts" in name:
                    nparams_shared_experts += p.numel()
                elif "moe.router" in name:
                    nparams_moe_router += p.numel()
                elif "moe.experts" in name:
                    nparams_experts += p.numel()
                else:
                    nparams_dense += p.numel()
            nparams_sparse_active = (
                nparams_moe_router
                + nparams_shared_experts
                + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
            )
            non_attn_flops = 6 * (
                nparams_dense - nparams_embedding + nparams_sparse_active
            )
        else:
            non_attn_flops = 6 * (nparams - nparams_embedding)

        attn_flops = 6 * self.n_heads * head_dims * eff_len_sum
        num_flops_per_token = non_attn_flops + attn_flops

        logger.info(
            f"MFU flops: corrected for sliding-window attention "
            f"(window={window}, seq_len={seq_len}, "
            f"sliding_layers={sum(uses_sliding)}/{self.n_layers})"
        )
        return nparams, num_flops_per_token
