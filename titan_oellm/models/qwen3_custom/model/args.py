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

    # FP8 activation storage: quantize the attention/FFN pre-projection activations
    # to FP8 E4M3 on the backward tape (halves stored activation memory) with a
    # straight-through-estimator backward. Training-time only; pure torch.
    fp8_activations: bool = False

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
        self.max_seq_len = seq_len

        # Update vocab_size if specified in config
        if hasattr(job_config.model, 'vocab_size'):
            self.vocab_size = job_config.model.vocab_size

        # Override flavor values only for fields explicitly set in config
        # (typed as X | None = None in oellm_job_config; None means "use flavor").
        cfg = job_config.model
        for field in (
            'qk_norm', 'rope_theta', 'head_dim', 'hidden_dim', 'norm_eps',
            'depth_init', 'enable_weight_tying', 'use_complex_rope',
            'moe_enabled', 'moe_inter_dim', 'fp8_activations',
        ):
            if hasattr(cfg, field):
                val = getattr(cfg, field)
                if val is not None:
                    setattr(self, field, val)
        for field in (
            'qk_norm', 'rope_theta', 'head_dim', 'hidden_dim', 'norm_eps',
            'depth_init', 'enable_weight_tying', 'use_complex_rope',
            'moe_enabled', 'moe_inter_dim', 'fp8_activations',
        ):
            if hasattr(cfg, field):
                val = getattr(cfg, field)
                if val is not None:
                    setattr(self, field, val)

        # MoE debug force load balance
        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

        # Populate the module-level config dicts for the custom optimizers so the
        # TrainSpec's build_optimizers_fn dispatch (by [optimizer].name) can read
        # them. These run unconditionally regardless of which optimizer is active.
        self._configure_custom_optimizers(job_config)

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

    @staticmethod
    def _configure_custom_optimizers(job_config: JobConfig) -> None:
        """Push the [constrained_adam] / [shared_bound_muon] / [bounded_muon]
        config sections into the optimizers' module-level config dicts.

        Mirrors titan-sci: each configure_*() is called unconditionally so the
        dicts are populated before the TrainSpec's build_optimizers_fn dispatch
        picks the active optimizer by [optimizer].name.
        """
        from titan_oellm.optimizer import (
            configure_constrained_adam,
            configure_adam_cpr,
            configure_bounded_spherical_adam,
            configure_shared_bound_muon,
            configure_bounded_muon,
        )

        if hasattr(job_config, "adam_cpr"):
            acpr = job_config.adam_cpr
            configure_adam_cpr(
                betas=tuple(acpr.betas),
                eps=acpr.eps,
                kappa_init_method=acpr.kappa_init_method,
                kappa_init_param=acpr.kappa_init_param,
                reg_function=acpr.reg_function,
                kappa_update=acpr.kappa_update,
                reg_step_size=acpr.reg_step_size,
                reg_ema_decay=acpr.reg_ema_decay,
                reg_embedding=acpr.reg_embedding,
                reg_by_lr=acpr.reg_by_lr,
                amsgrad=acpr.amsgrad,
            )

        if hasattr(job_config, "bounded_spherical_adam"):
            bsa = job_config.bounded_spherical_adam
            configure_bounded_spherical_adam(
                betas=bsa.betas,
                eps=bsa.eps,
                weight_decay=bsa.weight_decay,
                mode=bsa.mode,
                max_norm=bsa.max_norm,
                project_gradients=bsa.project_gradients,
                correct_v=bsa.correct_v,
                soft_blend=bsa.soft_blend,
                rotation_lr=bsa.rotation_lr,
                fallback_lr=bsa.fallback_lr,
                embedding_lr=bsa.embedding_lr,
                n_iter_spectral=bsa.n_iter_spectral,
                n_iter_ns=bsa.n_iter_ns,
                n_iter=bsa.n_iter,
                ffn_down_left_ns=bsa.ffn_down_left_ns,
                ns_alpha=bsa.ns_alpha,
                ns_mode=bsa.ns_mode,
                kappa_target=bsa.kappa_target,
                lambda_max=bsa.lambda_max,
                kappa_ema_beta=bsa.kappa_ema_beta,
                ns_schedule_steps=bsa.ns_schedule_steps,
                out_norm_dim_0=bsa.out_norm_dim_0,
            )

        if hasattr(job_config, "constrained_adam"):
            ca = job_config.constrained_adam
            configure_constrained_adam(
                betas=ca.betas,
                eps=ca.eps,
                weight_decay=ca.weight_decay,
                mode=ca.mode,
                max_norm=ca.max_norm,
                delta=ca.delta,
                project_momentum=ca.project_momentum,
                parallel_transport=ca.parallel_transport,
                fallback_lr=ca.fallback_lr,
                embedding_lr=ca.embedding_lr,
                embedding_norm=ca.embedding_norm,
            )

        if hasattr(job_config, "shared_bound_muon"):
            sbm = job_config.shared_bound_muon
            configure_shared_bound_muon(
                betas=sbm.betas,
                eps=sbm.eps,
                weight_decay=sbm.weight_decay,
                mode=sbm.mode,
                max_norm=sbm.max_norm,
                project_gradients=sbm.project_gradients,
                soft_blend=sbm.soft_blend,
                rotation_lr=sbm.rotation_lr,
                fallback_lr=sbm.fallback_lr,
                embedding_lr=sbm.embedding_lr,
                n_iter_spectral=sbm.n_iter_spectral,
                n_iter_ns=sbm.n_iter_ns,
                n_iter=sbm.n_iter,
                ffn_down_left_ns=sbm.ffn_down_left_ns,
                ns_alpha=sbm.ns_alpha,
                ns_mode=sbm.ns_mode,
                kappa_target=sbm.kappa_target,
                lambda_max=sbm.lambda_max,
                kappa_ema_beta=sbm.kappa_ema_beta,
                ns_schedule_steps=sbm.ns_schedule_steps,
                out_norm_dim_0=sbm.out_norm_dim_0,
                muon_on_gradient=sbm.muon_on_gradient,
                muon_ns_steps=sbm.muon_ns_steps,
                muon_ns_mode=sbm.muon_ns_mode,
                muon_preserve_norm=sbm.muon_preserve_norm,
                muon_ns_dtype=sbm.muon_ns_dtype,
            )

        if hasattr(job_config, "bounded_muon"):
            bm = job_config.bounded_muon
            configure_bounded_muon(
                betas=bm.betas,
                eps=bm.eps,
                weight_decay=bm.weight_decay,
                max_norm=bm.max_norm,
                project_gradients=bm.project_gradients,
                soft_blend=bm.soft_blend,
                out_norm_dim_0=bm.out_norm_dim_0,
                rotation_lr=bm.rotation_lr,
                fallback_lr=bm.fallback_lr,
                embedding_lr=bm.embedding_lr,
                muon_beta1=bm.muon_beta1,
                muon_nesterov=bm.muon_nesterov,
                muon_ns_steps=bm.muon_ns_steps,
                muon_ns_mode=bm.muon_ns_mode,
                muon_scale=bm.muon_scale,
                muon_norm_preserve=bm.muon_norm_preserve,
                muon_bias_correction=bm.muon_bias_correction,
                muon_geodesic=bm.muon_geodesic,
                muon_adam_scale=bm.muon_adam_scale,
                muon_flat_scale=bm.muon_flat_scale,
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return get_moe_model_nparams_and_flops(self, model, 2 * self.head_dim, seq_len)