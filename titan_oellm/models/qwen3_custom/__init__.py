# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.
#
# Qwen3 Custom Model for Titan-OELLM
# Adapted from torchtitan Qwen3 implementation with titan-sci integration

import torch

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers, OptimizersContainer
from titan_oellm.components.lr_scheduler_universal import build_lr_schedulers_auto
from titan_oellm.components.metrics_with_parameter_logging import build_metrics_processor_with_parameter_logging
from titan_oellm.components.validator import build_sci_validator

from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.models.moe import MoEArgs

from titan_oellm.datasets.sci_dataloader import build_sci_dataloader
from titan_oellm.datasets.sci_tokenizers.sci_tokenizer import build_sci_hf_tokenizer

from .infra.parallelize import parallelize_qwen3_custom
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from .model.args import Qwen3CustomModelArgs
from .model.model import Qwen3Model
from .model.state_dict_adapter import Qwen3StateDictAdapter

__all__ = [
    "parallelize_qwen3_custom",
    "Qwen3CustomModelArgs",
    "Qwen3Model",
    "Qwen3StateDictAdapter",
    "qwen3_custom_configs",
]


# Qwen3 model configurations based on official Qwen3 sizes
# Reference: https://huggingface.co/collections/Qwen/qwen3-6751c5cbf6fc98b0838a3d2f
qwen3_custom_configs = {
    "debugmodel": Qwen3CustomModelArgs(
        dim=128,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=151936,
        head_dim=64,
        hidden_dim=512,
        norm_eps=1e-6,
        rope_theta=10000,
        qk_norm=True,
        max_seq_len=512,
        depth_init=True,
    ),
    "125M": Qwen3CustomModelArgs(
        dim=512,
        n_layers=18,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=2048,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=8192,
        depth_init=True,
    ),
    "125M768": Qwen3CustomModelArgs(
        dim=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=6,
        head_dim=64,
        hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=500000,
        vocab_size=151936,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
        enable_weight_tying=True,
    ),
    "130Msci": Qwen3CustomModelArgs(
        dim=576,
        n_layers=18,
        n_heads=9,
        n_kv_heads=9,
        head_dim=64,
        hidden_dim=2304,
        norm_eps=1e-6,
        rope_theta=500000,
        vocab_size=50432,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
        enable_weight_tying=True,
    ),
    "0.5B": Qwen3CustomModelArgs(
        dim=896,
        n_layers=24,
        n_heads=14,
        n_kv_heads=2,
        vocab_size=151936,
        head_dim=64,
        hidden_dim=4864,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=32768,
        depth_init=True,
    ),
    "0.6B": Qwen3CustomModelArgs(
        dim=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=3072,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=32768,
        depth_init=True,
    ),
    "1.7B": Qwen3CustomModelArgs(
        dim=1536,
        n_layers=28,
        n_heads=12,
        n_kv_heads=2,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=8960,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=32768,
        depth_init=True,
    ),
    "1.7Bsci": Qwen3CustomModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=32,
        n_kv_heads=32,
        vocab_size=151936,
        head_dim=64,
        hidden_dim=8192,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
    ),
    "4B": Qwen3CustomModelArgs(
        dim=2560,
        n_layers=36,
        n_heads=20,
        n_kv_heads=4,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=13824,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=32768,
        depth_init=True,
    ),
    "8B": Qwen3CustomModelArgs(
        dim=3584,
        n_layers=36,
        n_heads=28,
        n_kv_heads=4,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=18944,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=32768,
        depth_init=True,
    ),
    "14B": Qwen3CustomModelArgs(
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=8,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=13824,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=32768,
        depth_init=True,
    ),
    "32B": Qwen3CustomModelArgs(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=27648,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=32768,
        depth_init=True,
    ),
}


# MoE Qwen3 variants for MoE experiments
def _qwen3_moe_args(
    num_experts: int = 32,
    top_k: int = 8,
    score_func: str = "softmax",
    route_norm: bool = True,
) -> MoEArgs:
    return MoEArgs(
        num_experts=num_experts,
        num_shared_experts=0,
        top_k=top_k,
        score_func=score_func,
        route_norm=route_norm,
        route_scale=1.0,
        score_before_experts=False,
    )


qwen3_moe_configs = {
    "30BA3B": Qwen3CustomModelArgs(
        # Matches Megatron qwen3_moe_30BA3B.yaml architecture:
        #   hidden_size=2048, ffn_hidden_size=6144, num_layers=48
        #   num_attention_heads=32, num_query_groups=4 (GQA), kv_channels=128
        #   num_experts=128, moe_router_topk=8, moe_ffn_hidden_size=768
        dim=2048,
        n_layers=48,
        n_heads=32,
        n_kv_heads=4,          # GQA: 4 KV heads (num_query_groups=4)
        head_dim=128,          # kv_channels=128
        hidden_dim=6144,       # dense FFN size (not used when moe_enabled=True)
        vocab_size=151936,     # Qwen3 vocab; override to 50304 for GPT-NeoX data
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
        moe_enabled=True,
        moe_inter_dim=768,     # moe_ffn_hidden_size=768 per expert
        moe_args=_qwen3_moe_args(
            num_experts=128,
            top_k=8,
        ),
    ),
    "235BA22B": Qwen3CustomModelArgs(
        # Matches Megatron qwen3_moe_235BA22B.yaml architecture:
        #   hidden_size=4096, ffn_hidden_size=12288, num_layers=94
        #   num_attention_heads=64, num_query_groups=4 (GQA), kv_channels=128
        #   num_experts=128, moe_router_topk=8, moe_ffn_hidden_size=1536
        #   rope_theta=5000000 (different from 30BA3B's 1000000)
        dim=4096,
        n_layers=94,
        n_heads=64,
        n_kv_heads=4,          # GQA: 4 KV heads
        head_dim=128,          # kv_channels=128; Q: 64×128=8192=2×hidden
        hidden_dim=12288,      # dense FFN size (not used when moe_enabled=True)
        vocab_size=151936,     # Qwen3 vocab; override to 50304 for GPT-NeoX data
        norm_eps=1e-6,
        rope_theta=5000000,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
        moe_enabled=True,
        moe_inter_dim=1536,    # moe_ffn_hidden_size=1536 per expert
        moe_args=_qwen3_moe_args(
            num_experts=128,
            top_k=8,
        ),
    ),
    "debugmodel_moe": Qwen3CustomModelArgs(
        dim=256,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=50432,  # neox vocab padded to multiple of 64 (actual tokens: 50277)
        head_dim=128,
        hidden_dim=1024,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
        moe_enabled=True,
        moe_inter_dim=512,
        moe_args=_qwen3_moe_args(
            num_experts=64,
            top_k=8,
        ),
    ),
    "600M-A60M": Qwen3CustomModelArgs(
        dim=512,
        n_layers=16,
        n_heads=8,
        n_kv_heads=4,
        vocab_size=151936,
        head_dim=128,
        hidden_dim=2048,
        norm_eps=1e-6,
        rope_theta=1000000,
        qk_norm=True,
        max_seq_len=4096,
        depth_init=True,
        # MoE configuration for ~600M total / ~60M active
        moe_enabled=True,
        moe_inter_dim=512,
        moe_args=_qwen3_moe_args(
            num_experts=32,
            top_k=8,
            score_func="softmax",
            route_norm=True,
        ),
    ),
}


# Merge baseline and MoE configs
qwen3_custom_configs = {**qwen3_custom_configs, **qwen3_moe_configs}


class _OptimizersContainerNoWdBiases(OptimizersContainer):
    """OptimizersContainer that excludes 1-D params (biases, norm scales) from weight decay.

    Matches Megatron-LM behavior: params with ndim < 2 (biases and norm scales)
    are placed in a separate param group with weight_decay=0.
    """

    def __init__(self, model_parts, optimizer_cls, optimizer_kwargs):
        wd = optimizer_kwargs.get("weight_decay", 0.0)
        # base_kwargs go to optimizer as defaults (no weight_decay — set per group)
        base_kwargs = {k: v for k, v in optimizer_kwargs.items() if k != "weight_decay"}

        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            decay_params = [p for p in params if p.dim() >= 2]
            nodecay_params = [p for p in params if p.dim() < 2]
            param_groups = [
                {"params": decay_params, "weight_decay": wd},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            self.optimizers.append(optimizer_cls(param_groups, **base_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)


def build_optimizers_no_wd_biases(model_parts, optimizer_config, parallel_dims, ft_manager=None):
    """Build optimizers with weight decay=0 for 1-D params (biases, norm scales).

    Falls back to the standard build_optimizers for special cases (early_step_in_backward,
    FaultTolerance). Matches Megatron-LM's no_wd parameter grouping.
    """
    if optimizer_config.early_step_in_backward or (
        ft_manager is not None and ft_manager.enabled
    ):
        return build_optimizers(model_parts, optimizer_config, parallel_dims, ft_manager)

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if optimizer_config.name not in optimizer_classes:
        return build_optimizers(model_parts, optimizer_config, parallel_dims, ft_manager)

    optimizer_cls = optimizer_classes[optimizer_config.name]
    impl = optimizer_config.implementation
    optimizer_kwargs = {
        "lr": optimizer_config.lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
        "fused": impl == "fused",
        "foreach": impl == "foreach",
    }
    return _OptimizersContainerNoWdBiases(model_parts, optimizer_cls, optimizer_kwargs)


def get_train_spec() -> TrainSpec:
    """
    Create and return the training specification for qwen3_custom model.

    This integrates the Qwen3 model with titan-sci infrastructure:
    - sci_dataloader for data loading
    - Universal LR scheduler (3-phase: warm, main, cooldown)
    - sci_validator for validation
    - Parameter logging with TensorBoard
    - HuggingFace checkpoint loading via state_dict_adapter
    """
    return TrainSpec(
        model_cls=Qwen3Model,
        model_args=qwen3_custom_configs,
        parallelize_fn=parallelize_qwen3_custom,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_no_wd_biases,  # Exclude biases/norms from WD
        build_lr_schedulers_fn=build_lr_schedulers_auto,  # Universal LR scheduler
        build_dataloader_fn=build_sci_dataloader,  # Sci dataloader with MMap support
        build_tokenizer_fn=build_sci_hf_tokenizer,  # HF tokenizer with BOS/EOS control
        build_loss_fn=build_cross_entropy_loss,  # Standard cross-entropy loss
        build_validator_fn=build_sci_validator,  # Sci validator
        build_metrics_processor_fn=build_metrics_processor_with_parameter_logging,
        state_dict_adapter=Qwen3StateDictAdapter,  # HF checkpoint loading
    )


# Register the train spec with torchtitan
register_train_spec("qwen3_custom", get_train_spec())
