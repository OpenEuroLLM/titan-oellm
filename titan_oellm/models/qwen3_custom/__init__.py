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

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers
from titan_oellm.components.lr_scheduler_universal import build_lr_schedulers_auto
from titan_oellm.components.metrics_with_parameter_logging import build_metrics_processor_with_parameter_logging
from titan_oellm.components.validator import build_sci_validator

from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.models.moe import MoEArgs

from titan_oellm.datasets.sci_dataloader import build_sci_dataloader
from titan_oellm.datasets.sci_tokenizers.sci_tokenizer import build_sci_hf_tokenizer

from .infra.parallelize import parallelize_qwen3_custom
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
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,  # Standard optimizer (AdamW)
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
