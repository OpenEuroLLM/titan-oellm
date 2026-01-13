# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from titan_oellm.components.lr_scheduler_universal import build_lr_schedulers_auto
from titan_oellm.components.metrics_with_parameter_logging import build_metrics_processor_with_parameter_logging
from titan_oellm.components.validator import build_sci_validator
from titan_oellm.datasets.sci_dataloader import build_sci_dataloader
from titan_oellm.datasets.sci_tokenizers.sci_tokenizer import build_sci_hf_tokenizer

from torchtitan.distributed.pipeline_parallel import pipeline_llm

from .infra.parallelize import parallelize_gpt_plus
from .model.args import TransformerModelArgs
from .model.model import Transformer

__all__ = [
    "parallelize_gpt_plus",
    "TransformerModelArgs",
    "Transformer",
    "gpt_plus_configs",
]


gpt_plus_configs = {
    "debugmodel": TransformerModelArgs(
        dim=128, n_layers=6, n_heads=8, rope_theta=500000  # head_dim=16 for flex_attn compatibility
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=128,
        n_layers=6,
        n_heads=8,  # head_dim=16 for flex_attn compatibility
        ffn_dim_multiplier=4,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "ref130M": TransformerModelArgs(
        dim=512,
        n_layers=22,
        n_heads=4,
        n_kv_heads=4,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "cc50M": TransformerModelArgs(
        dim=256,
        n_layers=16,
        n_heads=2,
        n_kv_heads=2,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "cc150M": TransformerModelArgs(
        dim=512,
        n_layers=20,
        n_heads=4,
        n_kv_heads=4,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "cc330M": TransformerModelArgs(
        dim=769,
        n_layers=24,
        n_heads=6,
        n_kv_heads=6,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "cc600M": TransformerModelArgs(
        dim=1024,
        n_layers=28,
        n_heads=8,
        n_kv_heads=8,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "cc950M": TransformerModelArgs(
        dim=1280,
        n_layers=30,
        n_heads=10,
        n_kv_heads=10,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "cc1.4B": TransformerModelArgs(
        dim=1536,
        n_layers=32,
        n_heads=12,
        n_kv_heads=12,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "cc2B": TransformerModelArgs(
        dim=1792,
        n_layers=34,
        n_heads=14,
        n_kv_heads=14,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
        tie_embedding=True,
    ),
    "125M": TransformerModelArgs(
        dim=512,
        n_layers=18,
        n_heads=4,
        n_kv_heads=4,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
    ),
    "0.25B": TransformerModelArgs(
        dim=768,
        n_layers=18,
        n_heads=6,
        n_kv_heads=6,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
    ),
    "0.5B": TransformerModelArgs(
        dim=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=8,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
    ),
    "1B": TransformerModelArgs(
        dim=1280,
        n_layers=36,
        n_heads=10,
        n_kv_heads=10,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
    ),
    "1.8B": TransformerModelArgs(
        dim=2048,
        n_layers=26,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=4,
        multiple_of=128,
        rope_theta=500000,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=4,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=4,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=4,
        multiple_of=4096,
        rope_theta=500000,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=gpt_plus_configs,
        parallelize_fn=parallelize_gpt_plus,
        pipelining_fn=pipeline_llm,  # Using standard torchtitan v0.2.0 pipeline function
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers_auto,  # Auto-select scheduler based on config
        build_dataloader_fn=build_sci_dataloader,
        build_tokenizer_fn=build_sci_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_sci_validator,
        build_metrics_processor_fn=build_metrics_processor_with_parameter_logging,
    )


# Register the train spec with torchtitan
register_train_spec("gpt_plus", get_train_spec())
