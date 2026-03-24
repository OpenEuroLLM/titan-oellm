# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import build_optimizers
from titan_oellm.components.lr_scheduler_universal import build_lr_schedulers_auto
from titan_oellm.components.metrics_with_parameter_logging import build_metrics_processor_with_parameter_logging
from titan_oellm.components.validator import build_sci_validator

from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from titan_oellm.datasets.sci_dataloader import build_sci_dataloader
from titan_oellm.datasets.sci_tokenizers.sci_tokenizer import build_sci_hf_tokenizer

from torchtitan.distributed.pipeline_parallel import pipeline_llm

from .infra.parallelize import parallelize_olmo3_custom
from .model.args import Olmo3CustomModelArgs
from .model.model import Olmo3Model
from .model.state_dict_adapter import Olmo3StateDictAdapter

__all__ = [
    "parallelize_olmo3_custom",
    "Olmo3CustomModelArgs",
    "Olmo3Model",
    "Olmo3StateDictAdapter",
    "olmo3_custom_configs",
]


olmo3_custom_configs = {
    "debugmodel": Olmo3CustomModelArgs(
        dim=128,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=100278,
        head_dim=64,
        hidden_dim=512,
        norm_eps=1e-6,
        rope_theta=500000,
        qk_norm=True,
        max_seq_len=512,
        depth_init=True,
        sliding_window=128,
    ),
    "7B": Olmo3CustomModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        vocab_size=100278,
        head_dim=128,
        hidden_dim=11008,
        norm_eps=1e-6,
        rope_theta=500000,
        qk_norm=True,
        max_seq_len=65536,
        depth_init=True,
        sliding_window=4096,
    ),
    "32B": Olmo3CustomModelArgs(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        vocab_size=100278,
        head_dim=128,
        hidden_dim=27648,
        norm_eps=1e-6,
        rope_theta=500000,
        qk_norm=True,
        max_seq_len=65536,
        depth_init=True,
        sliding_window=4096,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Olmo3Model,
        model_args=olmo3_custom_configs,
        parallelize_fn=parallelize_olmo3_custom,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers_auto,
        build_dataloader_fn=build_sci_dataloader,
        build_tokenizer_fn=build_sci_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_sci_validator,
        build_metrics_processor_fn=build_metrics_processor_with_parameter_logging,
        state_dict_adapter=Olmo3StateDictAdapter,
    )


register_train_spec("olmo3_custom", get_train_spec())
