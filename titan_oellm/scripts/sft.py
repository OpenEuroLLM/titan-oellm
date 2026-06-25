#!/usr/bin/env python
"""
Supervised Fine-Tuning (SFT) Script for TorchTitan

This script extends the base TorchTitan training framework to support supervised
fine-tuning with instruction-formatted datasets. It provides:
- Custom data formatting for instruction/response pairs
- SFT-specific loss computation (masking prompt tokens)
- Learning rate warmup and decay suitable for fine-tuning
- Support for different instruction formats (Alpaca, ChatML, etc.)

Usage:
    python titan_oellm/scripts/sft.py --job.config_file configs/sft_config.toml
"""

import sys
import os
import logging
from pathlib import Path

# Add torchtitan root first to avoid namespace package shadowing
project_root = Path(__file__).resolve().parents[2]
torchtitan_root = project_root / "torchtitan"
if torchtitan_root.exists():
    sys.path.insert(0, str(torchtitan_root))
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Iterable

# Inductor autotuning: trades a longer first-step compile (~1-2 min extra) for
# ~5-10% steady-state matmul speedup. Safe to leave on; worth it for long runs.
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

import torchtitan
from torchtitan.train import Trainer
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.config import JobConfig, ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.attention import VarlenMetadata
from torchtitan.tools.logging import logger, init_logger

from titan_oellm.distributed.ring_attention import RingAttentionZigZagLoadBalancer
from titan_oellm.models.olmo3_custom.model.model import (
    RingVarlenMetadata,
    _compute_ring_half_indices,
)


class RankZeroFilter(logging.Filter):
    """Logging filter that only allows logs from rank 0 in distributed training."""
    
    def filter(self, record):
        """Only allow logs from rank 0."""
        rank = int(os.environ.get("RANK", "0"))
        return rank == 0

# Import model specs to register them
import titan_oellm.models.gpt_plus  # noqa: F401
import titan_oellm.models.olmo3_custom  # noqa: F401
import titan_oellm.models.qwen3_custom  # noqa: F401

class SFTTrainer(Trainer):
    """
    Supervised Fine-Tuning Trainer that extends the base TorchTitan Trainer.
    
    Key differences from base training:
    1. Custom data processing for instruction-response format
    2. Loss masking to only compute loss on response tokens
    3. SFT-specific metrics tracking (response accuracy, etc.)
    """
    
    def __init__(self, job_config: JobConfig):
        """Initialize SFT trainer with additional SFT-specific components."""
        super().__init__(job_config)

        # SFT-specific configuration
        self.instruction_format = job_config.data.instruction_format
        self.mask_prompt = job_config.training.mask_prompt

        # Detect ring_varlen + CP so we can bypass the SDPA-only CP context manager
        # and do sequence sharding ourselves in post_dataloading_process.
        self._attn_type = getattr(self.model_args, "attn_type", "sdpa")
        self._use_ring_varlen_cp = (
            self._attn_type == "ring_varlen"
            and self.parallel_dims.cp_enabled
        )
        # Liger fused linear+CE LM head: skips materializing the (B*T, V)
        # logits, freeing ~13 GB at vocab=100278 / seq=32k. Requires labels
        # to be passed into model.forward; we inject them here. Currently
        # only wired for the non-PP, non-CP path.
        self._use_liger = bool(getattr(self.model_args, "use_liger_kernels", False))
        if self._use_liger and self.parallel_dims.cp_enabled:
            raise NotImplementedError(
                "use_liger_kernels + context_parallel is not supported yet. "
                "Run with context_parallel_degree=1 or disable use_liger_kernels."
            )
        if self._use_liger and self._use_ring_varlen_cp:
            raise NotImplementedError(
                "use_liger_kernels + ring_varlen context_parallel is not supported: "
                "LigerFusedLinearCrossEntropyLoss computes a local-shard mean, "
                "which produces incorrect gradients across CP ranks. "
                "Disable use_liger_kernels or set context_parallel_degree=1."
            )
        if self._use_liger and self.parallel_dims.pp_enabled:
            raise NotImplementedError(
                "use_liger_kernels + pipeline_parallel is not supported."
            )
        if self._use_ring_varlen_cp:
            cp_mesh = self.parallel_dims.get_mesh("cp")
            cp_pg = cp_mesh.get_group()
            self._ring_lb = RingAttentionZigZagLoadBalancer(
                cp_rank=dist.get_rank(cp_pg),
                cp_world_size=dist.get_world_size(cp_pg),
            )
            logger.info(
                f"SFT ring_varlen CP: rank={dist.get_rank(cp_pg)}, "
                f"world_size={dist.get_world_size(cp_pg)}"
            )
            # Tracks micro-batch index within a grad-accum cycle so the
            # CP-grad-reduce hook only fires on the final micro-batch
            # (otherwise it would re-reduce already-aggregated grads).
            self._ring_accum_step = 0

        logger.info(
            f"SFT Trainer initialized with instruction format: {self.instruction_format}, "
            f"mask_prompt: {self.mask_prompt}"
        )
    
    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing for SFT: mask prompt tokens in labels.
        
        For supervised fine-tuning, we typically only compute loss on the response
        tokens, not the instruction/prompt tokens. This is done by setting prompt
        token labels to -100 (ignore_index).
        
        The dataloader should provide 'loss_mask' in input_dict that indicates
        which tokens should be included in loss computation.
        """
        # Remove dataloader-only keys before they reach the model.
        loss_mask = input_dict.pop('loss_mask', None)
        cu_seqlens = input_dict.pop('cu_seqlens', None)
        max_seqlen = input_dict.pop('max_seqlen', None)
        # Pop attention_masks if the dataloader pre-computed it (SDPA doc-mask path);
        # the base class re-derives and sets it in extra_kwargs, so having it in
        # extra_inputs too causes a duplicate keyword argument error.
        precomputed_attention_masks = input_dict.pop('attention_masks', None)

        # Validate input tensors before calling parent
        inputs_raw = input_dict.get("input")
        if inputs_raw is not None:
            assert inputs_raw.dtype in [torch.long, torch.int64], f"Input dtype must be long, got {inputs_raw.dtype}"
            assert inputs_raw.min() >= 0, f"Input tokens must be non-negative, got min={inputs_raw.min()}"
            assert inputs_raw.max() < self.tokenizer.vocab_size, f"Input tokens exceed vocab size {self.tokenizer.vocab_size}, got max={inputs_raw.max()}"

        if cu_seqlens is not None:
            # Build VarlenMetadata directly from the dataloader's pre-computed boundaries,
            # skipping get_attention_masks() which would re-derive them via an expensive
            # device-to-host sync over EOS token positions.
            inputs = input_dict["input"]
            extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}

            if self._use_ring_varlen_cp:
                # --- Ring attention + CP path ---
                # cu_seqlens from the dataloader is the *global* cumulative sequence
                # length tensor (CPU, int32) spanning all documents in the batch.
                # We use the zigzag load balancer to:
                #   1. Shard input_ids and labels to this rank's local slice.
                #   2. Recompute per-rank cu_seqlens = global_cu_doc_lens // cp_world_size.
                # The ring kernel then handles all K/V communication internally.
                #
                # Constraint (matching OLMo-core): batch size must be 1 so the
                # entire packed sequence is a single flat instance.
                B = inputs.shape[0]
                if B != 1:
                    raise RuntimeError(
                        f"ring_varlen CP requires batch size 1 per rank (got {B}). "
                        "Set local_batch_size=1 in your training config."
                    )
                # cu_seqlens must be on CPU for batch_shard_by_document.
                cu_seqlens_cpu = cu_seqlens.cpu()
                pad_id = self.tokenizer.pad_id if hasattr(self.tokenizer, "pad_id") else 0
                # Include loss_mask in sharding so it stays aligned with labels.
                shard_inputs = [inputs, labels]
                shard_seq_dims = [1, 1]
                shard_pad_values = [pad_id, -100]
                if loss_mask is not None:
                    shard_inputs.append(loss_mask)
                    shard_seq_dims.append(1)
                    shard_pad_values.append(0)  # masked-out tokens → 0 (excluded from loss)

                sharded, extra = self._ring_lb.batch_shard_by_document(
                    inputs=shard_inputs,
                    seq_dims=shard_seq_dims,
                    cu_doc_lens=cu_seqlens_cpu,
                    pad_values=shard_pad_values,
                )
                inputs = sharded[0]
                labels = sharded[1]
                loss_mask = sharded[2] if loss_mask is not None else None

                # extra = {"cu_doc_lens": per_rank_cu_doc_lens (CPU), "max_doc_len": int}
                # Move cu_seqlens to GPU here so the model forward never does a
                # CPU→GPU transfer inside the traced region (which breaks fullgraph).
                per_rank_cu_seqlens = extra["cu_doc_lens"].to(inputs.device)
                per_rank_max_seqlen = int(extra["max_doc_len"])
                # Compute zigzag half-indices eagerly (before torch.compile) because
                # get_half_index uses data-dependent .item() calls that break dynamo.
                half_index0, half_index1 = _compute_ring_half_indices(per_rank_cu_seqlens)
                extra_kwargs: dict[str, Any] = {
                    "attention_masks": RingVarlenMetadata(
                        cu_seq_q=per_rank_cu_seqlens,
                        cu_seq_k=per_rank_cu_seqlens,
                        max_q=per_rank_max_seqlen,
                        max_k=per_rank_max_seqlen,
                        half_index0=half_index0,
                        half_index1=half_index1,
                    )
                }
            else:
                # --- Single-rank varlen path (no CP) ---
                extra_kwargs = {
                    "attention_masks": VarlenMetadata(
                        cu_seq_q=cu_seqlens,
                        cu_seq_k=cu_seqlens,
                        max_q=max_seqlen,
                        max_k=max_seqlen,
                    )
                }
            labels_out = labels
        else:
            inputs, labels_out, extra_inputs, extra_kwargs = super().post_dataloading_process(
                input_dict, labels
            )
            # If the dataloader pre-built the attention mask (SDPA doc-mask path),
            # prefer it over the one re-derived by the base class.
            if precomputed_attention_masks is not None:
                extra_kwargs["attention_masks"] = precomputed_attention_masks

        # Apply loss masking if provided by dataloader
        if self.mask_prompt and loss_mask is not None:
            labels_out = labels_out.clone()
            labels_out[loss_mask == 0] = -100
            logger.debug(f"Applied loss mask: {loss_mask.sum().item()} tokens kept for loss")

        return inputs, labels_out, extra_inputs, extra_kwargs
    
    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Override to skip the SDPA-only CP context manager for ring_varlen.

        For ring_varlen the ring kernel handles all cross-rank K/V communication
        internally.  PyTorch's _ContextParallel wrapper (used by the base class
        when cp_enabled) only supports SDPA/flex and would crash or produce wrong
        results here.  We set optional_context_parallel_ctx=None and let the
        already-sharded inputs flow straight into the model forward.
        """
        if self._use_liger:
            return self._forward_backward_step_liger(input_dict, labels)

        if not self._use_ring_varlen_cp:
            # Standard SDPA + CP also needs global-valid normalization: any
            # CP rank whose label slice is all -100 produces 0/0 -> NaN under
            # the default reduction="mean" loss. Replicate the ring_varlen
            # logic but with PyTorch's CP ctx (which the base class would
            # normally create) so SDPA gets K/V-exchanged across CP ranks.
            if self.parallel_dims.cp_enabled:
                return self._forward_backward_step_sdpa_cp(input_dict, labels)
            return super().forward_backward_step(input_dict, labels)

        # --- ring_varlen + CP: replicate base class logic with CP ctx = None ---
        model_parts = self.model_parts
        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        # Normalize the loss by the *global* valid-token count across the CP
        # group rather than the local count. This handles uneven label-mask
        # counts per CP rank (which a local-mean cross_entropy weights wrong).
        # Per-rank grad after backward is then partial_grad_i / global_valid;
        # the CP-SUM grad hook in parallelize._apply_cp_grad_reduce_hooks
        # aggregates these into G / global_valid (FSDP's mesh excludes CP
        # here, so it cannot perform that aggregation itself).
        cp_pg = self.parallel_dims.get_mesh("cp").get_group()
        valid_mask = labels != -100
        global_valid = valid_mask.sum().to(torch.int64)
        dist.all_reduce(global_valid, op=dist.ReduceOp.SUM, group=cp_pg)
        global_valid = global_valid.clamp(min=1)

        # Only fire the CP grad-reduce hook on the last micro-batch of a
        # grad-accum cycle; intermediate micro-batches accumulate partial
        # grads locally so the final SUM gives the correct sum-of-partials.
        is_last_accum = (
            self._ring_accum_step == self.gradient_accumulation_steps - 1
        )
        cp_state = getattr(model_parts[0], "_cp_grad_reduce_state", None)
        if cp_state is not None:
            cp_state["enabled"] = is_last_accum

        # No create_context_parallel_ctx call — ring kernel handles communication.
        with self.train_context(None):
            assert len(model_parts) == 1, "ring_varlen CP does not support PP"
            with self.maybe_enable_amp:
                pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                loss_sum = F.cross_entropy(
                    pred.flatten(0, 1).float(),
                    labels.flatten(0, 1),
                    reduction="sum",
                )
                loss = loss_sum / global_valid
            del pred
            loss.backward()

        self._ring_accum_step = (
            self._ring_accum_step + 1
        ) % self.gradient_accumulation_steps

        return loss

    def _forward_backward_step_sdpa_cp(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """SDPA + CP path with global-valid loss normalization.

        Same as the base class forward_backward_step (PyTorch's
        context_parallel ctx slices inputs/labels/freqs_cis and intercepts
        SDPA for K/V exchange), but loss is computed as
        sum(local) / sum_across_cp(global_valid) instead of local mean.
        Without this, any CP rank whose label slice is all -100 produces
        0/0 = NaN, which propagates through the all-reduce.
        """
        model_parts = self.model_parts
        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        cp_pg = self.parallel_dims.get_mesh("cp").get_group()
        valid_mask = labels != -100
        global_valid = valid_mask.sum().to(torch.int64)
        dist.all_reduce(global_valid, op=dist.ReduceOp.SUM, group=cp_pg)
        global_valid = global_valid.clamp(min=1)

        # Build the standard CP ctx (slices inputs/labels/freqs_cis along seq
        # dim, installs SDPA dispatcher hook for K/V exchange).
        cp_buffers: list[torch.Tensor] = [inputs, labels]
        cp_seq_dims = [1, 1]
        if hasattr(model_parts[0], "freqs_cis"):
            for m in model_parts:
                cp_buffers.append(m.freqs_cis)
            cp_seq_dims += [0 for _ in model_parts]

        cp_mesh = self.parallel_dims.get_mesh("cp")
        cp_ctx = dist_utils.create_context_parallel_ctx(
            cp_mesh=cp_mesh,
            cp_buffers=cp_buffers,
            cp_seq_dims=cp_seq_dims,
            cp_no_restore_buffers={inputs, labels},
            cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
        )

        with self.train_context(cp_ctx):
            assert len(model_parts) == 1, "SDPA CP path does not support PP"
            with self.maybe_enable_amp:
                pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                loss_sum = F.cross_entropy(
                    pred.flatten(0, 1).float(),
                    labels.flatten(0, 1),
                    reduction="sum",
                )
                loss = loss_sum / global_valid
            del pred
            loss.backward()

        # Each CP rank holds a partial fraction of the true loss
        # (loss_sum is local; global_valid spans the CP group). The base
        # class reports dist_mean(loss) over the DP*CP mesh, which would
        # under-report by a factor of CP. SUM across CP so every rank
        # carries the full loss; the DP*CP mean then collapses to mean_DP.
        # Detached: we want this to affect reporting only, not backward.
        loss_for_report = loss.detach().clone()
        dist.all_reduce(loss_for_report, op=dist.ReduceOp.SUM, group=cp_pg)
        return loss_for_report

    def _forward_backward_step_liger(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Liger fused linear+CE path. Passes labels into model.forward so the
        LM head can compute the loss without materializing logits. Bypasses
        self.loss_fn and applies grad-accum rescaling manually.
        """
        model_parts = self.model_parts
        assert len(model_parts) == 1, "Liger path does not support pipeline parallel"

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        with self.train_context(None):
            with self.maybe_enable_amp:
                # Model returns a scalar loss (mean over non-ignored tokens)
                # because labels was passed; LigerLMHead handles the fused CE.
                loss = model_parts[0](
                    inputs, **extra_inputs, **extra_kwargs, labels=labels
                )
                # rescale_accumulated_loss is wrapped on self.loss_fn by the
                # base trainer; we bypass loss_fn here, so apply the same
                # division ourselves to keep grads correctly scaled.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
            loss.backward()

        return loss

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Infinite-cycling wrapper around the parent batch generator.

        The base trainer breaks out of the training loop on DataloaderExhaustedError.
        For SFT we want to cycle through the dataset until the configured number of
        steps is reached, so we catch exhaustion, advance the sampler epoch for
        proper re-shuffling, and restart the iterator.
        """
        epoch = 0
        while True:
            try:
                yield from super().batch_generator(data_iterable)
            except DataloaderExhaustedError:
                epoch += 1
                sampler = getattr(data_iterable, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                logger.info(f"Dataloader epoch {epoch} complete, cycling for next epoch.")

    def log_sft_metrics(self, step: int, loss: float):
        """
        Log SFT-specific metrics.
        
        Args:
            step: Current training step
            loss: Current loss value
        """
        if step % self.job_config.metrics.log_freq == 0:
            logger.info(
                f"SFT Step {step}: loss={loss:.4f}, "
                f"tokens_seen={self.ntokens_seen:,}, "
                f"learning_rate={self.lr_schedulers.get_last_lr()[0]:.2e}"
            )


def create_sft_config_template():
    """
    Print an example SFT configuration template.
    This can be saved as a .toml file and customized.
    """
    template = """
# Supervised Fine-Tuning Configuration Template
# Save this as configs/sft_config.toml and customize for your use case

[job]
dump_folder = "./outputs/sft_run"
description = "Supervised Fine-Tuning with TorchTitan"
custom_config_module = "titan_oellm.configs.oellm_job_config"

[model]
name = "gpt_plus"  # or "qwen3_custom"
flavor = "0.5B"
vocab_size = 50432
# For fine-tuning, load pretrained checkpoint
checkpoint_path = "./checkpoints/pretrained_model"

[training]
dataset = "sft_dataset"
local_batch_size = 4  # Smaller batch size for fine-tuning
seq_len = 2048
steps = 5000  # Fewer steps for fine-tuning
mixed_precision_param = "bfloat16"
mask_prompt = true  # Only compute loss on response tokens

# Gradient accumulation for effective larger batch size
global_batch_size = 32  # effective batch = 32 with accumulation

[data]
# Your SFT dataset configuration
data_prefix = "./data/sft_dataset"
instruction_format = "alpaca"  # or "chatml", "sharegpt"
seed = 42

[optimizer]
name = "AdamW"
lr = 2.0e-5  # Lower learning rate for fine-tuning
weight_decay = 0.01

[lr_scheduler]
name = "cosine"
warmup_steps = 100  # Short warmup for fine-tuning
min_lr = 1.0e-6

[checkpoint]
enable = true
interval_type = "steps"
interval = 500
keep_latest_k = 3
checkpoint_dir = "./checkpoints/sft"

[metrics]
log_freq = 10
enable_tensorboard = true
save_tb_folder = "tb_sft"

[validation]
enable = true
interval = 500
"""
    print(template)


def main_sft():
    """Main entry point for SFT training."""
  
    
    init_logger()

    # Ensure single-process defaults when not launched via torchrun
    if "LOCAL_RANK" not in os.environ:
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    # Add rank-0-only filter to all loggers to suppress duplicate logs in distributed training
    rank_filter = RankZeroFilter()
    for handler in logging.root.handlers:
        handler.addFilter(rank_filter)
    # Also add to the logger instance if it has handlers
    for handler in logger.handlers:
        handler.addFilter(rank_filter)
    
    # Check if user wants to see config template
    if len(sys.argv) > 1 and sys.argv[1] in ['--template', '--help-config']:
        create_sft_config_template()
        sys.exit(0)
    
    logger.info("=" * 80)
    logger.info("Starting Supervised Fine-Tuning (SFT) with TorchTitan")
    logger.info("=" * 80)
    
    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    # Create config manager and explicitly pass sys.argv[1:] at runtime
    config_manager = ConfigManager()
    config = config_manager.parse_args(sys.argv[1:])

    # Seed numpy + python random in addition to torch/DTensor (which torchtitan's
    # set_determinism handles). Closes the gap for any dataloader path using
    # numpy/random.shuffle.
    if config.debug.deterministic and config.debug.seed is not None:
        import random
        import numpy as np
        random.seed(config.debug.seed)
        np.random.seed(config.debug.seed)
        logger.info(f"Seeded python random + numpy with {config.debug.seed}")

    trainer: SFTTrainer | None = None

    try:
        trainer = SFTTrainer(config)

        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main_sft()
