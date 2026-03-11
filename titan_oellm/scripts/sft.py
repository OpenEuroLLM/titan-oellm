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
from typing import Any

import torchtitan
from torchtitan.train import Trainer
from torchtitan.config import JobConfig, ConfigManager
from torchtitan.tools.logging import logger, init_logger


class RankZeroFilter(logging.Filter):
    """Logging filter that only allows logs from rank 0 in distributed training."""
    
    def filter(self, record):
        """Only allow logs from rank 0."""
        rank = int(os.environ.get("RANK", "0"))
        return rank == 0

# Import model specs to register them
import titan_oellm.models.gpt_plus  # noqa: F401
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
        # Extract and remove loss_mask from input_dict before parent processing
        # to prevent it from being passed to the model
        loss_mask = input_dict.pop('loss_mask', None)
        
        # Validate input tensors before calling parent
        inputs_raw = input_dict.get("input")
        if inputs_raw is not None:
            assert inputs_raw.dtype in [torch.long, torch.int64], f"Input dtype must be long, got {inputs_raw.dtype}"
            assert inputs_raw.min() >= 0, f"Input tokens must be non-negative, got min={inputs_raw.min()}"
            assert inputs_raw.max() < self.tokenizer.vocab_size, f"Input tokens exceed vocab size {self.tokenizer.vocab_size}, got max={inputs_raw.max()}"
        
        # Call parent class processing
        inputs, labels_out, extra_inputs, extra_kwargs = super().post_dataloading_process(
            input_dict, labels
        )
        
        # Apply loss masking if provided by dataloader
        if self.mask_prompt and loss_mask is not None:
            # Set masked positions to -100 (standard ignore index for cross entropy)
            labels_out = labels_out.clone()
            labels_out[loss_mask == 0] = -100
            logger.debug(f"Applied loss mask: {loss_mask.sum().item()} tokens kept for loss")
        
        return inputs, labels_out, extra_inputs, extra_kwargs
    
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
custom_config_module = "titan_oellm.configs.sci_job_config"

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
