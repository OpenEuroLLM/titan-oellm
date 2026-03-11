# Copyright (c) Titan-OELLM Custom Components.

from dataclasses import dataclass, field
import os
import re
from typing import Literal

from torchtitan.config import (
    JobConfig as BaseJobConfig,
    LRScheduler as BaseLRScheduler,
    Model as BaseModel,
)
from torchtitan.config.job_config import (
    Compile as BaseCompile,
    Job as BaseJob,
    Parallelism as BaseParallelism,
    Training as BaseTraining,
)
from torchtitan.tools.logging import logger


@dataclass
class Job(BaseJob):
    """Extended Job config."""

    continue_training: bool = False
    """Resume from latest checkpoint in dump_folder if available."""


@dataclass
class SciData:
    """SciData configuration for sci_dataloader."""

    local_batch_size: int = 8
    """Local batch size (per device)"""

    seq_len: int = 2048
    """Sequence length"""

    min_doc_len: int = 10
    """Minimum document length"""

    data_prefix: str = ""
    """Data prefix path for MMapDataset"""

    chunks_dir: str = ""
    """Directory containing data chunks for ChunkedMMapDataset"""

    dataloader: str = "MMapDataset"
    """Type of dataloader to use: 'MMapDataset', 'DeterministicPackedDataset', or 'ChunkedMMapDataset'"""

    seed: int = 42
    """Random seed for data loading"""


@dataclass
class Validation:
    """Validation configuration for evaluation during training."""

    enable: bool = False
    """Enable validation during training (aligned with torchtitan v0.2.0)"""

    freq: int = 100
    """Frequency of validation evaluation during training (aligned with torchtitan v0.2.0)"""

    steps: int = -1
    """Number of validation steps (-1 = full validation set, >0 = limit steps)"""

    data_prefix: str = ""
    """Data prefix path for validation MMapDataset"""

    local_batch_size: int = -1
    """Local batch size for validation (per device). If -1, uses training.local_batch_size"""

    max_eval_samples: int = -1
    """Total validation samples across all workers. -1 = full validation set, >0 = limit to N samples total"""

    dataloader: str = "MMapDataset"
    """Type of dataloader to use for validation: 'MMapDataset' or 'ChunkedMMapDataset'"""

    # Validation data source configuration
    data_source: str = "offline"
    """Validation data source: 'offline' (separate files via data_prefix) or 'split' (hold out from training data)"""

    split_samples: int = 0
    """Number of samples to hold out from training data end for validation (only used when data_source='split')"""

    split_seed: int = 42
    """Seed for deterministic split (ensures same split across runs)"""

    # Evaluation mode configuration
    eval_mode: str = "concatenated"
    """Evaluation mode: 'concatenated' (default, documents concatenated into fixed-length sequences) or
    'document' (each document evaluated independently with padding/truncation for comparable metrics)"""

    datasets: list[ValidationDataset] = field(default_factory=list)
    """
    Optional list of additional validation dataset configurations.
    If provided, each entry runs its own validation pass and logs metrics separately.
    """

@dataclass
class Benchmarks:
    """Standard benchmark evaluation configuration (WikiText, LAMBADA)."""

    enable: bool = True
    """Enable standard benchmark evaluation during validation"""

    wikitext2_path: str = ""
    """Path prefix to WikiText-2 dataset in bin/idx format (use download_benchmarks.py to create)"""

    wikitext103_path: str = ""
    """Path prefix to WikiText-103 dataset in bin/idx format (use download_benchmarks.py to create)"""

    lambada_path: str = ""
    """Path prefix to LAMBADA dataset in bin/idx format (use download_benchmarks.py to create)"""

    batch_size: int = 8
    """Batch size for benchmark evaluation"""


@dataclass
class SciTokenizer:
    """SciTokenizer configuration."""

    default_add_bos: bool = True
    """Default behavior for adding BOS tokens when encoding"""

    default_add_eos: bool = False
    """Default behavior for adding EOS tokens when encoding"""

    override_hf_bos_behavior: bool = False
    """Override HuggingFace tokenizer's automatic BOS behavior"""

    override_hf_eos_behavior: bool = False
    """Override HuggingFace tokenizer's automatic EOS behavior"""


@dataclass
class ParameterLogging:
    """Parameter statistics logging configuration."""

    enabled: bool = False
    """Enable parameter statistics logging to TensorBoard"""

    log_interval: int = 100
    """Interval (in steps) between parameter statistics logging"""

    log_parameters: bool = True
    """Log parameter statistics (max, min, norm, std)"""

    log_gradients: bool = True
    """Log gradient statistics (max, min, norm, std)"""

    log_optimizer_states: bool = True
    """Log AdamW optimizer state statistics"""

    log_gates: bool = False
    """Log MoE gate statistics (only applicable for MoE models)"""

    include_patterns: list[str] = field(default_factory=list)
    """Include only parameters matching these patterns (empty = include all)"""

    exclude_patterns: list[str] = field(default_factory=list)
    """Exclude parameters matching these patterns"""


@dataclass
class LRScheduler(BaseLRScheduler):
    """Extended LR Scheduler config with unified parameters for all scheduler types.

    Supports multiple scheduler types:
    - wsd/wdd/cosine: Traditional warmup-based schedulers
    - universal: 3-phase scheduler (warm → main → cooldown)
    """

    # Scheduler type selection
    scheduler_type: str = "wsd"
    """
    Type of learning rate scheduler to use:
    - "wsd": Warmup-Stable-Decay (default torchtitan scheduler)
    - "wdd": Warmup-Decay with Stable Decay
    - "cosine": Cosine annealing with warmup
    - "universal": 3-phase scheduler (warm → main → cooldown)
    """

    # NOTE: warmup_steps is inherited from BaseLRScheduler (default=200)
    # NOTE: decay_type is inherited from BaseLRScheduler (default="linear")
    # NOTE: lr_min is inherited from BaseLRScheduler (default=0.0)

    decay_ratio: float | None = None
    """
    Controls the proportion of remaining training steps (after warmup) allocated to decay/annealing.

    - None: All remaining steps after warmup are used for decay (no stable phase)
    - 0.0: No decay phase (flat after warmup)
    - 0.3: Last 30% of remaining steps for decay, 70% stable
    - 1.0: Immediate decay after warmup (no stable phase)
    """

    stable_decay_ratio: float = 0.0
    """
    Controls the amount of decay during the stable phase (WDD scheduler only).
    - 0.0 (default): No decay during stable phase (pure WSD behavior)
    - 0.1: Decay to 90% of max LR during stable phase
    """

    lr_min_ratio: float | None = None
    """
    Minimum learning rate as a ratio of the base learning rate.
    For example, lr_min_ratio=0.1 with base_lr=1e-3 results in minimum LR of 1e-4.
    """

    lr_min_absolute: float | None = None
    """
    Absolute minimum learning rate value.
    For example, lr_min_absolute=1e-5 will always result in minimum LR of 1e-5.
    """

    # Universal scheduler parameters (scheduler_type="universal")
    # Phase 1 - Warm phase
    warm_steps: int = 0
    """
    Absolute number of warm steps (universal scheduler).
    Mutually exclusive with warm_ratio.
    """

    warm_ratio: float = 0.0
    """
    Warm steps as ratio of total training steps (universal scheduler).
    Mutually exclusive with warm_steps.
    """

    warm_direction: str = "up"
    """
    Direction of warm phase (universal scheduler).
    - "up": Warmup from lr_min_absolute to base_lr
    - "down": Warmdown from warm_start_ratio × base_lr to base_lr
    """

    warm_type: str = "linear"
    """
    Decay curve type for warm phase: "linear", "cosine", "sqrt", "exp"
    """

    warm_start_ratio: float = 2.0
    """
    Starting LR multiplier for warmdown (only when warm_direction="down").
    """

    # Phase 2 - Main phase
    main_decay_type: str = "const"
    """
    Base schedule decay type for main phase (universal scheduler).
    - "const": Constant base LR throughout main phase
    - "linear", "cosine", "exp", "sqrt": Decay to main_decay_ratio × base_lr
    """

    main_decay_ratio: float = 0.2
    """
    Target LR ratio at end of main phase (universal scheduler).
    Only used when main_decay_type is not "const".
    """

    # Phase 3 - Cooldown phase
    cooldown_steps: int = 0
    """
    Cooldown phase duration in steps (universal scheduler).
    Mutually exclusive with cooldown_ratio.
    """

    cooldown_ratio: float = 0.0
    """
    Cooldown steps as ratio of total training steps (universal scheduler).
    Mutually exclusive with cooldown_steps.
    """

    cooldown_type: str = "cosine"



@dataclass
class Compile(BaseCompile):
    """Extended Compile config with mode support."""

    mode: str | None = None
    """torch.compile mode: 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'.
    None uses standard compilation without a specific mode."""


@dataclass
class Model(BaseModel):
    """Extended Model config with norm_gpt and gpt_plus specific arguments."""

    # Tokenizer configuration
    tokenizer_path: str = ""
    """Path to tokenizer directory (typically injected by cluster_config)"""

    # Vocabulary size (must match tokenizer)
    vocab_size: int = 50432
    """Vocabulary size for embedding and output layers (default: 50432). Must match tokenizer vocab size."""

    # MLP configuration
    mlp_layers: int = 2
    """Number of layers in MLP feedforward (2, 3, or 4)"""
    mlp_activation: str = "swiglu"
    """MLP activation type: 'swiglu' (gated) or 'silu' (ungated)"""

    # gpt_plus specific model arguments (QKNormPlus)
    qk_norm_type: str = "QKNormPlus"
    """QK normalization type: 'QKNormPlus' or 'RMSNorm'"""

    qk_norm_scale_mode: str = "scalar"
    """QKNormPlus scaling mode: 'scalar', 'head_dim', 'n_heads', 'matrix' or with '_pos' suffix for position-dependent"""

    # Embedding/head weight tying
    tie_embedding: bool = False
    """Tie embedding and output head weights (reduces parameters by vocab_size × dim)"""

    # FlexAttention parameters
    use_flex_attn: bool = False
    """Enable FlexAttention for document-aware masking"""
    attn_mask_type: str = "causal"
    """Attention mask type: 'causal' (standard) or 'block_causal' (document-aware)"""

    # Flash Attention parameters
    use_flash_attn: bool = False
    """Enable direct Flash Attention 2/3 (auto-selects FA3 on Hopper, FA2 otherwise)"""

    # Qwen3-specific parameters
    qk_norm: bool = True
    """Enable QK normalization in attention (Qwen3)"""
    rope_theta: float = 1000000
    """RoPE theta value for rotary embeddings (Qwen3)"""

    # RoPE scaling parameters (gpt_plus and other models supporting long context)
    rope_scaling_factor: float = 8.0
    """RoPE scaling multiplier for extending context length (default: 8.0)"""
    rope_low_freq_factor: float = 1.0
    """Scaling for low-frequency (long-wavelength) RoPE bands (default: 1.0)"""
    rope_high_freq_factor: float = 4.0
    """Scaling for high-frequency (short-wavelength) RoPE bands (default: 4.0)"""
    rope_original_max_position_embeddings: int = 8192
    """Original maximum position embeddings before scaling (default: 8192)"""

    head_dim: int = 128
    """Dimension per attention head (Qwen3)"""
    hidden_dim: int = 3072
    """FFN hidden dimension (Qwen3)"""
    norm_eps: float = 1e-6
    """Layer normalization epsilon (Qwen3)"""
    depth_init: bool = True
    """Use depth-dependent initialization (Qwen3)"""
    enable_weight_tying: bool = False
    """Tie embedding and output head weights (Qwen3)"""
    moe_enabled: bool = False
    """Enable Mixture of Experts (Qwen3 MoE variants)"""
    moe_inter_dim: int = 768
    """MoE intermediate dimension (Qwen3 MoE variants)"""


@dataclass
class Training(BaseTraining):
    """Extended Training config with MoE-specific fields."""

    debug_moe_force_load_balance: bool = False
    """Force load balancing for MoE debugging (only applicable for MoE models)"""


@dataclass
class Parallelism(BaseParallelism):
    """Extended Parallelism config with enable_compiled_autograd field."""

    enable_compiled_autograd: bool = False
    """Enable compiled autograd for improved performance"""


def apply_continue_training(job_config: "JobConfig") -> None:
    """Enable resume-from-checkpoint if requested and a checkpoint exists."""
    try:
        if not getattr(job_config.job, "continue_training", False):
            return

        checkpoint_folder = job_config.checkpoint.folder
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(job_config.job.dump_folder, checkpoint_folder)

        if not os.path.isdir(checkpoint_folder):
            logger.info(
                "continue_training is enabled but checkpoint folder does not exist: %s",
                checkpoint_folder,
            )
            return

        pattern = re.compile(r"step-(\d+)")
        steps: list[int] = []
        for entry in os.listdir(checkpoint_folder):
            match = pattern.search(entry)
            if not match:
                continue
            step_dir = os.path.join(checkpoint_folder, entry)
            if not os.path.isdir(step_dir):
                continue
            if os.path.isfile(os.path.join(step_dir, ".metadata")) or os.path.isfile(
                os.path.join(step_dir, "model.safetensors.index.json")
            ):
                steps.append(int(match.group(1)))

        if not steps:
            logger.info(
                "continue_training is enabled but no checkpoints found in %s",
                checkpoint_folder,
            )
            return

        job_config.checkpoint.enable = True
        if job_config.checkpoint.load_step == -1:
            job_config.checkpoint.load_step = -1

        logger.info(
            "continue_training enabled: will load latest checkpoint from %s",
            checkpoint_folder,
        )
    except Exception as exc:
        logger.warning("Failed to apply continue_training: %s", exc)


def apply_output_dir_prefix(job_config: "JobConfig") -> None:
    """Prefix dump_folder with OUTPUT_DIR if provided.

    Logic:
    - If dump_folder starts with "./" (explicit relative path), use as-is relative to CWD
    - If dump_folder starts with "/" (absolute path), use as-is
    - Otherwise (implicit relative like "path/..."), prefix with OUTPUT_DIR
    """
    try:
        import sys

        # CRITICAL FIX: Check if CLI is overriding dump_folder
        # If --job.dump_folder is in CLI args and starts with "./", skip prefixing
        # This handles the case where TOML has "qwen3_custom" but CLI has "./outputs/..."
        for arg in sys.argv[1:]:
            if arg.startswith("--job.dump_folder=") or arg.startswith("--job.dump-folder="):
                cli_value = arg.split("=", 1)[1]
                if cli_value.startswith("./") or cli_value.startswith("/"):
                    logger.debug(
                        "Skipping OUTPUT_DIR prefix due to CLI override: %s",
                        cli_value,
                    )
                    return

        output_dir = os.environ.get("OUTPUT_DIR", "").strip()
        if not output_dir:
            return

        dump_folder = job_config.job.dump_folder
        if not dump_folder:
            return

        # If dump_folder starts with "./", user is explicitly specifying relative path
        # Respect that choice and don't apply OUTPUT_DIR prefix
        if dump_folder.startswith("./"):
            logger.debug(
                "Skipping OUTPUT_DIR prefix for explicitly relative dump_folder: %s",
                dump_folder,
            )
            return

        # If dump_folder is an absolute path, use as-is
        if dump_folder.startswith("/"):
            logger.debug(
                "Skipping OUTPUT_DIR prefix for absolute dump_folder: %s",
                dump_folder,
            )
            return

        # For implicit relative paths (e.g., "path/..."), apply OUTPUT_DIR prefix
        job_config.job.dump_folder = os.path.join(output_dir, dump_folder)
        logger.info(
            "Prefixed dump_folder with OUTPUT_DIR: %s",
            job_config.job.dump_folder,
        )
    except Exception as exc:
        logger.warning("Failed to apply OUTPUT_DIR prefix: %s", exc)

@dataclass
class JobConfig(BaseJobConfig):
    """Extended JobConfig with SciData, SciTokenizer, Normalizer, ParameterLogging, and custom Validation.

    Inherits all standard torchtitan fields (job, metrics, optimizer, training, parallelism,
    checkpoint, activation_checkpoint, compile, quantize, experimental, etc.) from BaseJobConfig
    and adds titan-oellm custom fields.
    """

    # Override base fields with extended versions
    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    parallelism: Parallelism = field(default_factory=Parallelism)
    validation: Validation = field(default_factory=Validation)  # Override base Validation with custom implementation
    compile: Compile = field(default_factory=Compile)  # Override to add mode

    # Custom titan-oellm fields
    data: SciData = field(default_factory=SciData)
    sci_tokenizer: SciTokenizer = field(default_factory=SciTokenizer)
    parameter_logging: ParameterLogging = field(default_factory=ParameterLogging)
    benchmarks: Benchmarks = field(default_factory=Benchmarks)

    def __post_init__(self) -> None:
        apply_output_dir_prefix(self)
        apply_continue_training(self)