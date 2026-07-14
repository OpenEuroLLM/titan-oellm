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
from torchtitan.config.job_config import Checkpoint as BaseCheckpoint


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

    data_prefix: list[str] | str = ""
    """Data prefix path(s) for MMapDataset, or chunk directory(ies) for chunk-based dataloaders."""

    chunks_dir: list[str] | str = ""
    """Directory(ies) containing data chunks for ChunkedMMapDataset / DeterministicPackedDataset"""

    dataloader: str = "MMapDataset"
    """Type of dataloader to use: 'MMapDataset', 'DeterministicPackedDataset',
       'BestFitPackedDataset', or 'ChunkedMMapDataset'"""

    seed: int = 42
    """Random seed for data loading"""

    best_fit_buffer_size: int = 32
    """Fragment buffer size (BST/FIFO) for BestFitPackedDataset best-fit packing.
       Larger = tighter packing but slower plan build; >32 gives negligible gain
       at much higher build cost. Only used when dataloader='BestFitPackedDataset'."""

    bfp_cache_dir: str = ""
    """Shared cache dir for BestFitPackedDataset packing-plan files. Empty →
       <chunks_dir>/.packing_cache. Set to a writable shared path so a login-node
       prebuild and the multi-node job reuse the same plan (read-only dataset dirs)."""

    mask_eot_loss: bool = False
    """If True, mask EOS/EOT tokens from the loss (like Megatron's eod_mask_loss)."""

    eos_id: int | None = -1
    """EOS token ID inserted between documents during packing.
       -1 (default): use tokenizer.eos_id (backward-compatible behaviour).
       null / None: do NOT insert any EOS between documents (matches Megatron packing).
       >= 0: use this specific token ID."""


@dataclass
class ValidationDataset:
    """Validation dataset override configuration (optional fields override Validation defaults)."""

    name: str = ""
    """Name for metrics/logging (if empty, auto-assigned)."""

    data_prefix: str | None = None
    local_batch_size: int | None = None
    max_eval_samples: int | None = None
    dataloader: str | None = None
    data_source: str | None = None
    split_samples: int | None = None
    eval_mode: str | None = None
    steps: int | None = None

    best_fit_buffer_size: int = 500
    """Tree size for BestFitPackedDataset packing (default: 500)"""


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

    # fix to none as always warm_steps / warm_ratio should be used
    warmup_steps: int | None = None

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

    lr_steps: int | None = None
    """
    Number of steps to use as the LR scheduler's total training duration.
    If set, overrides ``training.steps`` for the scheduler only, so the training
    loop runs for ``training.steps`` while the LR is planned over ``lr_steps``.
    Useful for cooldown stages that resume from a checkpoint (self.step = start_step)
    and run total_steps - start_step iterations, but need the LR to decay fully
    over exactly decay_steps = total_steps - start_step.
    """

    # ── Ornstein-Uhlenbeck stochastic LR (scheduler_type="universal_ou") ──────
    # Opt into the "universal_ou" scheduler: the three-phase universal curve plus
    # a mean-reverting random walk of the LR during the main phase.
    use_ou_process: bool = False
    """Enable the OU mean-reverting random walk in the main phase (also enabled
       implicitly when scheduler_type == "universal_ou")."""
    ou_theta: float = 0.008
    """OU mean-reversion strength (pull back toward the base LR)."""
    ou_sigma: float = 0.1
    """OU noise scale (per-step diffusion)."""
    ou_max_change: float = 0.05
    """Clamp on the per-step multiplicative LR change."""
    ema_alpha: float = 0.99
    """EMA smoothing of the OU factor."""
    ou_seed: int = 1
    """Seed for the OU process (checkpointed for reproducible resume)."""
    ou_decay_type: str = "const"
    """Underlying decay applied around the OU fluctuation in the main phase."""
    ou_min_ratio: float = 0.2
    """Target LR ratio for the OU underlying decay."""


@dataclass
class Checkpoint(BaseCheckpoint):

    extra_steps: list[int] = field(default_factory=list)
    """Additional specific steps at which to save a checkpoint, regardless of interval."""

    initial_step: int = -1
    """Override the step counter after loading a model-only checkpoint.
    When >= 0 and initial_load_model_only is true, the train_state step
    will be set to this value after loading. Useful for running validation
    at the correct step when the final checkpoint was saved model-only."""


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

    # Attention Gating parameters (gpt_plus)
    attn_gate_type: str = "none"
    """Attention gate type: 'none', 'scalar', 'elementwise_dense', 'elementwise_lowrank'"""
    attn_gate_input: str = "x"
    """Gate input: 'x' (input to attention) or 'xv' (value vector)"""
    attn_gate_activation: str = "sigmoid"
    """Activation: 'sigmoid' or 'tanh_sq'"""
    attn_gate_lowrank_dim: int = 64
    """Low-rank dimension for elementwise_lowrank gate type"""
    attn_gate_bias: bool = True
    """Whether to use bias in gate linear layers"""

    # Qwen3-specific parameters (None = use flavor value)
    qk_norm: bool | None = None
    """Enable QK normalization in attention (Qwen3). None = use flavor value."""
    rope_theta: float | None = None
    """RoPE theta value for rotary embeddings (Qwen3). None = use flavor value."""

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
    fp8_activations: bool = False
    """Store attention/FFN pre-projection activations in FP8 E4M3 on the backward
       tape (halves activation memory; STE backward). Training-time only, pure torch."""


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
class ConstrainedAdamOptimizer:
    """ConstrainedAdam (CPR, Franke et al. arXiv:2311.09058) configuration.

    Selected via [optimizer].name = "constrained_adam". Two modes:
    - "bounded"    : each weight row constrained to ‖w_i‖ ≤ max_norm (CPR upper
                     bound). Safe on standard qwen3 — a regularizer, not a
                     reparametrization. **Default here.**
    - "normalized" : each weight row forced to unit norm ‖w_i‖ = 1 (unit sphere,
                     for hyperspherical/nGPT-style models).
    """

    mode: str = "bounded"
    """Constraint mode: "bounded" (CPR, default) or "normalized" (unit sphere)."""

    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_norm: float = 1.0
    """Upper bound on each weight row's L2 norm (bounded mode)."""
    delta: float = 0.0
    """Only project rows within delta of the boundary (bounded mode)."""
    project_momentum: bool = True
    parallel_transport: bool = True
    fallback_lr: float | None = None
    """LR for scalar (1D) params. None → optimizer.lr."""
    embedding_lr: float | None = None
    """LR for embedding params. None → fallback_lr."""
    embedding_norm: bool = False
    """Apply GPTNormalizer to embeddings post-step. Default False for qwen3
       (nGPT-style embedding normalization is off unless you want it)."""
    head_lr_mult: float = 1.0
    embed_lr_mult: float = 1.0
    scaler_weight_decay: float = 0.0


@dataclass
class AdamCPROptimizer:
    """AdamCPR — reference Constrained Parameter Regularization (arXiv:2311.09058,
    github.com/automl/CPR). Selected via [optimizer].name = "adam_cpr".

    Replaces weight decay with a hard per-parameter constraint on a regularization
    statistic (default: squared L2 norm), enforced by an augmented-Lagrangian
    update. The bound κ is set automatically by default (inflection_point).
    """

    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    kappa_init_method: str = "inflection_point"
    """κ init: "inflection_point" (auto), "warm_start", "dependent", "uniform"."""
    kappa_init_param: float = 1000.0
    """Meaning depends on kappa_init_method: warmup steps (warm_start) /
       scale factor (dependent) / fixed bound (uniform)."""
    reg_function: str = "l2"
    """Regularization statistic: "l2", "l1", "std", or "huber"."""
    kappa_update: float = 1.0
    """Lagrange-multiplier step size μ."""
    reg_step_size: int = 200
    """Sampling cadence for inflection-point detection."""
    reg_ema_decay: float = 0.99
    """EMA decay for the statistic in inflection-point mode."""
    reg_embedding: bool = False
    """Also regularize embedding weights (default: exclude)."""
    reg_by_lr: bool = False
    """Scale the constraint pullback by the current lr."""
    amsgrad: bool = False


@dataclass
class BoundedSphericalAdamOptimizer:
    """BoundedSphericalAdam (BSA) configuration.

    Selected via [optimizer].name = "bounded_spherical_adam". Tangent-projected
    Adam with a predictive constraint on each weight row. Modes:
    - "bounded"                    : ‖w_row‖ ≤ max_norm (ball with boundary).
    - "normalized"                 : ‖w_row‖ = 1 (unit sphere).
    - "partial_orthogonal"         : Newton-Schulz row-decorrelation.
    - "partial_orthogonal_bounded" : decorrelation + norm bound.
    Default "bounded" (CPR-like, safe on standard qwen3).
    """

    mode: str = "bounded"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_norm: float = 1.0
    project_gradients: bool = True
    correct_v: bool = False
    soft_blend: bool = False
    rotation_lr: float | None = None
    fallback_lr: float | None = None
    embedding_lr: float | None = None
    n_iter_spectral: int = 3
    n_iter_ns: int = 5
    n_iter: float = 1.0
    ffn_down_left_ns: bool = False
    ns_alpha: float = 1.0
    ns_mode: str = "full"
    kappa_target: float = 4.0
    lambda_max: float = 0.05
    kappa_ema_beta: float = 0.99
    ns_schedule_steps: int = 4000
    out_norm_dim_0: bool = False


@dataclass
class SharedBoundMuonOptimizer:
    """SharedBoundMuon config (base container for BoundedMuon).

    BSA gradient projection + optional Muon NS on the gradient direction. Used as
    the parent of BoundedMuon; selectable directly via
    [optimizer].name = "shared_bound_muon".
    """

    mode: str = "bounded"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_norm: float = 1.0
    project_gradients: bool = True
    soft_blend: bool = False
    rotation_lr: float | None = None
    fallback_lr: float | None = None
    embedding_lr: float | None = None
    n_iter_spectral: int = 3
    n_iter_ns: int = 5
    n_iter: float = 1.0
    ffn_down_left_ns: bool = False
    ns_alpha: float = 1.0
    ns_mode: str = "full"
    kappa_target: float = 4.0
    lambda_max: float = 0.05
    kappa_ema_beta: float = 0.99
    ns_schedule_steps: int = 4000
    out_norm_dim_0: bool = False
    muon_on_gradient: bool = False
    muon_ns_steps: int = 5
    muon_ns_mode: str = "muon"
    muon_preserve_norm: bool = True
    muon_ns_dtype: str = "bfloat16"


@dataclass
class BoundedMuonOptimizer:
    """BoundedMuon config — canonical "true Muon" (Newton-Schulz on the momentum
    buffer, pure SGD update) + BSA row-norm projection.

    Selected via [optimizer].name = "bounded_muon" (or "muon"). Works on standard
    transformers; the post-step row-norm constraint is aggressive on unnormalized
    weights — sweep max_norm carefully.
    """

    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_norm: float = 1.0
    project_gradients: bool = True
    soft_blend: bool = False
    out_norm_dim_0: bool = False
    rotation_lr: float | None = None
    fallback_lr: float | None = None
    embedding_lr: float | None = None
    muon_beta1: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_ns_mode: str = "muon"
    """NS coeff mode: "muon", "polar_express", "gram_polar_express", "convergent",
       "cubic", or a "dist_*" FSDP2 distributed-NS mode."""
    muon_scale: float = 0.2
    muon_norm_preserve: bool = False
    muon_bias_correction: bool = False
    muon_geodesic: bool = False
    muon_adam_scale: bool = False
    muon_flat_scale: bool = False
    reprojection_interval: int = 1


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
    checkpoint: Checkpoint = field(default_factory=Checkpoint)  # Override to add extra_steps
    validation: Validation = field(default_factory=Validation)  # Override base Validation with custom implementation
    compile: Compile = field(default_factory=Compile)  # Override to add mode

    # Custom titan-oellm fields
    data: SciData = field(default_factory=SciData)
    sci_tokenizer: SciTokenizer = field(default_factory=SciTokenizer)
    parameter_logging: ParameterLogging = field(default_factory=ParameterLogging)
    benchmarks: Benchmarks = field(default_factory=Benchmarks)

    # Custom optimizer sections (active one chosen by [optimizer].name).
    # These hold the extra hyperparameters; the base [optimizer] section still
    # provides name/lr/betas/weight_decay.
    constrained_adam: ConstrainedAdamOptimizer = field(default_factory=ConstrainedAdamOptimizer)
    adam_cpr: AdamCPROptimizer = field(default_factory=AdamCPROptimizer)
    bounded_spherical_adam: BoundedSphericalAdamOptimizer = field(default_factory=BoundedSphericalAdamOptimizer)
    shared_bound_muon: SharedBoundMuonOptimizer = field(default_factory=SharedBoundMuonOptimizer)
    bounded_muon: BoundedMuonOptimizer = field(default_factory=BoundedMuonOptimizer)

    def __post_init__(self) -> None:
        apply_output_dir_prefix(self)
        apply_continue_training(self)