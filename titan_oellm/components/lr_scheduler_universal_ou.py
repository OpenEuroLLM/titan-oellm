# Copyright (c) Titan-Sci Custom Components.
# All rights reserved.
#
# Universal OU Stochastic Learning Rate Scheduler for TorchTitan

"""
Universal OU Stochastic Learning Rate Scheduler - Unified Scheduler

This module implements a universal learning rate scheduler that replaces all other
LR schedulers through a flexible three-phase architecture with optional OU stochastic process.

The scheduler has three configurable phases:
1. Phase 1 (Warm): Optional warm-up or warm-down with bidirectional control
2. Phase 2 (Main): Base schedule with optional OU stochastic process
3. Phase 3 (Cooldown): Final annealing to minimum LR

Key features:
- Unified Parameter System: Consistent naming across all phases
- Bidirectional Warm Phase: warm_direction="up" (warmup) or "down" (warmdown)
- Optional OU Process: Controlled by use_ou_process flag
- Flexible Duration: Absolute steps or ratio of total steps
- Universal Decay Curves: All phases support linear, cosine, exp, sqrt
- State Persistence: Full support for checkpointing and resuming

New Parameter System:

Phase 1 - Warm:
  - warm_steps/warm_ratio: Duration (mutually exclusive)
  - warm_direction: "up" or "down"
  - warm_type: "linear", "cosine", "exp", "sqrt"
  - warm_start_ratio: Starting LR multiplier (for warmdown)

Phase 2 - Main:
  - use_ou_process: Enable/disable OU stochastic process
  - main_decay_type: "const", "linear", "cosine", "exp", "sqrt"
  - main_decay_ratio: Target LR ratio at end of main phase
  - OU params: ou_theta, ou_sigma, ou_max_change, ema_alpha, ou_seed

Phase 3 - Cooldown:
  - cooldown_steps/cooldown_ratio: Duration (mutually exclusive)
  - cooldown_type: "linear", "cosine", "exp", "sqrt"

Global:
  - lr_min_absolute: Absolute minimum LR floor

Example usage:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = build_lr_schedulers_universal_ou(
        optimizers_container,
        job_config  # Config with universal_ou parameters in lr_scheduler section
    )
"""

import copy
import math
from typing import Any, Dict, List

from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import LRScheduler as LRSchedulerConfig
from torchtitan.tools.logging import logger

# Import the base universal OU scheduler implementation
from titan_oellm.components.universal_ou_stochastic_lr import UniversalOUStochasticLR

__all__ = [
    "UniversalOULRSchedulersContainer",
    "build_lr_schedulers_universal_ou",
    "build_lr_schedulers_auto",  # Factory function for backward compatibility
]


class UniversalOULRSchedulersContainer(Stateful):
    """Container for multiple Universal OU stochastic learning rate schedulers.

    This container wraps multiple UniversalOUStochasticLR schedulers and provides a unified
    interface compatible with TorchTitan's training loop.

    Key Features:
    - Uses UniversalOUStochasticLR for maximum flexibility
    - Preserves stochastic state (OU process, EMA state) across checkpoints
    - Each scheduler maintains independent stochastic trajectories
    - Supports all scheduler types through unified parameter system

    Checkpointing:
    All schedulers share the same configuration but maintain independent stochastic
    states. During checkpointing, we save all scheduler states to preserve the
    exact stochastic trajectories.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
        lr_scheduler_config (LRSchedulerConfig): LR scheduler configuration.
        training_steps (int): Total number of training steps.
    """

    def __init__(
        self,
        optimizers: OptimizersContainer,
        lr_scheduler_config: LRSchedulerConfig,
        training_steps: int
    ) -> None:
        assert len(optimizers) > 0, "Must have at least one optimizer to create LRScheduler"

        # Get universal_ou parameters with defaults

        # Phase 1 - Warm
        warm_steps = getattr(lr_scheduler_config, 'warm_steps', 0)
        warm_ratio = getattr(lr_scheduler_config, 'warm_ratio', 0.0)
        warm_direction = getattr(lr_scheduler_config, 'warm_direction', 'up')
        warm_type = getattr(lr_scheduler_config, 'warm_type', 'linear')
        warm_start_ratio = getattr(lr_scheduler_config, 'warm_start_ratio', 2.0)

        # Phase 2 - Main
        use_ou_process = getattr(lr_scheduler_config, 'use_ou_process', False)
        main_decay_type = getattr(lr_scheduler_config, 'main_decay_type', 'const')
        main_decay_ratio = getattr(lr_scheduler_config, 'main_decay_ratio', 0.2)

        # OU parameters
        ou_theta = getattr(lr_scheduler_config, 'ou_theta', 0.008)
        ou_sigma = getattr(lr_scheduler_config, 'ou_sigma', 0.1)
        ou_max_change = getattr(lr_scheduler_config, 'ou_max_change', 0.05)
        ema_alpha = getattr(lr_scheduler_config, 'ema_alpha', 0.99)
        ou_seed = getattr(lr_scheduler_config, 'ou_seed', 1)

        # Phase 3 - Cooldown
        cooldown_steps = getattr(lr_scheduler_config, 'cooldown_steps', 0)
        cooldown_ratio = getattr(lr_scheduler_config, 'cooldown_ratio', 0.0)
        cooldown_type = getattr(lr_scheduler_config, 'cooldown_type', 'cosine')

        # Global
        lr_min_absolute = getattr(lr_scheduler_config, 'lr_min_absolute', None)
        if lr_min_absolute is None:
            # Fall back to min_lr_factor if lr_min_absolute not provided
            base_lr = optimizers.optimizers[0].param_groups[0]['lr']
            min_lr_factor = getattr(lr_scheduler_config, 'min_lr_factor', 0.0)
            lr_min_absolute = base_lr * min_lr_factor

        # Compute actual phase steps for logging
        actual_warm_steps = warm_steps if warm_steps > 0 else int(training_steps * warm_ratio)
        actual_cooldown_steps = cooldown_steps if cooldown_steps > 0 else int(training_steps * cooldown_ratio)
        actual_main_steps = training_steps - actual_warm_steps - actual_cooldown_steps

        # Build descriptive log message
        warm_desc = "disabled"
        if actual_warm_steps > 0:
            if warm_ratio > 0:
                warm_desc = f"{actual_warm_steps} steps ({warm_ratio*100:.1f}% of total)"
            else:
                warm_desc = f"{actual_warm_steps} steps (absolute)"
            warm_desc += f", direction={warm_direction}, type={warm_type}"
            if warm_direction == "down":
                warm_desc += f", start_ratio={warm_start_ratio}"

        cooldown_desc = "disabled"
        if actual_cooldown_steps > 0:
            if cooldown_ratio > 0:
                cooldown_desc = f"{actual_cooldown_steps} steps ({cooldown_ratio*100:.1f}% of total)"
            else:
                cooldown_desc = f"{actual_cooldown_steps} steps (absolute)"
            cooldown_desc += f", type={cooldown_type}"

        main_desc = f"{actual_main_steps} steps, decay_type={main_decay_type}, target_ratio={main_decay_ratio}"
        if use_ou_process:
            main_desc += f" (OU enabled: theta={ou_theta}, sigma={ou_sigma}, max_change={ou_max_change}, ema={ema_alpha})"
        else:
            main_desc += " (OU disabled)"

        logger.info(
            f"Creating Universal OU Stochastic LR Scheduler:\n"
            f"  Total steps: {training_steps}\n"
            f"  Phase distribution: {actual_warm_steps} warm → {actual_main_steps} main → {actual_cooldown_steps} cooldown\n"
            f"  Phase 1 Warm: {warm_desc}\n"
            f"  Phase 2 Main: {main_desc}\n"
            f"  Phase 3 Cooldown: {cooldown_desc}\n"
            f"  Min LR absolute: {lr_min_absolute:.2e}\n"
            f"  OU seed (reproducibility): {ou_seed}"
        )

        # Create universal OU schedulers for each optimizer
        self.schedulers: List[UniversalOUStochasticLR] = []
        for i, optimizer in enumerate(optimizers):
            scheduler = UniversalOUStochasticLR(
                optimizer=optimizer,
                total_steps=training_steps,
                # Phase 1
                warm_steps=warm_steps,
                warm_ratio=warm_ratio,
                warm_direction=warm_direction,
                warm_type=warm_type,
                warm_start_ratio=warm_start_ratio,
                # Phase 2
                use_ou_process=use_ou_process,
                main_decay_type=main_decay_type,
                main_decay_ratio=main_decay_ratio,
                ou_theta=ou_theta,
                ou_sigma=ou_sigma,
                ou_max_change=ou_max_change,
                ema_alpha=ema_alpha,
                seed=ou_seed,
                # Phase 3
                cooldown_steps=cooldown_steps,
                cooldown_ratio=cooldown_ratio,
                cooldown_type=cooldown_type,
                # Global
                lr_min_absolute=lr_min_absolute,
                last_epoch=-1,
                verbose=False  # We handle logging at container level
            )
            self.schedulers.append(scheduler)

            # Log initial LR for each optimizer
            initial_lrs = scheduler.get_last_lr()
            logger.info(f"Optimizer {i} initial LRs: {initial_lrs}")

    def __iter__(self):
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        """Step all schedulers and log current learning rates."""
        for i, scheduler in enumerate(self.schedulers):
            scheduler.step()

            # Log current LR periodically (every 100 steps)
            if scheduler.last_epoch % 100 == 0:
                current_lrs = scheduler.get_last_lr()
                phase = self._get_current_phase(scheduler)
                logger.debug(
                    f"Step {scheduler.last_epoch} - Optimizer {i} LRs: {current_lrs} "
                    f"(Phase: {phase})"
                )

    def _get_current_phase(self, scheduler: UniversalOUStochasticLR) -> str:
        """Determine current phase of the scheduler."""
        current_step = scheduler.last_epoch

        if current_step < scheduler.warm_steps:
            return f"warm ({scheduler.warm_direction})"
        elif current_step < scheduler.warm_steps + scheduler.main_steps:
            ou_status = "with OU" if scheduler.use_ou_process else "no OU"
            return f"main ({ou_status})"
        else:
            return "cooldown"

    def state_dict(self) -> Dict[str, Any]:
        """Save state dict for all schedulers.

        We save all scheduler states to preserve independent stochastic trajectories.
        """
        state = {
            'num_schedulers': len(self.schedulers),
            'scheduler_states': []
        }

        for i, scheduler in enumerate(self.schedulers):
            scheduler_state = scheduler.state_dict()
            # Add scheduler index for verification during load
            scheduler_state['scheduler_index'] = i
            state['scheduler_states'].append(scheduler_state)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict for all schedulers.

        Restores the exact stochastic state of each scheduler, including:
        - Current step (last_epoch)
        - OU process state (ou_factors)
        - EMA state (ema_ou_factors)
        - Per-parameter-group states
        """
        num_schedulers = state_dict.get('num_schedulers', 1)
        if num_schedulers != len(self.schedulers):
            logger.warning(
                f"Checkpoint has {num_schedulers} schedulers but current config has "
                f"{len(self.schedulers)}. Will load available states."
            )

        scheduler_states = state_dict.get('scheduler_states', [])

        # Load state for each scheduler
        for i, scheduler in enumerate(self.schedulers):
            if i < len(scheduler_states):
                # Load the corresponding state
                scheduler_state = copy.deepcopy(scheduler_states[i])
                # Remove our added metadata before loading
                scheduler_state.pop('scheduler_index', None)

                # Save the config-specified total_steps BEFORE load overwrites them
                config_total_steps = scheduler.total_steps
                config_main_steps = scheduler.main_steps
                config_warm_steps = scheduler.warm_steps
                config_cooldown_steps = scheduler.cooldown_steps
                saved_total_steps = scheduler_state.get('total_steps', config_total_steps)

                scheduler.load_state_dict(scheduler_state)

                # If total_steps changed (user set new training.steps), use the
                # NEW config values so the LR schedule covers the extended training.
                # DCP restores the checkpoint's total_steps, but the user explicitly
                # requested a different duration — honor that.
                if saved_total_steps != config_total_steps:
                    old_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, '_last_lr') and scheduler._last_lr else None
                    scheduler.total_steps = config_total_steps
                    scheduler.warm_steps = config_warm_steps
                    scheduler.cooldown_steps = config_cooldown_steps
                    scheduler.main_steps = config_main_steps
                    new_lr = scheduler._compute_main_base_lr(
                        scheduler.base_lrs[0],
                        max(0, scheduler.last_epoch - scheduler.warm_steps)
                    ) if scheduler.main_decay_type != "const" else scheduler.base_lrs[0]
                    logger.warning(
                        f"LR scheduler total_steps changed: checkpoint={saved_total_steps} → "
                        f"config={config_total_steps} (main_steps: {saved_total_steps - config_warm_steps - config_cooldown_steps} → "
                        f"{config_main_steps}). Using new schedule. "
                        f"LR at resume step {scheduler.last_epoch}: {old_lr} → ~{new_lr:.6f}. "
                        f"If this causes a loss spike, consider using main_decay_type=const "
                        f"or keeping training.steps unchanged."
                    )

                logger.info(
                    f"Loaded state for Universal OU scheduler {i} at step {scheduler.last_epoch} "
                    f"(total_steps={scheduler.total_steps}, main_steps={scheduler.main_steps})"
                )
            else:
                logger.warning(f"No checkpoint state available for Universal OU scheduler {i}")


def build_lr_schedulers_universal_ou(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> UniversalOULRSchedulersContainer:
    """Create a UniversalOULRSchedulersContainer with universal OU schedulers.

    This function creates a container of universal OU learning rate schedulers
    for the given optimizers. The universal OU scheduler can emulate all other
    schedulers through its flexible three-phase architecture.

    Configuration Parameters (in job_config.lr_scheduler):

    Phase 1 - Warm:
    - warm_steps: Absolute number of warm steps (default: 0, mutually exclusive with warm_ratio)
    - warm_ratio: Warm steps as ratio of total (default: 0.0, mutually exclusive with warm_steps)
    - warm_direction: "up" (warmup) or "down" (warmdown) (default: "up")
    - warm_type: Decay curve - "linear", "cosine", "exp", "sqrt" (default: "linear")
    - warm_start_ratio: Starting LR multiplier for warmdown (default: 2.0)

    Phase 2 - Main:
    - use_ou_process: Enable OU stochastic process (default: False)
    - main_decay_type: "const", "linear", "cosine", "exp", "sqrt" (default: "const")
    - main_decay_ratio: Target LR ratio at end of main phase (default: 0.2)
    - ou_theta: OU mean reversion rate (default: 0.008)
    - ou_sigma: OU noise amplitude (default: 0.1)
    - ou_max_change: Max OU factor change per step (default: 0.05)
    - ema_alpha: EMA smoothing factor (default: 0.99)
    - ou_seed: Random seed for reproducibility (default: 1)

    Phase 3 - Cooldown:
    - cooldown_steps: Absolute number of cooldown steps (default: 0, mutually exclusive with cooldown_ratio)
    - cooldown_ratio: Cooldown steps as ratio of total (default: 0.0, mutually exclusive with cooldown_steps)
    - cooldown_type: Decay curve - "linear", "cosine", "exp", "sqrt" (default: "cosine")

    Global:
    - lr_min_absolute: Absolute minimum LR floor (default: from min_lr_factor)

    Phase Behavior Examples:

    Example 1 - Replace old OU with warmdown:
        [lr_scheduler]
        scheduler_type = "universal_ou"
        warm_steps = 0
        warm_ratio = 0.2  # 20% warmdown
        warm_direction = "down"
        warm_type = "sqrt"
        warm_start_ratio = 2.0
        use_ou_process = true
        main_decay_type = "cosine"
        main_decay_ratio = 0.2
        cooldown_steps = 2000
        cooldown_type = "cosine"

    Example 2 - Pure cosine annealing (no OU):
        [lr_scheduler]
        scheduler_type = "universal_ou"
        warm_steps = 200
        warm_direction = "up"
        warm_type = "linear"
        use_ou_process = false
        main_decay_type = "cosine"
        main_decay_ratio = 0.0
        cooldown_steps = 0

    Example 3 - Warmup-Stable-Decay:
        [lr_scheduler]
        scheduler_type = "universal_ou"
        warm_ratio = 0.1
        warm_direction = "up"
        warm_type = "linear"
        use_ou_process = false
        main_decay_type = "const"
        cooldown_ratio = 0.1
        cooldown_type = "cosine"

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
        lr_scheduler_config (LRSchedulerConfig): The lr scheduler config.
        training_steps (int): The total number of training steps.

    Returns:
        UniversalOULRSchedulersContainer: Container with configured universal OU LR schedulers.
    """
    return UniversalOULRSchedulersContainer(optimizers, lr_scheduler_config, training_steps)


def build_lr_schedulers_auto(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> UniversalOULRSchedulersContainer:
    """
    Factory function for building learning rate schedulers.

    This function provides backward compatibility with the old lr_scheduler_factory.
    All scheduler types have been unified into the universal_ou scheduler.

    The old scheduler types (wsd, wdd, cosine, ou, doud) have been deprecated.
    The universal_ou scheduler can emulate all of them through its flexible
    three-phase architecture.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
        lr_scheduler_config (LRSchedulerConfig): The lr scheduler config.
        training_steps (int): The total number of training steps.

    Returns:
        UniversalOULRSchedulersContainer: Container with configured universal OU LR schedulers.

    Notes:
        - If scheduler_type is not "universal_ou", a warning will be logged
        - The universal_ou scheduler supports all functionality of previous schedulers
        - See build_lr_schedulers_universal_ou documentation for configuration examples
    """
    # Get scheduler type from config
    scheduler_type = getattr(lr_scheduler_config, 'scheduler_type', 'universal_ou')

    if scheduler_type != "universal_ou":
        logger.warning(
            f"scheduler_type='{scheduler_type}' is no longer supported. "
            f"All scheduler types have been unified into 'universal_ou'. "
            f"Using universal_ou scheduler instead.\n"
            f"Please update your config to use scheduler_type='universal_ou' and configure "
            f"the three-phase architecture (warm/main/cooldown) to achieve the desired behavior.\n"
            f"See documentation for examples of how to emulate old scheduler types."
        )

    logger.info("Using Universal OU LR Scheduler (unified scheduler for all use cases)")
    return build_lr_schedulers_universal_ou(optimizers, lr_scheduler_config, training_steps)
