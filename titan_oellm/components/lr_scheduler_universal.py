# Copyright (c) Titan-OELLM Custom Components.
# All rights reserved.
#
# Universal Learning Rate Scheduler for TorchTitan

"""
Universal Learning Rate Scheduler - Unified Scheduler

This module implements a universal learning rate scheduler that replaces all other
LR schedulers through a flexible three-phase architecture with optional process.

The scheduler has three configurable phases:
1. Phase 1 (Warm): Optional warm-up or warm-down with bidirectional control
2. Phase 2 (Main): Base schedule with optional process
3. Phase 3 (Cooldown): Final annealing to minimum LR

Key features:
- Unified Parameter System: Consistent naming across all phases
- Bidirectional Warm Phase: warm_direction="up" (warmup) or "down" (warmdown)
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
  - main_decay_type: "const", "linear", "cosine", "exp", "sqrt"
  - main_decay_ratio: Target LR ratio at end of main phase

Phase 3 - Cooldown:
  - cooldown_steps/cooldown_ratio: Duration (mutually exclusive)
  - cooldown_type: "linear", "cosine", "exp", "sqrt"

Global:
  - lr_min_absolute: Absolute minimum LR floor

Example usage:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = build_lr_schedulers_universal(
        optimizers_container,
        job_config  # Config with universal parameters in lr_scheduler section
    )
"""

import copy
import math
from typing import Any, Dict, List

from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import LRScheduler as LRSchedulerConfig
from torchtitan.tools.logging import logger

# Import the base universal scheduler implementation
from titan_oellm.components.universal_lr import UniversalLR

__all__ = [
    "UniversalLRSchedulersContainer",
    "build_lr_schedulers_universal",
    "build_lr_schedulers_auto",  # Factory function for backward compatibility
]


class UniversalLRSchedulersContainer(Stateful):
    """Container for multiple Universal learning rate schedulers.

    This container wraps multiple UniversalLR schedulers and provides a unified
    interface compatible with TorchTitan's training loop.

    Key Features:
    - Uses UniversalLR for maximum flexibility
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

        # Get universal parameters with defaults

        # Phase 1 - Warm
        warm_steps = getattr(lr_scheduler_config, 'warm_steps', 0)
        warm_ratio = getattr(lr_scheduler_config, 'warm_ratio', 0.0)
        warm_direction = getattr(lr_scheduler_config, 'warm_direction', 'up')
        warm_type = getattr(lr_scheduler_config, 'warm_type', 'linear')
        warm_start_ratio = getattr(lr_scheduler_config, 'warm_start_ratio', 2.0)

        # Phase 2 - Main
        main_decay_type = getattr(lr_scheduler_config, 'main_decay_type', 'const')
        main_decay_ratio = getattr(lr_scheduler_config, 'main_decay_ratio', 0.2)

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

        logger.info(
            f"Creating Universal LR Scheduler:\n"
            f"  Total steps: {training_steps}\n"
            f"  Phase distribution: {actual_warm_steps} warm → {actual_main_steps} main → {actual_cooldown_steps} cooldown\n"
            f"  Phase 1 Warm: {warm_desc}\n"
            f"  Phase 2 Main: {main_desc}\n"
            f"  Phase 3 Cooldown: {cooldown_desc}\n"
            f"  Min LR absolute: {lr_min_absolute:.2e}\n"
        )

        # Create universal schedulers for each optimizer
        self.schedulers: List[UniversalLR] = []
        for i, optimizer in enumerate(optimizers):
            scheduler = UniversalLR(
                optimizer=optimizer,
                total_steps=training_steps,
                # Phase 1
                warm_steps=warm_steps,
                warm_ratio=warm_ratio,
                warm_direction=warm_direction,
                warm_type=warm_type,
                warm_start_ratio=warm_start_ratio,
                # Phase 2
                main_decay_type=main_decay_type,
                main_decay_ratio=main_decay_ratio,
                # Phase 3
                cooldown_steps=cooldown_steps,
                cooldown_ratio=cooldown_ratio,
                cooldown_type=cooldown_type,
                # Global
                lr_min_absolute=lr_min_absolute,
                last_epoch=-1,
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

    def _get_current_phase(self, scheduler: UniversalLR) -> str:
        """Determine current phase of the scheduler."""
        current_step = scheduler.last_epoch

        if current_step < scheduler.warm_steps:
            return f"warm ({scheduler.warm_direction})"
        elif current_step < scheduler.warm_steps + scheduler.main_steps:
            return f"main"
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
                scheduler.load_state_dict(scheduler_state)
                logger.info(f"Loaded state for Universal scheduler {i} at step {scheduler.last_epoch}")
            else:
                logger.warning(f"No checkpoint state available for Universal scheduler {i}")


def build_lr_schedulers_universal(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> UniversalLRSchedulersContainer:
    """Create a UniversalLRSchedulersContainer with universal schedulers.

    This function creates a container of universal learning rate schedulers
    for the given optimizers. The universal scheduler can emulate all other
    schedulers through its flexible three-phase architecture.

    Configuration Parameters (in job_config.lr_scheduler):

    Phase 1 - Warm:
    - warm_steps: Absolute number of warm steps (default: 0, mutually exclusive with warm_ratio)
    - warm_ratio: Warm steps as ratio of total (default: 0.0, mutually exclusive with warm_steps)
    - warm_direction: "up" (warmup) or "down" (warmdown) (default: "up")
    - warm_type: Decay curve - "linear", "cosine", "exp", "sqrt" (default: "linear")
    - warm_start_ratio: Starting LR multiplier for warmdown (default: 2.0)

    Phase 2 - Main:
    - main_decay_type: "const", "linear", "cosine", "exp", "sqrt" (default: "const")
    - main_decay_ratio: Target LR ratio at end of main phase (default: 0.2)

    Phase 3 - Cooldown:
    - cooldown_steps: Absolute number of cooldown steps (default: 0, mutually exclusive with cooldown_ratio)
    - cooldown_ratio: Cooldown steps as ratio of total (default: 0.0, mutually exclusive with cooldown_steps)
    - cooldown_type: Decay curve - "linear", "cosine", "exp", "sqrt" (default: "cosine")

    Global:
    - lr_min_absolute: Absolute minimum LR floor (default: from min_lr_factor)

    Phase Behavior Examples:

    Example 1:
        [lr_scheduler]
        scheduler_type = "universal"
        warm_steps = 0
        warm_ratio = 0.2  # 20% warmdown
        warm_direction = "down"
        warm_type = "sqrt"
        warm_start_ratio = 2.0
        main_decay_type = "cosine"
        main_decay_ratio = 0.2
        cooldown_steps = 2000
        cooldown_type = "cosine"

    Example 2 - Pure cosine annealing:
        [lr_scheduler]
        scheduler_type = "universal"
        warm_steps = 200
        warm_direction = "up"
        warm_type = "linear"
        main_decay_type = "cosine"
        main_decay_ratio = 0.0
        cooldown_steps = 0

    Example 3 - Warmup-Stable-Decay:
        [lr_scheduler]
        scheduler_type = "universal"
        warm_ratio = 0.1
        warm_direction = "up"
        warm_type = "linear"
        main_decay_type = "const"
        cooldown_ratio = 0.1
        cooldown_type = "cosine"

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
        lr_scheduler_config (LRSchedulerConfig): The lr scheduler config.
        training_steps (int): The total number of training steps.

    Returns:
        UniversalLRSchedulersContainer: Container with configured universal LR schedulers.
    """
    return UniversalLRSchedulersContainer(optimizers, lr_scheduler_config, training_steps)


def build_lr_schedulers_auto(
    optimizers: OptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> UniversalLRSchedulersContainer:
    """
    Factory function for building learning rate schedulers.

    This function provides backward compatibility with the old lr_scheduler_factory.
    All scheduler types have been unified into the universal scheduler.

    The universal scheduler can emulate all of them through its flexible
    three-phase architecture.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
        lr_scheduler_config (LRSchedulerConfig): The lr scheduler config.
        training_steps (int): The total number of training steps.

    Returns:
        UniversalLRSchedulersContainer: Container with configured universal LR schedulers.

    Notes:
        - If scheduler_type is not "universal", a warning will be logged
        - The universal scheduler supports all functionality of previous schedulers
        - See build_lr_schedulers_universal documentation for configuration examples
    """
    # Get scheduler type from config
    scheduler_type = getattr(lr_scheduler_config, 'scheduler_type', 'universal')

    if scheduler_type != "universal":
        logger.warning(
            f"scheduler_type='{scheduler_type}' is no longer supported. "
            f"All scheduler types have been unified into 'universal'. "
            f"Using universal scheduler instead.\n"
            f"Please update your config to use scheduler_type='universal' and configure "
            f"the three-phase architecture (warm/main/cooldown) to achieve the desired behavior.\n"
            f"See documentation for examples of how to emulate old scheduler types."
        )

    logger.info("Using Universal LR Scheduler (unified scheduler for all use cases)")
    return build_lr_schedulers_universal(optimizers, lr_scheduler_config, training_steps)
