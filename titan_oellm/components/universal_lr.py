# Copyright (c) Titan-OELLM Custom Components.
# All rights reserved.
#
# Universal Learning Rate Scheduler

"""
Universal LR Scheduler - Unified Scheduler for All Use Cases

This module implements a universal learning rate scheduler that can emulate all other
schedulers through a flexible three-phase architecture with optional process.

The scheduler provides unified three-phase training:

**Three Phases:**
1. Phase 1 (Warm): Optional warm-up or warm-down with configurable direction and curves
2. Phase 2 (Main): Base schedule with optional process and configurable decay
3. Phase 3 (Cooldown): Final annealing to minimum LR with configurable decay curves

**Key Features:**
- Bidirectional Phase 1: warm_direction="up" (warmup) or "down" (warmdown)
- Unified Decay Curves: All phases support linear, cosine, exp, sqrt
- Flexible Duration: Absolute steps or ratio of total steps for all phases
- State Persistence: Full checkpointing support

**New Universal Parameter System:**

Phase 1 - Warm Phase:
- warm_steps/warm_ratio: Duration (mutually exclusive, both can be 0)
- warm_direction: "up" or "down"
- warm_type: "linear", "cosine", "exp", "sqrt"
- warm_start_ratio: Starting LR multiplier (for warm_direction="down")

Phase 2 - Main Phase:
- main_decay_type: "const", "linear", "cosine", "exp", "sqrt"
- main_decay_ratio: Target LR at end of main phase (e.g., 0.2 = 20% of base_lr)

Phase 3 - Cooldown Phase:
- cooldown_steps/cooldown_ratio: Duration (mutually exclusive, both can be 0)
- cooldown_type: "linear", "cosine", "exp", "sqrt"

Global:
- lr_min_absolute: Absolute minimum LR floor

Example :
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    >>> scheduler = UniversalLR(
    ...     optimizer,
    ...     total_steps=10000,
    ...     warm_steps=0,
    ...     warm_ratio=0.2,  # 20% of training for warmdown
    ...     warm_direction="down",
    ...     warm_type="sqrt",
    ...     warm_start_ratio=2.0,
    ...     main_decay_type="cosine",
    ...     main_decay_ratio=0.2,
    ...     cooldown_steps=2000,
    ...     cooldown_ratio=0.0,
    ...     cooldown_type="cosine",
    ...     lr_min_absolute=1e-6,
    ...     seed=42
    ... )
"""

import math
import random
import torch
from torch.optim.lr_scheduler import _LRScheduler


class UniversalLR(_LRScheduler):
    """
    Universal Learning Rate Scheduler.

    A unified scheduler that can replace all other LR schedulers through flexible
    three-phase architecture with optional process.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_steps (int): Total number of training steps.

        # Phase 1 - Warm
        warm_steps (int): Absolute number of warm steps. Mutually exclusive with warm_ratio. Default: 0
        warm_ratio (float): Warm steps as ratio of total_steps. Mutually exclusive with warm_steps. Default: 0.0
        warm_direction (str): Direction of warm phase - "up" or "down". Default: "up"
        warm_type (str): Decay curve - "linear", "cosine", "exp", "sqrt". Default: "linear"
        warm_start_ratio (float): Starting LR multiplier for warm_direction="down". Default: 2.0

        # Phase 2 - Main
        main_decay_type (str): Base schedule decay - "const", "linear", "cosine", "exp", "sqrt". Default: "const"
        main_decay_ratio (float): Target LR ratio at end of main phase. Default: 0.2

        # Phase 3 - Cooldown
        cooldown_steps (int): Absolute number of cooldown steps. Mutually exclusive with cooldown_ratio. Default: 0
        cooldown_ratio (float): Cooldown steps as ratio of total_steps. Mutually exclusive with cooldown_steps. Default: 0.0
        cooldown_type (str): Decay curve - "linear", "cosine", "exp", "sqrt". Default: "cosine"

        # Global
        lr_min_absolute (float): Absolute minimum learning rate floor. Default: 0

        last_epoch (int): The index of last epoch. Default: -1
        verbose (bool): If True, prints a message to stdout for each update. Default: False
    """

    def __init__(
        self,
        optimizer,
        total_steps,
        # Phase 1 - Warm
        warm_steps=0,
        warm_ratio=0.0,
        warm_direction="up",
        warm_type="linear",
        warm_start_ratio=2.0,
        # Phase 2 - Main
        main_decay_type="const",
        main_decay_ratio=0.2,
        # Phase 3 - Cooldown
        cooldown_steps=0,
        cooldown_ratio=0.0,
        cooldown_type="cosine",
        # Global
        lr_min_absolute=0,
        last_epoch=-1,
    ):
        self.total_steps = total_steps
        self.lr_min_absolute = lr_min_absolute

        # Phase 1 - Warm phase parameters
        self.warm_steps = int(warm_steps)
        self.warm_ratio = float(warm_ratio)
        self.warm_direction = warm_direction
        self.warm_type = warm_type
        self.warm_start_ratio = warm_start_ratio

        # Phase 2 - Main phase parameters
        self.main_decay_type = main_decay_type
        self.main_decay_ratio = main_decay_ratio

        # Phase 3 - Cooldown phase parameters
        self.cooldown_steps = int(cooldown_steps)
        self.cooldown_ratio = float(cooldown_ratio)
        self.cooldown_type = cooldown_type

        # Validate and compute phase durations (must be after all parameters are set)
        self._validate_and_compute_phases()

        # State for each parameter group
        self._main_end_lrs = None

        super().__init__(optimizer, last_epoch)

    def _validate_and_compute_phases(self):
        """Validate parameters and compute phase durations."""

        # Validate warm phase
        if self.warm_steps > 0 and self.warm_ratio > 0:
            raise ValueError(
                f"warm_steps ({self.warm_steps}) and warm_ratio ({self.warm_ratio}) are mutually exclusive. "
                f"At most one can be non-zero."
            )

        # Compute warm_steps from ratio if needed
        if self.warm_ratio > 0:
            self.warm_steps = int(self.total_steps * self.warm_ratio)

        # Validate warm_direction
        if self.warm_direction not in ["up", "down"]:
            raise ValueError(f"warm_direction must be 'up' or 'down', got '{self.warm_direction}'")

        # Validate warm_type
        if self.warm_type not in ["linear", "cosine", "exp", "sqrt"]:
            raise ValueError(
                f"warm_type must be 'linear', 'cosine', 'exp', or 'sqrt', got '{self.warm_type}'"
            )

        # Validate cooldown phase
        if self.cooldown_steps > 0 and self.cooldown_ratio > 0:
            raise ValueError(
                f"cooldown_steps ({self.cooldown_steps}) and cooldown_ratio ({self.cooldown_ratio}) are mutually exclusive. "
                f"At most one can be non-zero."
            )

        # Compute cooldown_steps from ratio if needed
        if self.cooldown_ratio > 0:
            self.cooldown_steps = int(self.total_steps * self.cooldown_ratio)

        # Validate cooldown_type
        if self.cooldown_type not in ["linear", "cosine", "exp", "sqrt"]:
            raise ValueError(
                f"cooldown_type must be 'linear', 'cosine', 'exp', or 'sqrt', got '{self.cooldown_type}'"
            )

        # Validate main_decay_type
        if self.main_decay_type not in ["const", "linear", "cosine", "exp", "sqrt"]:
            raise ValueError(
                f"main_decay_type must be 'const', 'linear', 'cosine', 'exp', or 'sqrt', got '{self.main_decay_type}'"
            )

        # Compute main phase steps
        self.main_steps = self.total_steps - self.warm_steps - self.cooldown_steps

        # Ensure we have at least some main steps
        if self.main_steps < 0:
            raise ValueError(
                f"Warm phase + cooldown phase must be < total_steps. "
                f"Got warm={self.warm_steps}, cooldown={self.cooldown_steps}, total={self.total_steps}"
            )

    def _compute_decay_factor(self, progress, decay_type):
        """
        Compute decay factor for a given progress and decay type.

        Args:
            progress: Progress through phase (0.0 to 1.0)
            decay_type: One of "linear", "cosine", "exp", "sqrt"

        Returns:
            Decay factor (1.0 at start, 0.0 at end)
        """
        # Clamp progress to [0, 1] to avoid numerical issues
        progress = max(0.0, min(1.0, progress))

        if decay_type == "linear":
            return 1.0 - progress

        elif decay_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        elif decay_type == "sqrt":
            return math.sqrt(1.0 - progress)

        elif decay_type == "exp":
            # Exponential decay: use progress^2 for faster decay
            return (1.0 - progress) ** 2

        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")

    def _compute_warm_lr(self, base_lr, t_warm):
        """
        Compute the warm phase learning rate.

        Args:
            base_lr: The base learning rate from optimizer
            t_warm: Step within warm phase (0-indexed, 0 to warm_steps-1)

        Returns:
            Warm phase learning rate
        """
        if self.warm_steps == 0:
            return base_lr

        # Compute progress (0.0 at start, 1.0 at end)
        progress = t_warm / max(1, self.warm_steps - 1)

        if self.warm_direction == "up":
            # Warmup: from lr_min_absolute to base_lr
            lr_start = self.lr_min_absolute
            lr_end = base_lr

            # For warmup, we want to go from 0% to 100%, so we use (1 - decay_factor)
            decay_factor = self._compute_decay_factor(progress, self.warm_type)
            lr = lr_end + (lr_start - lr_end) * decay_factor

        else:  # warm_direction == "down"
            # Warmdown: from warm_start_ratio × base_lr to base_lr
            lr_start = self.warm_start_ratio * base_lr
            lr_end = base_lr

            # For warmdown, we decay from start to end
            decay_factor = self._compute_decay_factor(progress, self.warm_type)
            lr = lr_end + (lr_start - lr_end) * decay_factor

        return lr

    def _compute_main_base_lr(self, base_lr, t_main):
        """
        Compute the base schedule learning rate for the main phase (Phase 2).

        Args:
            base_lr: The base learning rate from optimizer
            t_main: Step within main phase (0-indexed, 0 to main_steps-1)

        Returns:
            Base schedule learning rate
        """
        if self.main_decay_type == "const":
            return base_lr

        # Compute progress and decay factor
        progress = t_main / max(1, self.main_steps - 1)
        decay_factor = self._compute_decay_factor(progress, self.main_decay_type)

        # Decay from base_lr to max(lr_min_absolute, main_decay_ratio × base_lr)
        # This ensures proper annealing to lr_min_absolute when main_decay_ratio=0.0
        target_lr = max(self.lr_min_absolute, base_lr * self.main_decay_ratio)
        lr = target_lr + (base_lr - target_lr) * decay_factor

        return lr

    def _compute_cooldown_lr(self, lr_start, t_cooldown):
        """
        Compute the cooldown phase learning rate (Phase 3).

        Args:
            lr_start: Starting LR for cooldown (end of main phase)
            t_cooldown: Step within cooldown phase (0-indexed)

        Returns:
            Learning rate at current cooldown step
        """
        if self.cooldown_steps == 0:
            return lr_start

        progress = t_cooldown / max(1, self.cooldown_steps - 1)
        decay_factor = self._compute_decay_factor(progress, self.cooldown_type)

        # Decay from lr_start to lr_min_absolute
        lr = self.lr_min_absolute + (lr_start - self.lr_min_absolute) * decay_factor

        return lr

    def get_lr(self):
        """Compute learning rate for current step."""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "use `get_last_lr()`.", UserWarning)

        t = self.last_epoch
        new_lrs = []

        for i, base_lr in enumerate(self.base_lrs):
            # Phase 1: Warm phase
            if t < self.warm_steps:
                lr = self._compute_warm_lr(base_lr, t)


            # Phase 2: Main phase
            elif t < self.warm_steps + self.main_steps:
                t_main = t - self.warm_steps

                # Compute base schedule LR
                base_schedule_lr = self._compute_main_base_lr(base_lr, t_main)
                
                lr = base_schedule_lr

                # Ensure LR doesn't go below lr_min_absolute
                lr = max(self.lr_min_absolute, lr)

                # Store LR at end of main phase for smooth transition to cooldown
                if t == self.warm_steps + self.main_steps - 1:
                    self._main_end_lrs[i] = lr

            # Phase 3: Cooldown phase
            else:
                t_cooldown = t - self.warm_steps - self.main_steps

                # Start from where main phase ended
                if self._main_end_lrs[i] is None:
                    lr_start = base_lr  # Fallback
                else:
                    lr_start = self._main_end_lrs[i]

                # Cooldown to lr_min_absolute
                lr = self._compute_cooldown_lr(lr_start, t_cooldown)

            new_lrs.append(lr)

        return new_lrs
