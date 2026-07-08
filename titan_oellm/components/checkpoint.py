"""Patch torchtitan's CheckpointManager with extra_steps and initial_step support.

Instead of only replacing the module-level name (which can be bypassed by
``from torchtitan.components.checkpoint import CheckpointManager``), we patch
the **methods directly on the class object**.  This guarantees the patched
behaviour regardless of how the class was imported.
"""

import logging
import os
import re

from torchtitan.components.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Save originals so the patches can call them.
# ---------------------------------------------------------------------------
_original_init = CheckpointManager.__init__
_original_load = CheckpointManager.load
_original_should_save = CheckpointManager._should_save
_original_purge = CheckpointManager._purge_stale_checkpoints


# ---------------------------------------------------------------------------
# Patched __init__: read extra_steps and initial_step from checkpoint_config.
# ---------------------------------------------------------------------------
def _patched_init(self, *args, checkpoint_config, **kwargs):
    _original_init(self, *args, checkpoint_config=checkpoint_config, **kwargs)
    self.extra_steps = set(getattr(checkpoint_config, "extra_steps", []))
    self._initial_step = getattr(checkpoint_config, "initial_step", -1)
    logger.info(
        "[OELLM CheckpointManager] initial_step=%s, extra_steps=%s",
        self._initial_step,
        self.extra_steps,
    )


# ---------------------------------------------------------------------------
# Patched load: apply initial_step override after loading a model-only ckpt.
# ---------------------------------------------------------------------------
def _patched_load(self, step: int = -1) -> bool:
    loaded = _original_load(self, step)
    if loaded and self._initial_step >= 0 and "train_state" in self.states:
        trainer = self.states["train_state"]
        logger.info(
            "[OELLM] Setting step from %s to %s", trainer.step, self._initial_step
        )
        trainer.step = self._initial_step
    return loaded


# ---------------------------------------------------------------------------
# Patched _should_save: also save at extra_steps.
# ---------------------------------------------------------------------------
def _patched_should_save(self, curr_step: int, last_step: bool = False) -> bool:
    return _original_should_save(self, curr_step, last_step) or curr_step in self.extra_steps


# ---------------------------------------------------------------------------
# Patched _purge_stale_checkpoints: protect extra_steps from keep_latest_k.
# ---------------------------------------------------------------------------
def _patched_purge(self):
    if not self.extra_steps:
        return _original_purge(self)

    import torch.distributed as dist

    if not (
        self.keep_latest_k > 0
        and dist.get_rank() == 0
        and os.path.isdir(self.folder)
        and (
            not self.enable_ft_dataloader_checkpoints
            or (self.ft_manager and self.ft_manager.participating_rank() == 0)
        )
    ):
        return

    discovered_checkpoints = []
    for filename in os.listdir(self.folder):
        match = re.search(r"step-(\d+)", filename)
        if match:
            ckpt_step = int(match.group(1))
            path = os.path.join(self.folder, filename)
            discovered_checkpoints.append((ckpt_step, path))

    discovered_checkpoints.sort()

    # Split into protected (extra_steps) and purgeable checkpoints
    purgeable = [(s, p) for s, p in discovered_checkpoints if s not in self.extra_steps]
    to_delete = purgeable[: -1 * self.keep_latest_k] if len(purgeable) > self.keep_latest_k else []

    for _, path in to_delete:
        assert self.purge_thread is not None
        self.purge_queue.put(path)


# ---------------------------------------------------------------------------
# Apply patches to the class object.
# ---------------------------------------------------------------------------
CheckpointManager.__init__ = _patched_init
CheckpointManager.load = _patched_load
CheckpointManager._should_save = _patched_should_save
CheckpointManager._purge_stale_checkpoints = _patched_purge
