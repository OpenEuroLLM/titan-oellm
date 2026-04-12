import torchtitan.train as _torchtitan_train

from torchtitan.components.checkpoint import CheckpointManager as BaseCheckpointManager


class CheckpointManager(BaseCheckpointManager):
    """CheckpointManager extended with extra_steps support.

    Reads ``checkpoint_config.extra_steps`` (list[int]) and saves a checkpoint
    at each of those steps in addition to the regular interval.  Falls back
    gracefully when the attribute is absent so the class works with the vanilla
    torchtitan Checkpoint config as well.
    """

    def __init__(self, *args, checkpoint_config, **kwargs):
        super().__init__(*args, checkpoint_config=checkpoint_config, **kwargs)
        self.extra_steps = set(getattr(checkpoint_config, "extra_steps", []))

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        return super()._should_save(curr_step, last_step) or curr_step in self.extra_steps

    def _purge_stale_checkpoints(self):
        """Override to protect extra_steps checkpoints from keep_latest_k pruning."""
        if not self.extra_steps:
            return super()._purge_stale_checkpoints()

        import os
        import re

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
                step = int(match.group(1))
                path = os.path.join(self.folder, filename)
                discovered_checkpoints.append((step, path))

        discovered_checkpoints.sort()

        # Split into protected (extra_steps) and purgeable checkpoints
        purgeable = [(s, p) for s, p in discovered_checkpoints if s not in self.extra_steps]
        to_delete = purgeable[: -1 * self.keep_latest_k] if len(purgeable) > self.keep_latest_k else []

        for _, path in to_delete:
            assert self.purge_thread is not None
            self.purge_queue.put(path)


# Patch torchtitan.train so it uses this subclass when instantiating CheckpointManager.
_torchtitan_train.CheckpointManager = CheckpointManager
