import torchtitan.train as _torchtitan_train

from torchtitan.components import CheckpointManager as BaseCheckpointManager


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


# Patch torchtitan.train so it uses this subclass when instantiating CheckpointManager.
_torchtitan_train.CheckpointManager = CheckpointManager
