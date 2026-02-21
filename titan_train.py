#!/usr/bin/env python3
"""
Titan-OELLM Training Wrapper

This wrapper applies necessary patches before launching torchtitan training.
It fixes the CLI argument precedence issue where --job.config_file in args
interferes with tyro's ability to properly override TOML values with CLI args.

Usage:
    torchrun ... -m titan_train [args...]

This is equivalent to:
    torchrun ... -m torchtitan.train [args...]

But ensures CLI arguments properly override TOML config values.
"""

import sys


def _apply_config_manager_patch():
    """Apply monkey patch to ConfigManager before it's used."""
    from torchtitan.config.manager import ConfigManager

    _original_parse_args = ConfigManager.parse_args

    def _patched_parse_args(self, args: list[str] = sys.argv[1:]):
        """Fixed parse_args that filters --job.config_file before tyro.cli."""
        toml_values = self._maybe_load_toml(args)
        config_cls = self._maybe_add_custom_config(args, toml_values)

        base_config = (
            self._dict_to_dataclass(config_cls, toml_values)
            if toml_values
            else config_cls()
        )

        # Filter out --job.config_file from args before passing to tyro
        # This allows CLI args to properly override TOML values
        filtered_args = _filter_config_file_arg(args)

        # DEBUG: Log what args we're passing to tyro
        import logging
        logger = logging.getLogger(__name__)
        dump_folder_args = [a for a in filtered_args if "dump_folder" in a or "dump-folder" in a]
        logger.warning(f"[TITAN_TRAIN_DEBUG] Original args count: {len(args)}")
        logger.warning(f"[TITAN_TRAIN_DEBUG] Filtered args count: {len(filtered_args)}")
        logger.warning(f"[TITAN_TRAIN_DEBUG] dump_folder args in filtered_args: {dump_folder_args}")
        logger.warning(f"[TITAN_TRAIN_DEBUG] base_config.job.dump_folder BEFORE tyro.cli: '{base_config.job.dump_folder}'")

        import tyro
        from torchtitan.config.manager import custom_registry

        self.config = tyro.cli(
            config_cls, args=filtered_args, default=base_config, registry=custom_registry
        )

        # DEBUG: Log what we got after tyro.cli
        logger.warning(f"[TITAN_TRAIN_DEBUG] config.job.dump_folder AFTER tyro.cli: '{self.config.job.dump_folder}'")

        self._validate_config()
        return self.config

    def _filter_config_file_arg(args: list[str]) -> list[str]:
        """Remove --job.config_file from args since it's already been loaded."""
        filtered = []
        valid_keys = {"--job.config-file", "--job.config_file"}
        skip_next = False

        for arg in args:
            if skip_next:
                skip_next = False
                continue

            # Skip --job.config_file=value
            if "=" in arg:
                key = arg.split("=", 1)[0]
                if key in valid_keys:
                    continue
            # Skip --job.config_file value (two-part argument)
            elif arg in valid_keys:
                skip_next = True
                continue

            filtered.append(arg)

        return filtered

    # Apply the patch
    ConfigManager.parse_args = _patched_parse_args


# Apply patch before importing torchtitan.train
_apply_config_manager_patch()

# Now import and run the main training entry point
from torchtitan.train import main, Trainer

if __name__ == "__main__":
    main(Trainer)
