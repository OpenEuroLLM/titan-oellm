# Titan-OELLM Custom Components

# Import quantization components to register model converters (float8, mx, etc.)
import torchtitan.components.quantization  # noqa: F401


# ============================================================================
# MONKEY PATCH: Fix ConfigManager to properly handle CLI argument precedence
# ============================================================================
# Issue: --job.config_file in args interferes with tyro's ability to properly
# override TOML values with CLI arguments. Since ConfigManager already loads
# the TOML file via _maybe_load_toml(), we can safely filter --job.config_file
# from args before passing to tyro.cli, allowing CLI args to properly override.
# ============================================================================

import sys
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

    # NEW: Filter out --job.config_file from args before passing to tyro
    # This allows CLI args to properly override TOML values
    filtered_args = _filter_config_file_arg(args)

    import tyro
    from torchtitan.config.manager import custom_registry

    self.config = tyro.cli(
        config_cls, args=filtered_args, default=base_config, registry=custom_registry
    )

    self._validate_config()
    return self.config


def _filter_config_file_arg(args: list[str]) -> list[str]:
    """Remove --job.config_file from args since it's already been loaded."""
    filtered = []
    valid_keys = {"--job.config-file", "--job.config_file"}
    skip_next = False

    for i, arg in enumerate(args):
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