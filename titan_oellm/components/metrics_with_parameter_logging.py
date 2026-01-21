# Copyright (c) Titan-OELLM Contributors
# All rights reserved.

"""
Enhanced metrics processor with parameter logging integration.

This module provides a torchtitan-compatible metrics processor builder
that integrates parameter statistics logging seamlessly.
"""

import math
import os
from typing import Any

try:
    import tomli_w
except ImportError:
    tomli_w = None

from torchtitan.components.metrics import MetricsProcessor, build_metrics_processor
from torchtitan.config import JobConfig as BaseJobConfig
from torchtitan.distributed import ParallelDims

from titan_oellm.components.parameter_logger import ParameterStatsLogger
from titan_oellm.configs.sci_job_config import JobConfig


class EnhancedMetricsProcessor(MetricsProcessor):
    """
    Enhanced MetricsProcessor with parameter logging support.

    This extends torchtitan's MetricsProcessor to include parameter statistics
    logging without modifying the original torchtitan code.
    """

    def __init__(
        self,
        job_config: JobConfig,
        parallel_dims: ParallelDims,
        tag: str | None = None,
    ):
        # Initialize parameter logger FIRST to avoid AttributeError in property setters
        self.param_logger = None
        self._model_parts = None

        # Check if parameter logging is enabled
        param_logging_enabled = (
            hasattr(job_config, 'parameter_logging') and
            job_config.parameter_logging.enabled
        )

        if param_logging_enabled:
            self.param_logger = ParameterStatsLogger(
                config=job_config.parameter_logging,
                model_parts=None,  # Will be set later
                optimizers=None    # Will be set later
            )

        # Initialize base metrics processor (this may call property setters)
        super().__init__(job_config, parallel_dims, tag)

        # Override TensorBoard logger to use experiment_folder with OUTPUT_DIR
        if job_config.metrics.enable_tensorboard and hasattr(self.logger, '_loggers'):
            from datetime import datetime
            from torchtitan.components.metrics import TensorBoardLogger
            from torchtitan.tools.logging import logger

            # Build TensorBoard path using experiment_folder and OUTPUT_DIR
            experiment_dir = self._resolve_experiment_dir(job_config)
            tb_dir = os.path.join(
                experiment_dir,
                job_config.metrics.save_tb_folder,
                datetime.now().strftime("%Y%m%d-%H%M")
            )

            # Find and replace the TensorBoard logger in the container
            for i, log in enumerate(self.logger._loggers):
                if isinstance(log, TensorBoardLogger):
                    self.logger._loggers[i] = TensorBoardLogger(tb_dir, tag)
                    logger.info(f"TensorBoard logging redirected to experiment folder: {tb_dir}")
                    break

        # Log initialization status
        from torchtitan.tools.logging import logger
        if param_logging_enabled:
            logger.info(f"Parameter logging enabled with interval: {job_config.parameter_logging.log_interval}")
        else:
            logger.info("Parameter logging disabled")

        # Save configuration to dump folder for reproducibility
        self._save_config_to_toml(job_config)

    @staticmethod
    def _filter_none_recursive(obj: Any) -> Any:
        """
        Recursively filter None values from nested dictionaries.

        This is necessary because tomli_w cannot serialize None values,
        as the TOML spec does not support null/None values.

        Args:
            obj: Object to filter (dict, list, or primitive)

        Returns:
            Filtered object with None values removed from dicts
        """
        if isinstance(obj, dict):
            return {
                k: EnhancedMetricsProcessor._filter_none_recursive(v)
                for k, v in obj.items()
                if v is not None
            }
        elif isinstance(obj, list):
            return [EnhancedMetricsProcessor._filter_none_recursive(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _resolve_experiment_dir(job_config: JobConfig):
        """Resolve experiment folder relative to OUTPUT_DIR when available."""
        from torchtitan.tools.logging import logger

        folder = getattr(job_config.job, "experiment_folder", None)
        if folder is None:
            folder = getattr(job_config.job, "dump_folder", None)

        output_root = os.environ.get("OUTPUT_DIR")
        if output_root:
            return os.path.expandvars(os.path.join(output_root, folder or "experiment"))

        logger.warning("OUTPUT_DIR not set; using experiment_folder as provided")
        return os.path.expandvars(folder or "./outputs/experiment")

    def _save_config_to_toml(self, job_config: JobConfig) -> None:
        """Save job config to TOML file in resolved experiment folder (rank 0 only)."""
        import torch.distributed as dist
        from torchtitan.tools.logging import logger

        # Only rank 0 saves
        if not dist.is_initialized() or dist.get_rank() != 0:
            return

        if tomli_w is None:
            logger.warning("tomli_w not available, config saving disabled")
            return

        experiment_dir = self._resolve_experiment_dir(job_config)
        os.makedirs(experiment_dir, exist_ok=True)
        config_path = os.path.join(experiment_dir, "config.toml")

        try:
            # Filter None values before saving (TOML doesn't support None)
            config_dict = self._filter_none_recursive(job_config.to_dict())
            with open(config_path, "wb") as f:
                tomli_w.dump(config_dict, f)
            logger.info(f"Saved config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise

    # Add model_parts property
    @property
    def model_parts(self):
        return getattr(self, '_model_parts', None)

    @model_parts.setter
    def model_parts(self, value):
        self._model_parts = value
        if self.param_logger:
            self.param_logger.update_model_parts(value)

    # Override optimizers attribute setter using __setattr__
    def __setattr__(self, name, value):
        if name == 'optimizers':
            # Set in base class
            super().__setattr__(name, value)

            # Get model_parts from the caller (trainer) at the same time
            # This piggybacks on torchtitan's existing pattern: self.metrics_processor.optimizers = self.optimizers
            if hasattr(self, 'param_logger') and self.param_logger and not self.param_logger.model_parts:
                import inspect
                caller_frame = inspect.currentframe().f_back
                if 'self' in caller_frame.f_locals:
                    trainer = caller_frame.f_locals['self']
                    if hasattr(trainer, 'model_parts') and trainer.model_parts:
                        self.param_logger.update_model_parts(trainer.model_parts)
                        from torchtitan.tools.logging import logger
                        logger.info(f"Parameter logging: Connected to {len(trainer.model_parts)} model parts via optimizers setter")

            # Also set optimizers in parameter logger
            if hasattr(self, 'param_logger') and self.param_logger:
                self.param_logger.update_optimizers(value)
        elif name == 'lr_schedulers':
            # Set in base class
            super().__setattr__(name, value)
        else:
            # Default behavior for all other attributes
            super().__setattr__(name, value)

    def _get_current_learning_rate(self) -> float | None:
        """Extract current learning rate from the first optimizer."""
        try:
            if self.optimizers and len(self.optimizers) > 0:
                first_optimizer = next(iter(self.optimizers))
                if hasattr(first_optimizer, 'param_groups') and first_optimizer.param_groups:
                    return first_optimizer.param_groups[0].get('lr')
        except (AttributeError, IndexError, KeyError, StopIteration):
            pass
        return None

    def _calculate_perplexity(self, loss: float) -> float | None:
        """Calculate perplexity from loss, handling potential overflow."""
        try:
            # Clamp loss to prevent overflow in exp()
            if loss > 100:  # exp(100) ≈ 2.7e43, reasonable upper bound
                return None
            return math.exp(loss)
        except (OverflowError, ValueError):
            return None

    def log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        """Override log method to include parameter statistics, learning rate, perplexity, and grad_norm."""
        # Prepare combined metrics
        combined_extra_metrics = {}

        # Add any existing extra metrics
        if extra_metrics:
            combined_extra_metrics.update(extra_metrics)

        # Add grad_norm to metrics
        if grad_norm is not None:
            combined_extra_metrics['grad_norm'] = grad_norm

        # Add learning rate
        current_lr = self._get_current_learning_rate()
        if current_lr is not None:
            combined_extra_metrics['learning_rate'] = current_lr

        # Add perplexity
        perplexity = self._calculate_perplexity(global_avg_loss)
        if perplexity is not None:
            combined_extra_metrics['loss_metrics/global/avg_ppl'] = perplexity

        # Add rounding state if available (from weight normalizer)
        if hasattr(self, 'optimizers') and self.optimizers:
            try:
                first_opt = next(iter(self.optimizers), None)
                if first_opt and hasattr(first_opt, 'normalizer'):
                    rounding_state = first_opt.normalizer.get_rounding_state()
                    if 'int_bits' in rounding_state:
                        combined_extra_metrics['rounding/int_bits'] = rounding_state['int_bits']
                    if 'exponent' in rounding_state:
                        combined_extra_metrics['rounding/exponent'] = rounding_state['exponent']
            except (StopIteration, AttributeError):
                pass

        # Add parameter statistics if enabled and should log
        if self.param_logger and self.param_logger.should_log(step):
            param_stats = self.param_logger.compute_stats(step)
            if param_stats:
                combined_extra_metrics.update(param_stats)
                # Log info about parameter statistics
                from torchtitan.tools.logging import logger
                logger.info(f"Step {step}: Added {len(param_stats)} parameter statistics to metrics")

        # Call parent log method with combined metrics
        super().log(step, global_avg_loss, global_max_loss, grad_norm, combined_extra_metrics or None)


def build_metrics_processor_with_parameter_logging(
    job_config: BaseJobConfig,
    parallel_dims: ParallelDims,
    model_args=None,
    tag: str | None = None,
) -> MetricsProcessor:
    """
    Build an enhanced metrics processor with parameter logging support.

    This is a drop-in replacement for torchtitan's build_metrics_processor
    that adds parameter logging capabilities when using titan_oellm JobConfig.

    Args:
        job_config: Job configuration (should be titan_oellm.configs.sci_job_config.JobConfig)
        parallel_dims: Parallel dimensions configuration
        model_args: Unused, kept for compatibility
        tag: Optional tag for metrics

    Returns:
        MetricsProcessor: Enhanced metrics processor with parameter logging
    """
    # Check if we have titan_oellm JobConfig with parameter logging
    if hasattr(job_config, 'parameter_logging'):
        # Use our enhanced metrics processor
        return EnhancedMetricsProcessor(job_config, parallel_dims, tag)
    else:
        # Fall back to standard torchtitan metrics processor
        return build_metrics_processor(job_config, parallel_dims, model_args, tag)


# Compatibility alias
build_enhanced_metrics_processor = build_metrics_processor_with_parameter_logging