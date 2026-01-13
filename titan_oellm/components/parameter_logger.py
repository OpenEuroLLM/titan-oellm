# Copyright (c) 2024 Titan-OELLM Contributors
# All rights reserved.

import torch
from typing import Any, Optional, Union, Dict, List
from dataclasses import dataclass


@dataclass
class ParameterLoggingConfig:
    """Configuration for parameter statistics logging."""
    enabled: bool = False
    log_interval: int = 100
    log_parameters: bool = True
    log_gradients: bool = True
    log_optimizer_states: bool = True
    log_gates: bool = False  # Log ACG and alpha-dependent gates
    # Optional: limit to specific parameter patterns
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


class ParameterStatsLogger:
    """
    Standalone component for logging parameter and optimizer statistics to TensorBoard.

    This component integrates with torchtitan's existing TensorBoard logging infrastructure
    without requiring modifications to torchtitan core files.

    Features:
    - Parameter statistics: max, min, norm, std
    - Gradient statistics: max, min, norm, std
    - AdamW optimizer state statistics
    - Configurable logging intervals
    - DTensor support for distributed training
    - Pattern-based filtering

    Usage:
        config = ParameterLoggingConfig(enabled=True, log_interval=50)
        logger = ParameterStatsLogger(config, model_parts, optimizers)

        # In training loop:
        if logger.should_log(step):
            extra_metrics = logger.compute_stats(step)
            metrics_processor.log(step, loss, max_loss, extra_metrics)
    """

    def __init__(
        self,
        config: Union[ParameterLoggingConfig, Any],
        model_parts: Optional[List[torch.nn.Module]] = None,
        optimizers: Optional[Any] = None,
    ):
        # Convert from dataclass config if needed (simplified)
        if hasattr(config, 'enabled'):
            # This is a dataclass config from sci_job_config
            self.config = ParameterLoggingConfig(
                enabled=config.enabled,
                log_interval=config.log_interval,
                log_parameters=config.log_parameters,
                log_gradients=config.log_gradients,
                log_optimizer_states=config.log_optimizer_states,
                log_gates=getattr(config, 'log_gates', False),  # New field, default False for backward compat
                include_patterns=config.include_patterns,
                exclude_patterns=config.exclude_patterns,
            )
        else:
            # This is already a ParameterLoggingConfig
            self.config = config

        self.model_parts = model_parts or []
        self.optimizers = optimizers
        self.last_log_step = -1

    def should_log(self, step: int) -> bool:
        """Check if we should log statistics at this step."""
        if not self.config.enabled:
            return False

        if step == 1:  # Always log first step
            return True

        return step % self.config.log_interval == 0

    def _matches_patterns(self, name: str) -> bool:
        """Check if parameter name matches include/exclude patterns."""
        if self.config.exclude_patterns:
            for pattern in self.config.exclude_patterns:
                if pattern in name:
                    return False

        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if pattern in name:
                    return True
            return False  # If include patterns specified, must match one

        return True

    def _iter_model_parameters(self):
        """Helper to iterate over all model parameters with consistent naming."""
        for model_idx, model in enumerate(self.model_parts):
            model_prefix = f"model_{model_idx}" if len(self.model_parts) > 1 else "model"

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                if not self._matches_patterns(name):
                    continue

                # Clean parameter name for TensorBoard
                clean_name = name.replace('.', '/')
                yield model_prefix, clean_name, param

    def _get_tensor_stats(self, tensor: torch.Tensor, prefix: str) -> Dict[str, float]:
        """Compute statistics for a tensor, handling DTensor if present."""
        if tensor is None:
            return {}

        # Handle DTensor for distributed training
        if hasattr(tensor, 'full_tensor'):
            # This is likely a DTensor
            tensor_data = tensor.full_tensor()
        else:
            tensor_data = tensor

        with torch.no_grad():
            try:
                if tensor_data.dim() == 1:
                    stats = {
                        # f"{prefix}/max": tensor_data.max().item(),
                        # f"{prefix}/min": tensor_data.min().item(),
                        f"{prefix}/mean": tensor_data.mean().item(),
                        f"{prefix}/std": tensor_data.std().item(),
                    }
                elif tensor_data.dim() >= 2:
                    stats = {
                    # f"{prefix}/max": tensor_data.max().item(),
                    # f"{prefix}/min": tensor_data.min().item(),
                    f"{prefix}/norm": tensor_data.norm().item(),
                    f"{prefix}/mean": tensor_data.mean().item(),
                    f"{prefix}/norm_md1": tensor_data.norm(dim=1).mean().item(),
                    f"{prefix}/std": tensor_data.std().item(),
                }
                else:
                    stats = {
                    f"{prefix}/scalar": tensor_data.item(),
                }
                return stats
            except Exception as e:
                # Handle edge cases (e.g., empty tensors, scalar tensors)
                # Return numeric error indicator instead of string to avoid TensorBoard errors
                return {f"{prefix}/error": -1.0}

    def _compute_parameter_stats(self) -> Dict[str, Any]:
        """Compute parameter statistics for all model parts."""
        if not self.config.log_parameters:
            return {}

        param_stats = {}
        for model_prefix, clean_name, param in self._iter_model_parameters():
            prefix = f"params/{model_prefix}/{clean_name}"
            stats = self._get_tensor_stats(param.data, prefix)
            param_stats.update(stats)

        return param_stats

    def _compute_gradient_stats(self) -> Dict[str, Any]:
        """Compute gradient statistics for all model parts."""
        if not self.config.log_gradients:
            return {}

        grad_stats = {}
        for model_prefix, clean_name, param in self._iter_model_parameters():
            if param.grad is not None:
                prefix = f"grads/{model_prefix}/{clean_name}"
                stats = self._get_tensor_stats(param.grad.data, prefix)
                grad_stats.update(stats)

        return grad_stats

    def _compute_optimizer_stats(self) -> Dict[str, Any]:
        """Compute AdamW optimizer statistics."""
        if not self.config.log_optimizer_states or self.optimizers is None:
            return {}

        optimizer_stats = {}

        # Get optimizer list (simplified handling)
        optimizers_list = (
            list(self.optimizers.optimizers) if hasattr(self.optimizers, 'optimizers')
            else [self.optimizers]
        )

        for opt_idx, optimizer in enumerate(optimizers_list):
            opt_prefix = f"optimizer_{opt_idx}" if len(optimizers_list) > 1 else "optimizer"

            if not hasattr(optimizer, 'state'):
                continue

            for group_idx, param_group in enumerate(optimizer.param_groups):
                for param_idx, param in enumerate(param_group['params']):
                    if param not in optimizer.state:
                        continue

                    state = optimizer.state[param]
                    param_prefix = f"{opt_prefix}/group_{group_idx}/param_{param_idx}"

                    # AdamW momentum statistics (exp_avg)
                    if 'exp_avg' in state:
                        exp_avg_stats = self._get_tensor_stats(
                            state['exp_avg'],
                            f"{param_prefix}/exp_avg"
                        )
                        optimizer_stats.update(exp_avg_stats)

                    # AdamW second moment statistics (exp_avg_sq)
                    if 'exp_avg_sq' in state:
                        exp_avg_sq_stats = self._get_tensor_stats(
                            state['exp_avg_sq'],
                            f"{param_prefix}/exp_avg_sq"
                        )
                        optimizer_stats.update(exp_avg_sq_stats)

        return optimizer_stats

    def _compute_gate_stats(self) -> Dict[str, Any]:
        """
        Compute gate statistics for ACG, ACG_all, and alpha-dependent gates.

        Returns:
            Dictionary of gate metrics
        """
        if not self.config.log_gates or not self.model_parts:
            return {}

        gate_stats = {}

        # Iterate through model parts (typically just one for non-PP)
        for model in self.model_parts:
            # Alpha-dependent gates (per-layer residual gating)
            if hasattr(model, 'layers'):
                for layer_id, layer in enumerate(model.layers):
                    # Attention residual gate
                    if (hasattr(layer, 'attn_update') and
                        hasattr(layer.attn_update, 'alpha_dependent') and
                        layer.attn_update.alpha_dependent and
                        hasattr(layer.attn_update, 'last_gate_values') and
                        layer.attn_update.last_gate_values is not None):

                        gate_flat = layer.attn_update.last_gate_values.squeeze(-1).flatten()
                        if gate_flat.numel() > 0:
                            gate_stats[f'gating/train_attn_layer_{layer_id}/mean'] = gate_flat.mean().item()
                            gate_stats[f'gating/train_attn_layer_{layer_id}/std'] = gate_flat.std().item()

                    # FFN residual gate
                    if (hasattr(layer, 'ffn_update') and
                        hasattr(layer.ffn_update, 'alpha_dependent') and
                        layer.ffn_update.alpha_dependent and
                        hasattr(layer.ffn_update, 'last_gate_values') and
                        layer.ffn_update.last_gate_values is not None):

                        gate_flat = layer.ffn_update.last_gate_values.squeeze(-1).flatten()
                        if gate_flat.numel() > 0:
                            gate_stats[f'gating/train_ffn_layer_{layer_id}/mean'] = gate_flat.mean().item()
                            gate_stats[f'gating/train_ffn_layer_{layer_id}/std'] = gate_flat.std().item()

        return gate_stats

    def compute_stats(self, step: int) -> Dict[str, Any]:
        """
        Compute all enabled statistics.

        Args:
            step: Current training step

        Returns:
            Dictionary of metrics to be logged via extra_metrics parameter
        """
        if not self.config.enabled:
            return {}

        all_stats = {}

        # Compute parameter statistics
        param_stats = self._compute_parameter_stats()
        all_stats.update(param_stats)

        # Compute gradient statistics
        grad_stats = self._compute_gradient_stats()
        all_stats.update(grad_stats)

        # Compute optimizer statistics
        optimizer_stats = self._compute_optimizer_stats()
        all_stats.update(optimizer_stats)

        # Compute gate statistics
        gate_stats = self._compute_gate_stats()
        all_stats.update(gate_stats)

        # Track last log step
        self.last_log_step = step

        return all_stats

    def update_model_parts(self, model_parts: List[torch.nn.Module]) -> None:
        """Update model parts reference."""
        self.model_parts = model_parts

    def update_optimizers(self, optimizers: Any) -> None:
        """Update optimizers reference."""
        self.optimizers = optimizers