"""
anGPT weight matrix normalization component for TorchTitan.

This module implements the weight normalization strategy from anGPT,
which normalizes weight matrices after each optimizer step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

# Check if DTensor is available (PyTorch >= 2.0 with distributed)
try:
    from torch.distributed._tensor import DTensor
    DTENSOR_AVAILABLE = True
except ImportError:
    DTENSOR_AVAILABLE = False
    DTensor = None


def spectral_norm_power_iteration(
    W: Tensor, u: Optional[Tensor] = None, v: Optional[Tensor] = None, num_iters: int = 2, eps: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Power iteration to estimate top singular value and vectors.

    Implementation aligned with PyTorch's official spectral_norm using torch.mv
    and F.normalize for efficiency and correctness. Handles DTensor by accessing
    the local tensor directly since torch.mv doesn't support DTensor natively.

    Args:
        W: Weight matrix of shape [out_features, in_features] (can be DTensor)
        u: Left singular vector (optional, for warm-starting)
        v: Right singular vector (optional, for warm-starting)
        num_iters: Number of power iteration steps
        eps: Small value to avoid division by zero

    Returns:
        (u, v, sigma) - singular vectors and σ₁ (top singular value)
    """
    # Handle DTensor: access local tensor directly for spectral norm computation
    # Note: PyTorch's official spectral_norm also doesn't support DTensor natively
    is_dtensor = DTENSOR_AVAILABLE and isinstance(W, DTensor)
    if is_dtensor:
        W_local = W._local_tensor
    else:
        W_local = W

    dtype = W_local.dtype
    W_compute = W_local.float()
    m, n = W_compute.shape

    # Initialize u and v if not provided or if dimensions don't match
    # Dimension mismatch can occur with DTensor when local shard size differs from cached vectors
    if u is None or u.size(0) != m:
        u = F.normalize(torch.randn(m, device=W_compute.device, dtype=W_compute.dtype), dim=0, eps=eps)
    else:
        u = u.float()
        # Ensure u is on the correct device
        if u.device != W_compute.device:
            u = u.to(W_compute.device)

    if v is None or v.size(0) != n:
        v = F.normalize(torch.randn(n, device=W_compute.device, dtype=W_compute.dtype), dim=0, eps=eps)
    else:
        v = v.float()
        # Ensure v is on the correct device
        if v.device != W_compute.device:
            v = v.to(W_compute.device)

    # Power iteration - following PyTorch's official implementation
    # Uses torch.mv for matrix-vector products and F.normalize for normalization
    for _ in range(num_iters):
        # u = normalize(W @ v)
        u = F.normalize(torch.mv(W_compute, v), dim=0, eps=eps, out=u)
        # v = normalize(W^H @ u) - using .H for conjugate transpose (supports complex)
        v = F.normalize(torch.mv(W_compute.H, u), dim=0, eps=eps, out=v)

    # Compute sigma = u^T W v using torch.dot for efficiency
    sigma = torch.dot(u, torch.mv(W_compute, v)).abs()

    return u.to(dtype), v.to(dtype), sigma.to(dtype)


def spectral_normalize(
    W: Tensor, u: Optional[Tensor] = None, v: Optional[Tensor] = None, num_iters: int = 2
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize W so that σ₁(W) = 1.0 (full spectral normalization).

    Args:
        W: Weight matrix of shape [out_features, in_features]
        u: Left singular vector (optional, for warm-starting)
        v: Right singular vector (optional, for warm-starting)
        num_iters: Number of power iteration steps

    Returns:
        (W_normalized, u, v) - normalized weight matrix and updated singular vectors
    """
    # Handle DTensor: work with local tensor directly
    is_dtensor = DTENSOR_AVAILABLE and isinstance(W, DTensor)
    if is_dtensor:
        # Access DTensor's underlying local tensor directly
        W_local = W._local_tensor
    else:
        W_local = W

    dtype = W_local.dtype
    u, v, sigma = spectral_norm_power_iteration(W_local, u, v, num_iters)

    # Normalize: W_norm = W / σ₁
    W_normalized = W_local.float() / sigma.clamp(min=1e-8)
    W_normalized = W_normalized.to(dtype)

    # Update in-place
    if is_dtensor:
        # Modify DTensor's local tensor in-place
        W._local_tensor.copy_(W_normalized)
        return W, u, v
    else:
        return W_normalized, u, v


def spectral_normalize_bounded(
    W: Tensor, u: Optional[Tensor] = None, v: Optional[Tensor] = None, num_iters: int = 2
) -> tuple[Tensor, Tensor, Tensor]:
    """Bound W so that σ₁(W) ≤ 1.0 (only rescales if σ₁ > 1.0).

    Args:
        W: Weight matrix of shape [out_features, in_features]
        u: Left singular vector (optional, for warm-starting)
        v: Right singular vector (optional, for warm-starting)
        num_iters: Number of power iteration steps

    Returns:
        (W_bounded, u, v) - bounded weight matrix and updated singular vectors
    """
    # Handle DTensor: work with local tensor directly
    is_dtensor = DTENSOR_AVAILABLE and isinstance(W, DTensor)
    if is_dtensor:
        # Access DTensor's underlying local tensor directly
        W_local = W._local_tensor
    else:
        W_local = W

    dtype = W_local.dtype
    u, v, sigma = spectral_norm_power_iteration(W_local, u, v, num_iters)

    # Only rescale if σ₁ > 1.0 (clamp min to 1.0)
    W_bounded = W_local.float() / sigma.clamp(min=1.0)
    W_bounded = W_bounded.to(dtype)

    # Update in-place
    if is_dtensor:
        # Modify DTensor's local tensor in-place
        W._local_tensor.copy_(W_bounded)
        return W, u, v
    else:
        return W_bounded, u, v


def justnorm(x, idim=-1):
    """Normalize tensor along specified dimension."""
    dtype = x.dtype
    x = x.float()
    res = (x / x.norm(p=2, dim=idim, keepdim=True).clamp(min=1e-8)).to(dtype=dtype)
    return res


def justnorm_min(x, idim=-1):
    """Normalize tensor along specified dimension with minimum clipping."""
    dtype = x.dtype
    x = x.float()
    res = (x / x.norm(p=2, dim=idim, keepdim=True).clamp(min=1.0)).to(dtype=dtype)
    return res


def justnorm_full_tensor(x):
    """Normalize entire tensor globally (not per-dimension).

    This computes L2 norm across ALL elements and normalizes the entire tensor
    to have global L2 norm = 1.0. This results in tighter exponent clustering
    compared to per-dimension normalization.
    """
    dtype = x.dtype
    x = x.float()
    global_norm = x.norm(p=2).clamp(min=1e-8)
    res = (x / global_norm).to(dtype=dtype)
    return res


def justnorm_full_tensor_min(x):
    """Normalize entire tensor globally with minimum clipping."""
    dtype = x.dtype
    x = x.float()
    global_norm = x.norm(p=2).clamp(min=1.0)
    res = (x / global_norm).to(dtype=dtype)
    return res


def round_weights_avoid_zero(x, exponent: int = 2):
    """Round weights to 10^-exponent precision, avoiding zero values.

    Values that would round to 0 are instead rounded to ±10^-exponent
    based on the original sign of the value.

    Args:
        x: Input tensor
        exponent: Negative exponent (1-5). E.g., 2 → round to 0.01

    Returns:
        Rounded tensor with no zero values
    """
    dtype = x.dtype
    x = x.float()

    precision = 10 ** (-exponent)  # e.g., 0.01 for exponent=2

    # Standard rounding to precision
    rounded = torch.round(x / precision) * precision

    # Find values that rounded to zero
    zero_mask = rounded == 0

    # For values that rounded to zero, use the smallest non-zero value
    # with the sign of the original value
    signs = torch.sign(x)
    # Handle exact zeros in original by defaulting to positive
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    # Replace zeros with ±precision based on original sign
    rounded = torch.where(zero_mask, signs * precision, rounded)

    return rounded.to(dtype=dtype)


# Bit-width to max value mapping for symmetric signed integers
# Formula: max_val = 2^(bits-1) - 1
# Supports intermediate bit-widths for smoother ramp-up transitions
INT_BITS_TO_MAX = {
    8: 127,           # INT8
    10: 511,          # INT10
    12: 2047,         # INT12
    14: 8191,         # INT14
    16: 32767,        # INT16
    20: 524287,       # INT20
    24: 8388607,      # INT24
    32: 2147483647,   # INT32
}

# Ramp-up schedule for INT channel-wise mode: start high precision, decrease to INT8
# Smoother transitions with intermediate bit-widths (ratio ~4-16× between steps)
# Note: INT32 omitted as its precision (2.1B levels) is essentially "no rounding"
INT_RAMPUP_SEQUENCE = [24, 20, 16, 14, 12, 10, 8]  # Smooth progression to INT8


def round_weights_int_channelwise(x, bits: int = 8):
    """Round weights to INT-compatible grid with per-channel scales.

    For integer quantization compatibility, each input channel (dim=1) has its
    own scale derived from its maximum absolute value. Values are rounded to
    the nearest representable integer value and then dequantized.

    Args:
        x: Input tensor with shape [out_features, in_features]
        bits: Bit-width for quantization (8, 10, 12, 14, 16, 20, 24, or 32)

    Returns:
        Tensor rounded to INT-compatible values with no zeros
    """
    dtype = x.dtype
    x = x.float()

    max_val = INT_BITS_TO_MAX.get(bits, 127)

    # Compute per-input-channel scale: reduce over out_features (dim=0)
    # Result shape: [1, in_features] - one scale per input channel (column)
    channel_max = x.abs().max(dim=0, keepdim=True).values
    scale = channel_max / max_val
    scale = scale.clamp(min=1e-8)  # Avoid division by zero

    # Quantize to INT range
    q = torch.round(x / scale)
    q = q.clamp(-max_val, max_val)

    # Avoid zeros: replace 0 with ±1 based on original sign
    zero_mask = q == 0
    signs = torch.sign(x)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = torch.where(zero_mask, signs, q)

    # Dequantize
    result = q * scale

    return result.to(dtype=dtype)


def round_weights_int_global(x, bits: int = 8):
    """Round weights to INT-compatible grid with single global scale.

    Uses one scale for the entire matrix (max absolute value across all elements).
    This is simpler than per-channel scaling but may lose more precision for
    matrices with varying magnitude across channels.

    Args:
        x: Input tensor with shape [out_features, in_features]
        bits: Bit-width for quantization (8, 10, 12, 14, 16, 20, 24, or 32)

    Returns:
        Tensor rounded to INT-compatible values with no zeros
    """
    dtype = x.dtype
    x = x.float()

    max_val = INT_BITS_TO_MAX.get(bits, 127)

    # Single global scale for entire matrix
    global_max = x.abs().max()
    scale = global_max / max_val
    scale = scale.clamp(min=1e-8)

    # Quantize to INT range
    q = torch.round(x / scale)
    q = q.clamp(-max_val, max_val)

    # Avoid zeros: replace 0 with ±1 based on original sign
    zero_mask = q == 0
    signs = torch.sign(x)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = torch.where(zero_mask, signs, q)

    # Dequantize
    result = q * scale

    return result.to(dtype=dtype)


class GPTNormalizer:
    """
    Weight matrix normalizer specifically for anGPT models.

    Applies weight normalization to anGPT model weights after optimizer steps,
    following the anGPT normalization strategy. Supports both L2 normalization
    (row/column-wise) and spectral normalization (σ₁ constraint).
    """

    def __init__(self, model_parts: List[nn.Module], out_norm_dim_0: bool = True, bounded: bool = False,
                 normalize_full_tensor: bool = False, rounding_enabled: bool = False, rounding_exponent: int = 2,
                 rounding_rampup_steps: int = 0, rounding_int_stages: bool = False,
                 rounding_int_per_channel: bool = True, rounding_int_target_bits: int = 8,
                 rounding_late_start: bool = False, total_steps: int = 0, normalize_every_n_steps: int = 1,
                 l2_norm_enabled: bool = True, spectral_norm_enabled: bool = False, spectral_num_iters: int = 2):
        """
        Args:
            model_parts: List of anGPT model parts to normalize
            out_norm_dim_0: Whether to normalize output projection weights along dim 0
            bounded: Whether to use bounded normalization (clip min to 1.0 for L2, σ₁ ≤ 1.0 for spectral)
            normalize_full_tensor: Whether to normalize entire weight matrices globally
                                   (not per-dimension). This results in tighter exponent
                                   clustering suitable for INT8 quantization.
            rounding_enabled: Enable weight rounding after L2 normalization to avoid dead weights
            rounding_exponent: Negative exponent for rounding precision (1-5). E.g., 2 → round to 0.01
            rounding_rampup_steps: Steps per exponent level during ramp-up (0 = disabled)
            rounding_int_stages: Enable INT quantization stages (INT24 → ... → target_bits).
                                 When False (default), uses fixed precision rounding with rounding_exponent.
            rounding_int_per_channel: When True (default), use per-input-channel scales (standard INT8 quantization).
                                      When False, use a single global scale for the entire weight matrix.
                                      Only applies when rounding_int_stages=True.
            rounding_int_target_bits: Target bit-width for INT stages (8, 10, 12, or 14).
                                      Ramp-up stops at this target. Default is 8 (INT8).
            rounding_late_start: When True, apply rounding only in the final training phase
                                 instead of ramping up from the beginning. Requires total_steps.
            total_steps: Total training steps. Required when rounding_late_start=True.
            normalize_every_n_steps: Normalize weights every N optimizer steps
            l2_norm_enabled: Enable L2 normalization (row/column-wise). Default: True
            spectral_norm_enabled: Enable spectral normalization (σ₁ constraint). Default: False
            spectral_num_iters: Power iteration steps for spectral norm estimation. Default: 2
        """
        self.model_parts = model_parts
        self.out_norm_dim_0 = out_norm_dim_0
        self.bounded = bounded
        self.normalize_full_tensor = normalize_full_tensor
        self.rounding_enabled = rounding_enabled
        self.rounding_exponent = rounding_exponent
        self.rounding_rampup_steps = rounding_rampup_steps
        self.rounding_int_stages = rounding_int_stages
        self.rounding_int_per_channel = rounding_int_per_channel
        self.rounding_int_target_bits = rounding_int_target_bits
        self.rounding_late_start = rounding_late_start
        self.total_steps = total_steps
        self.normalize_every_n_steps = normalize_every_n_steps
        self._step_count = 0  # Track normalization calls for ramp-up

        # L2 normalization settings
        self.l2_norm_enabled = l2_norm_enabled

        # Spectral normalization settings
        self.spectral_norm_enabled = spectral_norm_enabled
        self.spectral_num_iters = spectral_num_iters
        self._spectral_cache: dict[str, tuple[Tensor, Tensor]] = {}  # Cache for (u, v) vectors

        # Select appropriate normalization function
        if normalize_full_tensor:
            self.norm_fn = justnorm_full_tensor_min if bounded else justnorm_full_tensor
            self._norm_fn_name = "justnorm_full_tensor_min" if bounded else "justnorm_full_tensor"
        else:
            self.norm_fn = justnorm_min if bounded else justnorm
            self._norm_fn_name = "justnorm_min" if bounded else "justnorm"

    def _get_rounding_phase_length(self) -> int:
        """Calculate total length of rounding phase (all ramp-up levels)."""
        if self.rounding_rampup_steps <= 0:
            return 0

        if self.rounding_int_stages:
            target_bits = self.rounding_int_target_bits
            if target_bits in INT_RAMPUP_SEQUENCE:
                num_levels = INT_RAMPUP_SEQUENCE.index(target_bits) + 1
            else:
                num_levels = 1
        else:
            # Fixed precision: exponent 8 → target (e.g., 8 → 2 = 7 levels)
            num_levels = 8 - self.rounding_exponent + 1

        return num_levels * self.rounding_rampup_steps

    def _get_effective_step(self) -> int:
        """Get effective step for ramp-up calculation.

        For late_start mode: returns steps since rounding phase began.
        For normal mode: returns actual step count.
        Returns -1 if rounding should not be applied yet (late_start mode only).
        """
        if not self.rounding_late_start:
            return self._step_count

        if self.total_steps <= 0:
            return self._step_count  # Fallback to normal behavior

        rounding_phase_length = self._get_rounding_phase_length()
        rounding_start_step = self.total_steps - rounding_phase_length

        if self._step_count < rounding_start_step:
            return -1  # Not in rounding phase yet

        return self._step_count - rounding_start_step

    def _get_current_exponent(self) -> Optional[int]:
        """Compute current exponent based on ramp-up schedule (for fixed precision mode).

        Returns None if in late_start mode and not yet in rounding phase.
        """
        effective_step = self._get_effective_step()
        if effective_step < 0:
            return None  # Late start: not in rounding phase yet

        if self.rounding_rampup_steps <= 0:
            return self.rounding_exponent  # No ramp-up, use target directly

        # Start at exponent 8, decrease by 1 every rampup_steps
        start_exponent = 8
        levels_passed = effective_step // self.rounding_rampup_steps
        current_exponent = start_exponent - levels_passed

        # Clamp to target exponent (don't go below target)
        return max(current_exponent, self.rounding_exponent)

    def _get_current_bits(self) -> Optional[int]:
        """Compute current bit-width based on ramp-up schedule (for INT channel-wise mode).

        Target is configurable via rounding_int_target_bits (8, 10, 12, or 14).
        Ramp-up progresses through intermediate bit-widths and stops at target:
        INT32 → INT24 → INT20 → INT16 → INT14 → INT12 → INT10 → INT8

        Returns None if in late_start mode and not yet in rounding phase.
        """
        effective_step = self._get_effective_step()
        if effective_step < 0:
            return None  # Late start: not in rounding phase yet

        target_bits = self.rounding_int_target_bits

        if self.rounding_rampup_steps <= 0:
            return target_bits  # No ramp-up, use target directly

        # Find target index in sequence
        if target_bits in INT_RAMPUP_SEQUENCE:
            target_idx = INT_RAMPUP_SEQUENCE.index(target_bits)
        else:
            return target_bits  # Invalid target, use as-is

        # Progress through sequence: INT32 → INT24 → ... → target
        levels_passed = effective_step // self.rounding_rampup_steps
        current_idx = min(levels_passed, target_idx)

        return INT_RAMPUP_SEQUENCE[current_idx]

    def _apply_rounding(self, weight_data):
        """Apply rounding if enabled."""
        if not self.rounding_enabled:
            return weight_data

        if self.rounding_int_stages:
            bits = self._get_current_bits()
            if bits is None:  # Late start: not in rounding phase yet
                return weight_data
            if self.rounding_int_per_channel:
                return round_weights_int_channelwise(weight_data, bits)
            else:
                return round_weights_int_global(weight_data, bits)
        else:
            exponent = self._get_current_exponent()
            if exponent is None:  # Late start: not in rounding phase yet
                return weight_data
            return round_weights_avoid_zero(weight_data, exponent)

    def get_rounding_state(self) -> dict[str, int | None]:
        """Get current rounding state for TensorBoard logging.

        Returns dict with either 'int_bits' or 'exponent' key, or empty if disabled.
        """
        if not self.rounding_enabled:
            return {}

        if self.rounding_int_stages:
            bits = self._get_current_bits()
            return {'int_bits': bits} if bits is not None else {}
        else:
            exponent = self._get_current_exponent()
            return {'exponent': exponent} if exponent is not None else {}

    @torch.compiler.disable
    def get_normalization_fn(self):
        """
        Get a normalization function that applies spectral norm, L2 norm, and rounding.

        Returns a callable that takes a weight tensor and returns normalized/rounded tensor.
        This function encapsulates spectral normalization, L2 normalization, and rounding logic.

        When both spectral and L2 normalization are enabled:
        1. Apply spectral norm first (bounds σ₁) - only for 2D weight matrices, not embeddings/head
        2. Then apply L2 norm (normalizes rows/cols)
        3. Finally apply rounding if enabled
        """
        def normalization_fn(weight_data, dim=None, is_output=False, use_justnorm=False, weight_name=None,
                             is_embedding=False, is_head=False, only_bound=False):
            """
            Apply spectral normalization, L2 normalization, and optional rounding to weight tensor.

            Args:
                weight_data: Weight tensor to normalize
                dim: Dimension to normalize along (None for full tensor mode)
                is_output: True for layer output projections (wo, ffn output_layer) that use out_norm_dim_0 config
                use_justnorm: True to force using justnorm instead of self.norm_fn (for tok_embeddings)
                weight_name: Unique name for this weight tensor (for spectral norm caching)
                is_embedding: True for embedding weights (skip spectral norm)
                is_head: True for output head weights (skip spectral norm)
                only_bound: If not None, overrides self.bounded for this weight (True = bounded, False = unbounded)

            Returns:
                Normalized and optionally rounded weight tensor
            """
            result = weight_data

            # Determine bounded mode: use only_bound if provided, otherwise use self.bounded
            use_bounded = only_bound if only_bound else self.bounded

            # Step 1: Apply spectral norm (if enabled)
            # Skip for: 1D/scalar tensors, embeddings, head weights
            # Spectral norm only makes sense for 2D weight matrices in linear layers
            skip_spectral = (
                weight_data.ndim != 2 or  # Only 2D matrices
                is_embedding or           # Skip embeddings
                is_head or                # Skip output head
                use_justnorm              # tok_embeddings uses this flag
            )
            if self.spectral_norm_enabled and not skip_spectral:
                cache_key = weight_name or f"tensor_{id(weight_data)}"
                u, v = self._spectral_cache.get(cache_key, (None, None))

                if use_bounded:
                    result, u, v = spectral_normalize_bounded(result, u, v, self.spectral_num_iters)
                else:
                    result, u, v = spectral_normalize(result, u, v, self.spectral_num_iters)

                self._spectral_cache[cache_key] = (u, v)

            # Step 2: Apply L2 norm (if enabled) - row/column-wise
            if self.l2_norm_enabled:
                # Determine dimension for layer output projections (affected by out_norm_dim_0)
                if is_output and not self.normalize_full_tensor:
                    dim = 0 if self.out_norm_dim_0 else 1

                if self.normalize_full_tensor:
                    # Full tensor normalization - select bounded/unbounded based on use_bounded
                    if use_bounded:
                        result = justnorm_full_tensor_min(result)
                    else:
                        result = justnorm_full_tensor(result)
                else:
                    if use_justnorm:
                        # tok_embeddings specifically uses unbounded justnorm
                        result = justnorm(result, dim)
                    else:
                        # Select bounded/unbounded based on use_bounded
                        if use_bounded:
                            result = justnorm_min(result, dim)
                        else:
                            result = justnorm(result, dim)

            # Step 3: Apply rounding if enabled
            return self._apply_rounding(result)

        return normalization_fn

    def normalize_weights(self):
        """Apply weight normalization to all model parts.

        Each model implements its own normalize_weights() method that knows
        its architecture. This normalizer just provides the normalization function.
        """
        with torch.no_grad():
            # Get the normalization function with current rounding settings
            normalization_fn = self.get_normalization_fn()

            for model_part in self.model_parts:
                # Unwrap model if needed (DataParallel, DistributedDataParallel, FSDP)
                model = model_part
                if hasattr(model, 'module'):
                    model = model.module
                if hasattr(model, '_orig_mod'):  # FSDP wrapped
                    model = model._orig_mod

                # Each model knows how to normalize its own weights
                if hasattr(model, 'normalize_weights'):
                    model.normalize_weights(normalization_fn)
                else:
                    raise ValueError(
                        f"Model {type(model).__name__} does not implement normalize_weights() method. "
                        "All models must implement this method to support weight normalization."
                    )

        self._step_count += 1
