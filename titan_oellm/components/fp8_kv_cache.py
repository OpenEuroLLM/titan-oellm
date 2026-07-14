"""
FP8 KV Cache for inference.

Stores K and V tensors in float8_e4m3fn with per-head-per-position scales,
halving KV cache memory compared to bf16 storage.  Dequantizes to bf16
when the cached K,V are needed for attention computation.

Usage (inside Attention.forward):
    # First call: creates cache
    cache = FP8KVCache(max_batch, max_seq, n_kv_heads, head_dim, device)

    # Each forward pass: update cache with new K,V
    xk_full, xv_full = cache.update(xk_new, xv_new, start_pos)
    # xk_full, xv_full are bf16 tensors covering positions [0, start_pos + seq_len)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

E4M3_MAX: float = 448.0
E4M3_MIN_SCALE: float = 1e-12


class FP8KVCache(nn.Module):
    """KV cache that stores keys and values in FP8 E4M3.

    Quantization is per-head-per-position (one scale per (batch, position, head)),
    which preserves accuracy across heads with different value ranges.

    Args:
        max_batch: Maximum batch size.
        max_seq: Maximum sequence length.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension per head.
        device: Device for the cache buffers.
    """

    def __init__(
        self,
        max_batch: int,
        max_seq: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        # FP8 data buffers: (max_batch, max_seq, n_kv_heads, head_dim)
        self.register_buffer(
            "k_data",
            torch.zeros(max_batch, max_seq, n_kv_heads, head_dim,
                        dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_data",
            torch.zeros(max_batch, max_seq, n_kv_heads, head_dim,
                        dtype=torch.float8_e4m3fn, device=device),
            persistent=False,
        )

        # Per-head-per-position scales: (max_batch, max_seq, n_kv_heads, 1)
        self.register_buffer(
            "k_scale",
            torch.ones(max_batch, max_seq, n_kv_heads, 1,
                       dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "v_scale",
            torch.ones(max_batch, max_seq, n_kv_heads, 1,
                       dtype=torch.float32, device=device),
            persistent=False,
        )

        self.seq_len = 0

    def reset(self):
        """Reset cache for a new sequence."""
        self.seq_len = 0

    @staticmethod
    def _quantize(x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize (batch, seq, heads, dim) to FP8 with per-head-per-pos scales.

        Returns:
            (data_fp8, scale) where data_fp8 is float8_e4m3fn and
            scale is float32 with shape (..., 1) for broadcasting.
        """
        # Per-head-per-position amax: reduce over head_dim only
        amax = x.float().abs().amax(dim=-1, keepdim=True)  # (B, S, H, 1)
        scale = (amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)
        data_fp8 = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return data_fp8, scale

    @staticmethod
    def _dequantize(data_fp8: Tensor, scale: Tensor, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        """Dequantize FP8 data back to the given dtype."""
        return data_fp8.to(torch.float32).mul(scale).to(dtype)

    def update(
        self,
        k_new: Tensor,
        v_new: Tensor,
        start_pos: int,
    ) -> tuple[Tensor, Tensor]:
        """Quantize new K,V to FP8 and append to cache.

        Args:
            k_new: New key tensor, shape (batch, seq_new, n_kv_heads, head_dim), bf16.
            v_new: New value tensor, same shape as k_new.
            start_pos: Starting position in the sequence for these tokens.

        Returns:
            (k_full, v_full): Dequantized K,V covering [0, start_pos + seq_new),
            in bf16, shape (batch, total_seq, n_kv_heads, head_dim).
        """
        seq_new = k_new.shape[1]
        end_pos = start_pos + seq_new

        # Quantize new tokens
        k_fp8, k_sc = self._quantize(k_new)
        v_fp8, v_sc = self._quantize(v_new)

        # Store in cache
        batch = k_new.shape[0]
        self.k_data[:batch, start_pos:end_pos] = k_fp8
        self.v_data[:batch, start_pos:end_pos] = v_fp8
        self.k_scale[:batch, start_pos:end_pos] = k_sc
        self.v_scale[:batch, start_pos:end_pos] = v_sc

        self.seq_len = end_pos

        # Dequantize full cache for attention
        k_full = self._dequantize(
            self.k_data[:batch, :end_pos],
            self.k_scale[:batch, :end_pos],
            dtype=k_new.dtype,
        )
        v_full = self._dequantize(
            self.v_data[:batch, :end_pos],
            self.v_scale[:batch, :end_pos],
            dtype=v_new.dtype,
        )

        return k_full, v_full
