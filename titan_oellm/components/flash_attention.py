# Copyright (c) Titan-OELLM Custom Components.
# Flash Attention 2/3 wrapper with automatic GPU detection.

import torch
from torch import nn

# Will be populated with flash attention functions to allow in compiled graphs
_flash_attn_funcs_registered = False


def _register_flash_attn_in_graph():
    """Register flash attention functions as allowed in torch.compile graphs."""
    global _flash_attn_funcs_registered
    if _flash_attn_funcs_registered:
        return
    _flash_attn_funcs_registered = True

    # Try to register FA3 functions
    try:
        from flash_attn_3.flash_attn_interface import (
            flash_attn_varlen_func as fa3_varlen_func,
            flash_attn_func as fa3_func,
        )
        torch._dynamo.allow_in_graph(fa3_varlen_func)
        torch._dynamo.allow_in_graph(fa3_func)
    except ImportError:
        pass

    # Try to register FA2 functions
    try:
        from flash_attn import (
            flash_attn_varlen_func as fa2_varlen_func,
            flash_attn_func as fa2_func,
        )
        torch._dynamo.allow_in_graph(fa2_varlen_func)
        torch._dynamo.allow_in_graph(fa2_func)
    except ImportError:
        pass


def get_flash_attention_backend():
    """Detect GPU and return appropriate flash attention functions.

    Returns:
        Tuple of (varlen_func, standard_func, backend_name) where:
        - varlen_func: flash_attn_varlen_func for variable-length sequences
        - standard_func: flash_attn_func for fixed-length sequences
        - backend_name: "FA3" for Hopper, "FA2" for Ampere/Ada, None if unavailable
    """
    if not torch.cuda.is_available():
        return None, None, None

    capability = torch.cuda.get_device_capability()

    # Hopper: SM 90 - try FA3 first
    # FA3 package is named "flash_attn_3" when built from hopper/ subdirectory
    if capability >= (9, 0):
        try:
            from flash_attn_3.flash_attn_interface import (
                flash_attn_varlen_func as fa3_varlen_func,
                flash_attn_func as fa3_func,
            )
            return fa3_varlen_func, fa3_func, "FA3"
        except ImportError:
            pass  # Fall back to FA2

    # Ampere/Ada/Hopper fallback: SM 80+ - use FA2
    if capability >= (8, 0):
        try:
            from flash_attn import (
                flash_attn_varlen_func as fa2_varlen_func,
                flash_attn_func as fa2_func,
            )
            return fa2_varlen_func, fa2_func, "FA2"
        except ImportError:
            return None, None, None

    return None, None, None


class FlashAttentionWrapper(nn.Module):
    """Wrapper for Flash Attention 2/3 with variable-length sequence support.

    Automatically selects FA3 on Hopper GPUs (H100) and FA2 on Ampere/Ada GPUs.
    Supports document-level causal masking via cu_seqlens for Flash Attention.

    Input shapes:
        q, k, v: (batch, n_heads, seqlen, head_dim)  # Standard PyTorch format
        cu_seqlens: (num_docs + 1,) cumulative sequence lengths starting at 0
        max_seqlen: int, maximum sequence length in batch

    Output shape:
        (batch, n_heads, seqlen, head_dim)
    """

    def __init__(self):
        super().__init__()
        self.flash_attn_varlen_fn, self.flash_attn_fn, self.backend = get_flash_attention_backend()
        if self.flash_attn_varlen_fn is None:
            raise RuntimeError(
                "Flash Attention not available. Install flash-attn package:\n"
                "  pip install flash-attn --no-build-isolation"
            )
        # Register flash attention functions as allowed in torch.compile graphs
        _register_flash_attn_in_graph()

    def forward(
        self,
        q: torch.Tensor,  # (batch, n_heads, seqlen, head_dim)
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        scale: float = 1.0,
        causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass through Flash Attention.

        Args:
            q: Query tensor (batch, n_heads, seqlen, head_dim)
            k: Key tensor (batch, n_heads, seqlen, head_dim)
            v: Value tensor (batch, n_heads, seqlen, head_dim)
            cu_seqlens: Cumulative sequence lengths for document masking (optional)
            max_seqlen: Maximum sequence length (required if cu_seqlens provided)
            scale: Softmax scale factor (typically sqrt(head_dim) or 1.0)
            causal: Whether to apply causal masking

        Returns:
            Output tensor (batch, n_heads, seqlen, head_dim)
        """
        # Transpose from (batch, n_heads, seqlen, head_dim) to (batch, seqlen, n_heads, head_dim)
        # Flash Attention expects (batch, seqlen, n_heads, head_dim) or (total, n_heads, head_dim) for varlen
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        batch, seqlen, n_heads, head_dim = q.shape

        # Validate cu_seqlens and max_seqlen are consistent
        if cu_seqlens is not None:
            if max_seqlen is None:
                raise ValueError("max_seqlen must be provided when cu_seqlens is provided")
            if not cu_seqlens.is_cuda or cu_seqlens.device != q.device:
                cu_seqlens = cu_seqlens.to(q.device)
        
        if cu_seqlens is not None and self.flash_attn_varlen_fn is not None:
            # Variable-length attention with document masking
            # Flatten to (total_tokens, n_heads, head_dim) as required by flash_attn_varlen_func
            q_flat = q.reshape(-1, n_heads, head_dim)
            k_flat = k.reshape(-1, n_heads, head_dim)
            v_flat = v.reshape(-1, n_heads, head_dim)

            # Ensure cu_seqlens is on the same device
            cu_seqlens = cu_seqlens.to(q.device)

            output = self.flash_attn_varlen_fn(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=scale,
                causal=causal,
            )
            # Reshape back to (batch, seqlen, n_heads, head_dim)
            output = output.reshape(batch, seqlen, n_heads, head_dim)
        else:
            # Standard attention without document boundaries
            output = self.flash_attn_fn(q, k, v, softmax_scale=scale, causal=causal)

        # Transpose back to (batch, n_heads, seqlen, head_dim)
        return output.transpose(1, 2)
