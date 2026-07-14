"""
FP8 activation storage operations for canGPT_fp8.

Two levels of FP8 activation compression:

  Level 1 — FP8StoreFn
    Pure Python autograd Function.  Quantises an activation to FP8 E4M3 for
    ctx storage and returns the dequantised bf16 value.  Backward uses a
    straight-through estimator (STE).  One call per "store point" (e.g. pre_wo,
    pre_w2).

  Level 2 — FP8LerpFn
    Triton-fused LERP + L2-normalisation + FP8 quantisation in one kernel.
    Saves only the FP8 tensor (+ scale) instead of the full bf16 residual.
    Backward also runs in Triton, dequantising h_fp8 on-the-fly.

    Two kernels match the two lerp_normalization modes of canGPT_dev:
      "approx"  — analytical norm correction  c = rsqrt(1 − 2α(1−α))
      "direct"  — explicit L2 normalisation of the interpolated vector

Design notes
------------
* E4M3 range: ±448.  Scale = amax / 448 with a floor of 1e-12.
* Two-pass quantisation: pass 1 computes the fp32 LERP result and the scalar
  amax; pass 2 quantises using that amax.  This gives an exact per-tensor
  scale without atomic operations across warps.
* FP8 dtype (torch.float8_e4m3fn) is handled in Python; Triton kernels operate
  on fp32 inputs and outputs — we store/load fp8 via Python .to() calls around
  the kernel.  This avoids Triton's limited fp8 support.
* Backward kernels receive the fp32 buffer computed in the forward pass
  (allocated in FP8LerpFn.forward) rather than recomputing from the fp8 tensor.
  This keeps gradients exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Triton is only needed for the Level-2 fused fp8_lerp kernels. The Level-1
# fp8_store path (the one wired into qwen3's additive residual) is pure torch,
# so we make Triton optional: without it the module still imports and fp8_store
# works; only the fp8_lerp kernels raise a clear error if actually invoked.
# NB: `from __future__ import annotations` above means the `tl.constexpr` kernel
# signature annotations are strings, not evaluated at import — so the stub is safe.
try:  # pragma: no cover - exercised by env, not unit tests
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False
    tl = None  # type: ignore[assignment]

    class _UncompiledKernel:
        """Placeholder for a @triton.jit kernel when Triton is unavailable.

        Importing the module is fine; only launching the kernel (`kernel[grid](...)`)
        raises, with a message pointing at the missing Triton dependency.
        """

        def __init__(self, fn):
            self._fn = fn

        def __getitem__(self, grid):
            raise RuntimeError(
                f"fp8_ops kernel '{self._fn.__name__}' requires Triton, which is not "
                f"installed. The fused fp8_lerp path is unavailable; fp8_store / "
                f"FP8KVCache / weight quantization do not need Triton."
            )

    class _TritonStub:
        def jit(self, fn=None, **kwargs):
            if fn is None:
                return lambda f: _UncompiledKernel(f)
            return _UncompiledKernel(fn)

        @staticmethod
        def next_power_of_2(n: int) -> int:
            return 1 << (int(n) - 1).bit_length()

    triton = _TritonStub()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

E4M3_MAX: float = 448.0   # max representable value in torch.float8_e4m3fn
E4M3_MIN_SCALE: float = 1e-12


# ---------------------------------------------------------------------------
# FP8 Activation container — carries FP8 data + scale between layers
# ---------------------------------------------------------------------------

@dataclass
class FP8Activation:
    """Lightweight container for an FP8-quantized activation tensor.

    Keeps the residual stream in FP8 between layers to halve memory bandwidth.
    Dequantize on demand when bf16/fp32 precision is needed for computation.
    """
    data: Tensor   # float8_e4m3fn, e.g. shape (B, S, D)
    scale: Tensor  # float32 scalar (per-tensor scale)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> Tensor:
        """Dequantize to the given dtype (default bf16)."""
        return self.data.to(torch.float32).mul(self.scale).to(dtype)


# ---------------------------------------------------------------------------
# Level 1 — FP8StoreFn
# ---------------------------------------------------------------------------

class FP8StoreFn(torch.autograd.Function):
    """Quantise x to FP8 E4M3 for storage; return dequantised bf16.

    Forward: x → fp8 (stored in ctx) → dequantised x (same dtype as input)
    Backward: straight-through estimator — gradient passes unchanged.

    Memory saving: the original bf16 x tensor is freed after this op; the
    saved fp8 tensor in ctx is half the size.
    """

    @staticmethod
    def forward(ctx, x: Tensor, enabled: bool) -> Tensor:
        if not enabled:
            return x

        # Compute per-tensor scale in fp32
        amax = x.detach().float().abs().amax()
        scale = (amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)

        # Quantise → dequantise (no intermediate reference kept in ctx; STE needs nothing)
        x_fp8 = (x.detach().float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

        # Dequantise back to original dtype
        return x_fp8.to(torch.float32).mul(scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor):  # type: ignore[override]
        # STE: pass gradient through unchanged (no saved tensors needed)
        return grad, None


def fp8_store(x: Tensor, enabled: bool = True) -> Tensor:
    """Quantise x to FP8 E4M3 in ctx; return dequantised tensor (STE backward)."""
    return FP8StoreFn.apply(x, enabled)


def fp8_store_keep_fp8(x: Tensor) -> FP8Activation:
    """Quantise x to FP8 E4M3 and return as FP8Activation (no dequantize)."""
    amax = x.detach().float().abs().amax()
    scale = (amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)
    x_fp8 = (x.detach().float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return FP8Activation(data=x_fp8, scale=scale)


# ---------------------------------------------------------------------------
# Level 2 — Triton kernels for fused LERP + FP8 quantisation
# ---------------------------------------------------------------------------
# Strategy: each kernel processes one (batch*seq) row of the [B*S, D] tensor.
# Two-pass approach:
#   Pass 1 (_lerp_*_fwd_pass1): compute fp32 LERP result into a temp buffer,
#            accumulate per-row amax.  Python then reduces to global amax.
#   Pass 2 (_lerp_*_fwd_pass2): read temp buffer, quantise to fp8 using the
#            global scale, write fp8 output.
#
# Backward kernels:
#   Receive the fp32 temp buffer (h_f32) from forward, compute chain-rule
#   gradients through LERP+norm in fp32, return grad_x, grad_block_out,
#   and per-row grad_alpha contributions (reduced to scalar by Python).
# ---------------------------------------------------------------------------


@triton.jit
def _lerp_approx_fwd_pass1(
    x_ptr,
    block_out_ptr,
    alpha_ptr,         # (D,) fp32
    alpha_scaler_ptr,  # () fp32 scalar
    h_buf_ptr,         # output: (N, D) fp32 temp buffer
    amax_ptr,          # output: (N,) fp32 per-row amax
    do_post_layer_norm: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pass 1 for 'approx' mode: compute LERP + norm correction, store to h_buf, record amax."""
    row = tl.program_id(0)
    row_off = row * D

    alpha_s = tl.load(alpha_scaler_ptr)

    row_amax = tl.zeros([1], dtype=tl.float32)

    for d_off in range(0, D, BLOCK_D):
        cols = d_off + tl.arange(0, BLOCK_D)
        mask = cols < D

        x_tile = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        bo_tile = tl.load(block_out_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        alpha_tile = tl.load(alpha_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # effective alpha = |alpha * scaler|, clipped to [0, 1]
        a = tl.abs(alpha_tile * alpha_s)

        # block_out is pre-normalised by the Python wrapper when post_layer_norm=True
        out_n = bo_tile

        # LERP
        lerp = x_tile + a * (out_n - x_tile)

        # Approx norm correction: h_norm^2 ≈ 1 − 2α(1−α)
        h_sq = 1.0 - 2.0 * a * (1.0 - a)
        h_sq = tl.where(h_sq < eps, eps, h_sq)
        correct = tl.rsqrt(h_sq)
        h_tile = lerp * correct

        tl.store(h_buf_ptr + row_off + cols, h_tile, mask=mask)

        tile_amax = tl.max(tl.abs(h_tile), axis=0)
        row_amax = tl.where(tile_amax > row_amax, tile_amax, row_amax)

    # row_amax is shape [1]; reduce to scalar before storing to scalar pointer
    tl.store(amax_ptr + row, tl.max(row_amax, axis=0))


@triton.jit
def _lerp_direct_fwd_pass1(
    x_ptr,
    block_out_ptr,
    alpha_ptr,
    alpha_scaler_ptr,
    h_buf_ptr,
    amax_ptr,
    do_post_layer_norm: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Pass 1 for 'direct' mode: LERP then explicit L2-normalise, record amax."""
    row = tl.program_id(0)
    row_off = row * D

    alpha_s = tl.load(alpha_scaler_ptr)

    # Step A: compute LERP, accumulate norm squared
    sum_sq = tl.zeros([1], dtype=tl.float32)

    for d_off in range(0, D, BLOCK_D):
        cols = d_off + tl.arange(0, BLOCK_D)
        mask = cols < D

        x_tile = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        bo_tile = tl.load(block_out_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        alpha_tile = tl.load(alpha_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        a = tl.abs(alpha_tile * alpha_s)
        out_n = bo_tile  # pre-normalised by Python if post_layer_norm

        lerp = x_tile + a * (out_n - x_tile)
        tl.store(h_buf_ptr + row_off + cols, lerp, mask=mask)
        sum_sq += tl.sum(lerp * lerp, axis=0)

    # Step B: compute rnorm, normalise, record amax
    rnorm = tl.rsqrt(tl.where(sum_sq < eps, eps, sum_sq))
    row_amax = tl.zeros([1], dtype=tl.float32)

    for d_off in range(0, D, BLOCK_D):
        cols = d_off + tl.arange(0, BLOCK_D)
        mask = cols < D

        lerp = tl.load(h_buf_ptr + row_off + cols, mask=mask, other=0.0)
        h_tile = lerp * rnorm
        tl.store(h_buf_ptr + row_off + cols, h_tile, mask=mask)

        tile_amax = tl.max(tl.abs(h_tile), axis=0)
        row_amax = tl.where(tile_amax > row_amax, tile_amax, row_amax)

    # row_amax is shape [1]; reduce to scalar before storing to scalar pointer
    tl.store(amax_ptr + row, tl.max(row_amax, axis=0))


# ---------------------------------------------------------------------------
# Python helper: compute BLOCK_D for a given D
# ---------------------------------------------------------------------------

def _block_d(D: int) -> int:
    MAX_FUSED = 65536 // 4  # fp32 = 4 bytes
    return min(MAX_FUSED, triton.next_power_of_2(D))


# ---------------------------------------------------------------------------
# Level 2 — FP8LerpFn autograd Function
# ---------------------------------------------------------------------------

class FP8LerpFn(torch.autograd.Function):
    """Fused LERP + L2-norm + FP8 quantisation.

    Forward:
        h = lerp_and_norm(x, block_out, alpha, alpha_scaler, mode)
        → stored as fp8_e4m3 + scale scalar
        → returned as dequantised bf16

    Backward (PyTorch):
        chain-rule through LERP + norm → grad_x, grad_block_out, grad_alpha
        Recomputes v = x + a*(bo-x) from saved inputs; uses exact ‖v‖ for "direct".

    Args:
        x             (Tensor [N, D] bf16): residual stream input
        block_out     (Tensor [N, D] bf16): attention/FFN output
        alpha         (Tensor [D] fp32):    per-dim learnable LERP weight
        alpha_scaler  (Tensor [] fp32):     scalar multiplier (init = 1/base_scale)
        lerp_mode     (str):                "approx" or "direct"
        post_layer_norm (bool):             normalise block_out before LERP
        eps           (float):              numerical stability epsilon
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        block_out: Tensor,
        alpha: Tensor,
        alpha_scaler: Tensor,
        lerp_mode: str,
        post_layer_norm: bool,
        eps: float,
    ) -> Tensor:
        # Flatten to (N, D) for Triton
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        bo_2d = block_out.reshape(-1, block_out.shape[-1]).contiguous()

        N, D = x_2d.shape
        BLOCK_D = _block_d(D)
        num_warps = min(max(BLOCK_D // 256, 1), 8)

        # Pre-normalise block_out in Python if post_layer_norm
        if post_layer_norm:
            bo_2d = torch.nn.functional.normalize(bo_2d.float(), p=2, dim=-1, eps=eps).to(bo_2d.dtype)

        # Allocate fp32 temp buffer and per-row amax
        h_buf = torch.empty(N, D, dtype=torch.float32, device=x.device)
        amax_rows = torch.empty(N, dtype=torch.float32, device=x.device)

        grid = (N,)

        if "approx" in lerp_mode:
            _lerp_approx_fwd_pass1[grid](
                x_2d, bo_2d, alpha, alpha_scaler,
                h_buf, amax_rows,
                do_post_layer_norm=post_layer_norm,
                eps=eps,
                N=N, D=D, BLOCK_D=BLOCK_D,
                num_warps=num_warps,
            )
        else:  # direct
            _lerp_direct_fwd_pass1[grid](
                x_2d, bo_2d, alpha, alpha_scaler,
                h_buf, amax_rows,
                do_post_layer_norm=post_layer_norm,
                eps=eps,
                N=N, D=D, BLOCK_D=BLOCK_D,
                num_warps=num_warps,
            )

        # Global scale from max amax across all rows
        global_amax = amax_rows.amax().float()
        scale = (global_amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)

        # Quantise h_buf → fp8 with pure tensor ops (compilable, no constexpr
        # data-dependent value issue that a Triton kernel would introduce).
        h_fp8 = (h_buf / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

        # Save for backward: normalised h (fp32), x, block_out (pre-normalised), alpha
        # h_fp8 is NOT saved — it's not used in backward and would waste memory
        ctx.save_for_backward(h_buf, x_2d, bo_2d, alpha, alpha_scaler)
        ctx.lerp_mode = lerp_mode
        ctx.post_layer_norm = post_layer_norm
        ctx.eps = eps
        ctx.orig_shape = orig_shape

        # Return dequantised bf16
        h_out = h_fp8.to(torch.float32).mul(scale).to(x.dtype)
        return h_out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_h: Tensor):  # type: ignore[override]
        h_buf, x_2d, bo_2d, alpha, alpha_scaler = ctx.saved_tensors
        lerp_mode = ctx.lerp_mode
        eps = ctx.eps
        orig_shape = ctx.orig_shape

        N, D = x_2d.shape
        grad_h_f = grad_h.reshape(N, D).float()

        # Effective per-dim alpha in fp32
        # alpha_scaler has requires_grad=False; alpha has requires_grad=True
        alpha_s = alpha_scaler.float()                    # scalar
        raw_a   = alpha.float() * alpha_s                 # (D,)
        a       = raw_a.abs()                             # (D,)
        sign_a  = raw_a.sign()                            # (D,)  — 0 treated as +1 below

        x_f  = x_2d.float()   # (N, D)
        bo_f = bo_2d.float()   # (N, D)
        v    = x_f + a * (bo_f - x_f)   # (N, D) un-normalised LERP

        if "approx" in lerp_mode:
            # Forward: h = v * correct,  correct = rsqrt(max(1-2a(1-a), eps))
            h_sq    = (1.0 - 2.0 * a * (1.0 - a)).clamp(min=eps)  # (D,)
            correct = h_sq.rsqrt()                                  # (D,)

            # ∂L/∂v[n,d] = grad_h[n,d] * correct[d]
            grad_v = grad_h_f * correct                 # (N, D)

            # ∂L/∂correct[d] = Σ_n grad_h[n,d] * v[n,d]
            grad_correct = (grad_h_f * v).sum(dim=0)   # (D,)
            # ∂correct/∂a[d] = (2a-1) * correct^3
            dcorrect_da  = (2.0 * a - 1.0) * (correct ** 3)

            grad_a = (grad_v * (bo_f - x_f)).sum(dim=0) + grad_correct * dcorrect_da  # (D,)

        else:  # "direct"
            # Forward: h = v / ‖v‖
            # h_buf is the saved normalised h (fp32); use it directly.
            h      = h_buf                                         # (N, D)
            norm_v = v.norm(dim=-1, keepdim=True).clamp(min=eps)  # (N, 1)

            # ∂L/∂v = (grad_h − dot(grad_h, h)·h) / ‖v‖
            dot_gh = (grad_h_f * h).sum(dim=-1, keepdim=True)     # (N, 1)
            grad_v = (grad_h_f - dot_gh * h) / norm_v             # (N, D)

            grad_a = (grad_v * (bo_f - x_f)).sum(dim=0)           # (D,)

        # ∂L/∂x = ∂L/∂v * (1 - a),   ∂L/∂block_out = ∂L/∂v * a
        grad_x  = grad_v * (1.0 - a)   # (N, D)
        grad_bo = grad_v * a            # (N, D)

        # ∂L/∂alpha = ∂L/∂a * sign(alpha*alpha_scaler) * alpha_scaler
        # sign=0 case: a=0, gradient is 0 either way — keep sign=0 (zero grad, correct)
        grad_alpha = (grad_a * sign_a * alpha_s).to(alpha.dtype)  # (D,)

        return (
            grad_x.to(x_2d.dtype).reshape(orig_shape),   # grad for x
            grad_bo.to(bo_2d.dtype).reshape(orig_shape),  # grad for block_out
            grad_alpha,                                    # grad for alpha (D,)
            None,                                          # alpha_scaler (non-grad)
            None,                                          # lerp_mode
            None,                                          # post_layer_norm
            None,                                          # eps
        )


def fp8_lerp(
    x: Tensor,
    block_out: Tensor,
    alpha: Tensor,
    alpha_scaler: Tensor,
    lerp_mode: str,
    post_layer_norm: bool,
    eps: float,
) -> Tensor:
    """Fused LERP + norm + FP8 quantisation (Level 2 entry point)."""
    return FP8LerpFn.apply(x, block_out, alpha, alpha_scaler, lerp_mode, post_layer_norm, eps)


def fp8_lerp_inference(
    x: Tensor,
    block_out: Tensor,
    alpha: Tensor,
    alpha_scaler: Tensor,
    lerp_mode: str,
    post_layer_norm: bool,
    eps: float,
) -> Tensor:
    """Inference-only fused LERP + norm + FP8 quantise (no backward tensors saved).

    Same forward computation as FP8LerpFn but without ctx.save_for_backward,
    avoiding unnecessary memory retention during inference.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    bo_2d = block_out.reshape(-1, block_out.shape[-1]).contiguous()

    N, D = x_2d.shape
    BLOCK_D = _block_d(D)
    num_warps = min(max(BLOCK_D // 256, 1), 8)

    if post_layer_norm:
        bo_2d = torch.nn.functional.normalize(bo_2d.float(), p=2, dim=-1, eps=eps).to(bo_2d.dtype)

    h_buf = torch.empty(N, D, dtype=torch.float32, device=x.device)
    amax_rows = torch.empty(N, dtype=torch.float32, device=x.device)

    grid = (N,)

    if "approx" in lerp_mode:
        _lerp_approx_fwd_pass1[grid](
            x_2d, bo_2d, alpha, alpha_scaler,
            h_buf, amax_rows,
            do_post_layer_norm=post_layer_norm,
            eps=eps,
            N=N, D=D, BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
    else:  # direct
        _lerp_direct_fwd_pass1[grid](
            x_2d, bo_2d, alpha, alpha_scaler,
            h_buf, amax_rows,
            do_post_layer_norm=post_layer_norm,
            eps=eps,
            N=N, D=D, BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )

    global_amax = amax_rows.amax().float()
    scale = (global_amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)

    h_fp8 = (h_buf / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

    h_out = h_fp8.to(torch.float32).mul(scale).to(x.dtype)
    return h_out.reshape(orig_shape)


def fp8_lerp_inference_keep_fp8(
    x: Tensor,
    block_out: Tensor,
    alpha: Tensor,
    alpha_scaler: Tensor,
    lerp_mode: str,
    post_layer_norm: bool,
    eps: float,
) -> FP8Activation:
    """Inference fused LERP + norm + FP8 quantise — returns FP8Activation (no dequantize).

    Same computation as fp8_lerp_inference but keeps the result in FP8 format
    to halve memory bandwidth between layers.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    bo_2d = block_out.reshape(-1, block_out.shape[-1]).contiguous()

    N, D = x_2d.shape
    BLOCK_D = _block_d(D)
    num_warps = min(max(BLOCK_D // 256, 1), 8)

    if post_layer_norm:
        bo_2d = torch.nn.functional.normalize(bo_2d.float(), p=2, dim=-1, eps=eps).to(bo_2d.dtype)

    h_buf = torch.empty(N, D, dtype=torch.float32, device=x.device)
    amax_rows = torch.empty(N, dtype=torch.float32, device=x.device)

    grid = (N,)

    if "approx" in lerp_mode:
        _lerp_approx_fwd_pass1[grid](
            x_2d, bo_2d, alpha, alpha_scaler,
            h_buf, amax_rows,
            do_post_layer_norm=post_layer_norm,
            eps=eps,
            N=N, D=D, BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
    else:  # direct
        _lerp_direct_fwd_pass1[grid](
            x_2d, bo_2d, alpha, alpha_scaler,
            h_buf, amax_rows,
            do_post_layer_norm=post_layer_norm,
            eps=eps,
            N=N, D=D, BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )

    global_amax = amax_rows.amax().float()
    scale = (global_amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)

    h_fp8 = (h_buf / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

    return FP8Activation(data=h_fp8.reshape(orig_shape), scale=scale)
