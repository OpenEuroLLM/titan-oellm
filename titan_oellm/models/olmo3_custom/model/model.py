# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

try:
    # Import directly from submodule to avoid ring_flash_attn/__init__.py pulling
    # in transformers adapters (incompatible with transformers>=5.x).
    from ring_flash_attn.zigzag_ring_flash_attn_varlen import (
        zigzag_ring_flash_attn_varlen_func as _zigzag_ring_varlen_func,
        zigzag_ring_flash_attn_varlen_forward as _zigzag_ring_varlen_forward,
        zigzag_ring_flash_attn_varlen_backward as _zigzag_ring_varlen_backward,
        get_half_index as _get_half_index,
    )

    # ProcessGroup cannot be passed as a custom_op argument (not a tensor), so we
    # store it in a module-level variable set once by parallelize_olmo3_custom.
    _ring_attn_cp_group: dist.ProcessGroup | None = None

    def register_ring_attn_cp_group(group: dist.ProcessGroup) -> None:
        global _ring_attn_cp_group
        _ring_attn_cp_group = group

    # Register as a proper torch custom op so that:
    #   1. The selective-AC checkpoint policy can recognise it as an op to SAVE
    #      (preventing expensive cross-GPU ring-attention recomputation in op mode).
    #   2. Dynamo can trace through it via the fake/meta implementation, enabling
    #      fullgraph torch.compile on the surrounding model code.
    # Returns (out, softmax_lse) so setup_context can save lse for the backward
    # without re-running the forward pass.
    @torch.library.custom_op("ring_flash_attn::zigzag_varlen", mutates_args=())
    def _zigzag_ring_varlen_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        softmax_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert _ring_attn_cp_group is not None, (
            "ring_flash_attn CP group not registered — call register_ring_attn_cp_group "
            "from parallelize_olmo3_custom before running the model."
        )
        half_index0 = _get_half_index(cu_seqlens, front=True)
        half_index1 = _get_half_index(cu_seqlens, front=False)
        out, lse = _zigzag_ring_varlen_forward(
            _ring_attn_cp_group, q, k, v, cu_seqlens, max_seqlen,
            half_index0, half_index1,
            softmax_scale=softmax_scale, dropout_p=0.0, causal=True,
        )
        return out, lse

    @_zigzag_ring_varlen_op.register_fake
    def _zigzag_ring_varlen_op_fake(q, k, v, cu_seqlens, max_seqlen, softmax_scale):
        # out: (total_seqlen, n_heads, head_dim) — same as q
        # lse: (n_heads, total_seqlen) — log-sum-exp from flash attention
        n_heads = q.shape[1]
        total_seqlen = q.shape[0]
        lse = torch.empty((n_heads, total_seqlen), dtype=torch.float32, device=q.device)
        return torch.empty_like(q), lse

    def _zigzag_setup_context(ctx, inputs, output):
        q, k, v, cu_seqlens, max_seqlen, softmax_scale = inputs
        out, lse = output
        half_index0 = _get_half_index(cu_seqlens, front=True)
        half_index1 = _get_half_index(cu_seqlens, front=False)
        ctx.is_half_index_tensor = isinstance(half_index0, torch.Tensor)
        if ctx.is_half_index_tensor:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens, half_index0, half_index1)
        else:
            ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
            ctx.half_index0 = half_index0
            ctx.half_index1 = half_index1
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale

    def _zigzag_backward(ctx, dout, _dlse):
        # _dlse is the grad w.r.t. lse — lse is only used internally so this is zeros.
        if ctx.is_half_index_tensor:
            q, k, v, out, lse, cu_seqlens, half_index0, half_index1 = ctx.saved_tensors
        else:
            q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
            half_index0 = ctx.half_index0
            half_index1 = ctx.half_index1
        dq, dk, dv = _zigzag_ring_varlen_backward(
            _ring_attn_cp_group, dout, q, k, v, out, lse,
            cu_seqlens, ctx.max_seqlen, half_index0, half_index1,
            softmax_scale=ctx.softmax_scale, dropout_p=0.0, causal=True,
        )
        return dq, dk, dv, None, None, None  # no grad for cu_seqlens, max_seqlen, softmax_scale

    _zigzag_ring_varlen_op.register_autograd(_zigzag_backward, setup_context=_zigzag_setup_context)

    _RING_FLASH_ATTN_AVAILABLE = True
except ImportError:
    _RING_FLASH_ATTN_AVAILABLE = False
    _zigzag_ring_varlen_func = None
    _zigzag_ring_varlen_op = None

    def register_ring_attn_cp_group(group: dist.ProcessGroup) -> None:  # type: ignore[misc]
        pass

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    _LIGER_AVAILABLE = True
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None
    LigerRMSNorm = None
    LigerSiLUMulFunction = None
    _LIGER_AVAILABLE = False


def _make_rms_norm(dim: int, eps: float, use_liger: bool) -> nn.Module:
    """RMSNorm factory. Liger variant runs as a single fused Triton kernel.

    in_place=False: the default in_place=True would mutate inside selective-AC
    checkpointed regions, which is unsafe (see PyTorch's checkpoint warning
    about in-place ops in the checkpointed region).
    """
    if use_liger:
        if not _LIGER_AVAILABLE:
            raise ImportError(
                "use_liger_kernels=True but liger-kernel is not installed."
            )
        norm = LigerRMSNorm(dim, eps=eps, in_place=False)
        # LigerRMSNorm has no reset_parameters; the model's init_weights
        # calls it on every norm. Attach a method that re-inits weight to
        # ones (LigerRMSNorm uses init_fn='ones' at construction).
        def _reset(self):
            with torch.no_grad():
                self.weight.fill_(1.0)
        norm.reset_parameters = _reset.__get__(norm, type(norm))
        return norm
    return nn.RMSNorm(dim, eps=eps)

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.moe import MoE
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import Olmo3CustomModelArgs


# Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_positional_embeddings.py
def precompute_rope_cache(
    dim: int,
    max_seq_len: int,
    base: float = 1_000_000.0,
    scaling_factor: float = 1.0,
    old_context_len: int = 8192,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    attention_factor = 1.0

    # OLMo-core YaRN formulation: blend extrapolation/interpolation inverse
    # frequencies by a dimension-wise linear ramp and apply attention rescaling.
    if scaling_factor > 1.0:
        inv_freq_extrapolation = freqs
        inv_freq_interpolation = inv_freq_extrapolation / scaling_factor

        half_dim = inv_freq_extrapolation.shape[0]
        idx = torch.arange(half_dim, dtype=torch.float32, device=freqs.device)

        def _dim_from_rot(n_rot: int) -> float:
            return (
                dim
                * math.log(old_context_len / (n_rot * 2.0 * math.pi))
                / (2.0 * math.log(base))
            )

        low = max(int(math.floor(_dim_from_rot(beta_fast))), 0)
        high = min(int(math.ceil(_dim_from_rot(beta_slow))), half_dim - 1)

        span = max(high - low, 1e-3)
        ramp = ((idx - low) / span).clamp_(0, 1)

        freqs = inv_freq_interpolation * ramp + inv_freq_extrapolation * (1.0 - ramp)
        attention_factor = 0.1 * math.log(scaling_factor) + 1.0

    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.outer(t, freqs).float()

    # We cache the cos and sin embeddings instead of the IDs. This helps
    # ensure we have correct behavior when training with bf16
    # Size: [max_seq_len, (dim * 2)]
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1) * attention_factor
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Reshape frequency tensor (represented by cos, sin) for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, head_dim * 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        rope_cache (torch.Tensor): RoPE tensor (cos and sin) to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache.
            Shape is (1, seqlen) or (bz, seqlen). Defaults to None.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        # The shape of rope_cache is (seqlen, head_dim * 2) because we concate cos and sin
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        rope_cache = rope_cache[positions.squeeze(0)]
        # The shape of rope_cache is (seqlen, head_dim * 2)
        assert rope_cache.shape == (seqlen, head_dim * 2)
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    else:
        assert positions.shape == (bz, seqlen)
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        # The shape of rope_cache is (bz, seqlen, 1, head_dim * 2)
        assert rope_cache.shape == (bz, seqlen, 1, head_dim * 2)
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # input tensor x has shape [bsz, seq_len, num_heads, head_dim]
    head_dim = xq.shape[-1]

    rope_cache = reshape_for_broadcast(rope_cache, xq, positions)

    # [bsz, seq_len, 1, head_dim]
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)

    # xq:  [bsz, seq_len, num_heads, head_dim]
    # xk:  [bsz, seq_len, num_kv_heads, head_dim]
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    q_norm: nn.RMSNorm | None
    k_norm: nn.RMSNorm | None

    def __init__(
        self,
        model_args: Olmo3CustomModelArgs,
        use_sliding_attention: bool = True,
    ):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_type = getattr(model_args, "attn_type", "sdpa")
        self.sliding_window = getattr(model_args, "sliding_window", 0)
        self.use_sliding_attention = use_sliding_attention

        # RMSNorm added here to the here to include the q-k norm
        # This is one of the main differences between Llama3 and OLMo-3
        if model_args.qk_norm:
            q_norm_dim = model_args.n_heads * self.head_dim
            k_norm_dim = self.n_kv_heads * self.head_dim
            use_liger = getattr(model_args, "use_liger_kernels", False)
            self.q_norm = _make_rms_norm(q_norm_dim, model_args.norm_eps, use_liger)
            self.k_norm = _make_rms_norm(k_norm_dim, model_args.norm_eps, use_liger)
        else:
            self.q_norm = None
            self.k_norm = None

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        # Set after model creation by parallelize_olmo3_custom when attn_type="ring_varlen".
        self.cp_pg: dist.ProcessGroup | None = None

        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                # pyrefly: ignore [bad-assignment]
                self.inner_attention = VarlenAttentionWrapper()
            case "ring_varlen":
                if not _RING_FLASH_ATTN_AVAILABLE:
                    raise ImportError(
                        "ring_varlen attention requires the ring-flash-attn package. "
                        "Install it with: pip install ring-flash-attn"
                    )
            case "sdpa":
                # pyrefly: ignore [bad-assignment]
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask), (
                    "flex attention requires a BlockMask attention_masks"
                )
                output = self.inner_attention(
                    xq, xk, xv, block_mask=attention_masks, scale=self.scaling
                )
            case "varlen":
                # TODO: pass self.scaling into varlen attention
                assert isinstance(attention_masks, VarlenMetadata), (
                    "varlen attention requires VarlenMetadata attention_masks"
                )
                output = self.inner_attention(
                    xq,
                    xk,
                    xv,
                    self.head_dim,
                    attention_masks,
                )
            case "ring_varlen":
                # Ring attention: each rank holds a zigzag-sharded slice of the
                # sequence.  Per-rank cu_seqlens are already set in VarlenMetadata
                # by SFTTrainer.post_dataloading_process.  The ring kernel handles
                # all cross-rank K/V communication internally via group.
                assert isinstance(attention_masks, VarlenMetadata), (
                    "ring_varlen attention requires VarlenMetadata attention_masks"
                )
                assert self.cp_pg is not None, (
                    "ring_varlen requires cp_pg to be set by parallelize_olmo3_custom"
                )
                n_local_heads = xq.shape[1]
                # Repack from (bs, n_heads, seqlen, head_dim) → (bs*seqlen, n_heads, head_dim)
                # matching the layout expected by flash-attn varlen kernels.
                # xq/xk/xv are already transposed to (bs, n_heads, seqlen, head_dim);
                # .transpose(1,2) un-does that to (bs, seqlen, n_heads, head_dim) before
                # flattening — same as VarlenAttentionWrapper.
                xq_packed = xq.transpose(1, 2).reshape(-1, n_local_heads, self.head_dim)
                xk_packed = xk.transpose(1, 2).reshape(-1, n_local_heads, self.head_dim)
                xv_packed = xv.transpose(1, 2).reshape(-1, n_local_heads, self.head_dim)
                # cu_seqlens is already on GPU — moved in SFTTrainer.post_dataloading_process
                # to avoid a CPU→GPU transfer inside the compiled graph.
                cu_seqlens = attention_masks.cu_seq_q
                max_seqlen = int(attention_masks.max_q)
                output, _lse = _zigzag_ring_varlen_op(
                    xq_packed,
                    xk_packed,
                    xv_packed,
                    cu_seqlens,
                    max_seqlen,
                    self.scaling,
                )
            case "sdpa":
                assert attention_masks is None
                if self.use_sliding_attention and 0 < self.sliding_window < seqlen:
                    q_positions = torch.arange(seqlen, device=xq.device).view(seqlen, 1)
                    kv_positions = torch.arange(seqlen, device=xq.device).view(1, seqlen)
                    # True entries are allowed attention locations.
                    sw_mask = (kv_positions <= q_positions) & (
                        q_positions - kv_positions < self.sliding_window
                    )
                    output = F.scaled_dot_product_attention(
                        xq,
                        xk,
                        xv,
                        attn_mask=sw_mask.view(1, 1, seqlen, seqlen),
                        scale=self.scaling,
                        is_causal=False,
                    )
                else:
                    output = self.inner_attention(xq, xk, xv, scale=self.scaling)
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        use_liger: bool = False,
    ):
        super().__init__()

        # Hidden dimension is directly added from the model argsS
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self._use_liger_swiglu = use_liger
        if use_liger and not _LIGER_AVAILABLE:
            raise ImportError("use_liger_kernels=True but liger-kernel is not installed.")

    def forward(self, x):
        if self._use_liger_swiglu:
            # Fused silu(w1) * w3 in one Triton kernel; saves one elementwise pass.
            return self.w2(LigerSiLUMulFunction.apply(self.w1(x), self.w3(x)))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(
        self,
        layer_id: int,
        model_args: Olmo3CustomModelArgs,
        use_sliding_attention: bool = True,
    ):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(
            model_args,
            use_sliding_attention=use_sliding_attention,
        )

        self.moe_enabled = model_args.moe_enabled
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=model_args.hidden_dim,
                use_liger=getattr(model_args, "use_liger_kernels", False),
            )
        use_liger = getattr(model_args, "use_liger_kernels", False)
        self.attention_norm = _make_rms_norm(model_args.dim, model_args.norm_eps, use_liger)
        self.ffn_norm = _make_rms_norm(model_args.dim, model_args.norm_eps, use_liger)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            rope_cache (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # OLMo-3 ReorderedNormTransformerBlock: norms are applied to the *output*
        # of each sublayer (attention / feed-forward), not to the input. See
        # OLMo-core/src/olmo_core/nn/transformer/block.py:ReorderedNormTransformerBlock.
        x = x + self.attention_norm(
            self.attention(x, rope_cache, attention_masks, positions)
        )

        if self.moe_enabled:
            x = x + self.ffn_norm(self.moe(x))
        else:
            x = x + self.ffn_norm(self.feed_forward(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class LigerLMHead(nn.Module):
    """Liger fused linear + cross-entropy LM head.

    Same param shape as nn.Linear(dim, vocab_size, bias=False) so FSDP/TP
    sharding and checkpoint loading are unchanged. Forward branches on
    whether labels are passed:
      - labels is None → returns logits via plain F.linear (eval / generation).
      - labels given   → returns scalar loss via LigerFusedLinearCrossEntropyLoss,
                         which never materializes the (B*T, V) logits tensor.

    The loss is reduction="mean" with ignore_index=-100, matching torchtitan's
    default cross_entropy contract.
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        if not _LIGER_AVAILABLE:
            raise ImportError(
                "use_liger_kernels=True but liger-kernel is not installed. "
                "pip install liger-kernel"
            )
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        self._liger_ce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean")

    def forward(self, h: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            return F.linear(h, self.weight)
        # h: (B, T, D) → (B*T, D); labels: (B, T) → (B*T,)
        h_flat = h.reshape(-1, h.shape[-1])
        labels_flat = labels.reshape(-1)
        return self._liger_ce(self.weight, h_flat, labels_flat)


class Olmo3Model(nn.Module, ModelProtocol):
    """
    Olmo3Model Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: Olmo3CustomModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Default to OLMo-3's layer pattern (see Olmo3CustomModelArgs.default_uses_sliding).
        # Single source of truth shared with get_nparams_and_flops so MFU
        # accounting matches the attention pattern that actually executes.
        self._layer_uses_sliding_attention = model_args.default_uses_sliding()
        if model_args.layer_types:
            if len(model_args.layer_types) != model_args.n_layers:
                raise ValueError(
                    f"layer_types length ({len(model_args.layer_types)}) must match "
                    f"n_layers ({model_args.n_layers})"
                )
            normalized_layer_modes = []
            for idx, layer_type in enumerate(model_args.layer_types):
                if layer_type == "sliding_attention":
                    normalized_layer_modes.append(True)
                elif layer_type == "full_attention":
                    normalized_layer_modes.append(False)
                else:
                    raise ValueError(
                        f"Unknown layer_types[{idx}]='{layer_type}'. "
                        "Expected 'sliding_attention' or 'full_attention'."
                    )
            self._layer_uses_sliding_attention = normalized_layer_modes

        self._has_mixed_layer_attention = any(
            self._layer_uses_sliding_attention
        ) and not all(self._layer_uses_sliding_attention)

        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id,
                model_args,
                use_sliding_attention=self._layer_uses_sliding_attention[layer_id],
            )
        self.norm = _make_rms_norm(
            model_args.dim,
            model_args.norm_eps,
            getattr(model_args, "use_liger_kernels", False),
        )

        if getattr(model_args, "use_liger_kernels", False):
            self.output = LigerLMHead(model_args.dim, model_args.vocab_size)
        else:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                # pyrefly: ignore [not-callable]
                layer.init_weights(buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        # If weight tying is enabled, we don't need to initialize the output layer
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
            self.model_args.rope_scaling_factor,
            self.model_args.rope_old_context_len,
            self.model_args.rope_beta_fast,
            self.model_args.rope_beta_slow,
        )

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        if self._has_mixed_layer_attention and self.model_args.sliding_window > 0:
            raise NotImplementedError(
                "attn_type='flex' does not support mixed layer_types with sliding_window>0; "
                "use attn_type='sdpa' for per-layer sliding/full attention behavior"
            )

        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )
        if (
            getattr(self.model_args, "sliding_window", 0) > 0
            and all(self._layer_uses_sliding_attention)
        ):
            mask_mods.append(get_sliding_window_mask_mod(self.model_args.sliding_window))
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.model_args.attn_type:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.model_args.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal \
                        attention mask type, got {self.model_args.attn_mask_type}"
                    )
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise NotImplementedError(
                    "Only varlen and flex attn masks are supported"
                )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        # pyrefly: ignore [not-callable]
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks, positions)

        # pyrefly: ignore [not-callable]
        h = self.norm(h) if self.norm else h
        if self.output is None:
            return h
        # Liger path: when labels provided AND head is LigerLMHead, fused
        # linear+CE runs inside the head's forward (so FSDP's pre-forward
        # hook gathers self.output.weight). Returns scalar loss.
        if labels is not None and isinstance(self.output, LigerLMHead):
            # pyrefly: ignore [not-callable]
            return self.output(h, labels)
        # pyrefly: ignore [not-callable]
        return self.output(h)
