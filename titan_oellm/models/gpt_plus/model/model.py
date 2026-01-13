# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.protocols.model import ModelProtocol, AttentionMasksType
from torchtitan.protocols.train_spec import BaseTokenizer
from torchtitan.models.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    create_attention_mask,
)
from torch.nn.attention.flex_attention import and_masks, BlockMask

from titan_oellm.components.flash_attention import FlashAttentionWrapper

from .args import RoPEScalingArgs, TransformerModelArgs


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    scaling_args: RoPEScalingArgs = RoPEScalingArgs(),
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.
        scaling_args (RoPEScalingArgs | None): RoPE scaling arguments. Defaults to None.
            scaling_factor (float): RoPE scaling multiplier; larger values
                stretch positions to support longer contexts. Defaults to 8.0.
            low_freq_factor (float): Extra scaling applied to the low-frequency
                (long-wavelength) RoPE bands. Defaults to 1.0.
            high_freq_factor (float): Extra scaling applied to the high-frequency
                (short-wavelength) RoPE bands. Defaults to 4.0.
            original_max_position_embeddings (int): Maximum position embeddings
                for original model. Defaults to 8192.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # apply rope scaling
    scaling_factor = scaling_args.scaling_factor
    low_freq_factor = scaling_args.low_freq_factor
    high_freq_factor = scaling_args.high_freq_factor
    original_max_position_embeddings = scaling_args.original_max_position_embeddings
    wavelen = 2 * math.pi / freqs
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by scaling factor
    freqs = torch.where(wavelen > low_freq_wavelen, freqs / scaling_factor, freqs)
    # wavelen in between: linear interpolation of the scaled freqs and the original freqs
    smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_freqs = (
        1 - smooth_factor
    ) * freqs / scaling_factor + smooth_factor * freqs
    is_medium_freqs = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    freqs = torch.where(is_medium_freqs, smoothed_freqs, freqs)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
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


class QKNormPlus(nn.Module):
    """
    Normalization layer for Query or Key tensors with configurable scaling.

    Applies L2 normalization followed by learnable scaling to match RMSNorm behavior.
    The sqrt(head_dim) scaling ensures output L2 norm matches RMSNorm's natural output.

    Base modes (static scaling):
    - "scalar": single learnable scalar
    - "head_dim": learnable vector of size head_dim
    - "n_heads": learnable vector of size n_heads
    - "matrix": learnable matrix of size (n_heads, head_dim)

    Position-dependent modes (scale = alpha + beta * position):
    - "scalar_pos": scalar alpha and beta
    - "head_dim_pos": head_dim-sized alpha and beta vectors
    - "n_heads_pos": n_heads-sized alpha and beta vectors
    - "matrix_pos": (n_heads, head_dim) alpha and beta matrices

    Args:
        model_args: Model configuration with n_heads and dim
        eps: Epsilon for L2 normalization (default: 1e-8)
    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.head_dim = model_args.dim // model_args.n_heads
        self.eps = model_args.norm_eps

        gamma_shape = (self.n_heads, self.head_dim)
        self.gamma_q = nn.Parameter(torch.empty(gamma_shape))
        self.gamma_k = nn.Parameter(torch.empty(gamma_shape))

        self.scale_alpha = nn.Parameter(torch.empty(self.n_heads))
        self.scale_beta = nn.Parameter(torch.empty(self.n_heads))


    def init_weights(self):

        nn.init.ones_(self.gamma_q)
        nn.init.ones_(self.gamma_k)

        nn.init.constant_(self.scale_alpha, 4.0)
        nn.init.constant_(self.scale_beta, 1.0)


    def _compute_scale(self, xq: torch.Tensor) -> torch.Tensor:
        """
        Compute the scale factor, optionally position-dependent.

        Args:
            x: Input tensor of shape (batch, seqlen, n_heads, head_dim)

        Returns:
            Scale tensor with appropriate shape for broadcasting
        """

        batch, seqlen, n_heads, head_dim = xq.shape
        device = xq.device
        dtype = xq.dtype

        pos = (torch.arange(seqlen, device=device, dtype=dtype) +2).log()

        alpha = self.scale_alpha
        beta = self.scale_beta 
        
        scale = alpha + beta * pos.unsqueeze(-1)
        scale = scale.unsqueeze(0).unsqueeze(-1)
        scale = scale * math.sqrt(self.head_dim) # TODO this is because we do not remove the scale in the attention in this model

        return scale

    def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply L2 normalization to Q and K, then scale Q.

        Args:
            xq: Query tensor of shape (batch, seqlen, n_heads, head_dim)
            xk: Key tensor of shape (batch, seqlen, n_heads, head_dim)

        Returns:
            Tuple of (normalized_scaled_q, normalized_k)
        """

        input_dtype = xq.dtype

        xq_norm = F.normalize(xq, p=2, dim=-1, eps=self.eps)
        xk_norm = F.normalize(xk, p=2, dim=-1, eps=self.eps)

        xq_norm = xq_norm * self.gamma_q.unsqueeze(0).unsqueeze(0)
        xk_norm = xk_norm * self.gamma_k.unsqueeze(0).unsqueeze(0)

        scale = self._compute_scale(xq)

        xq_scaled = xq_norm * scale

        # Cast back to input dtype for AMP compatibility
        return xq_scaled.to(input_dtype), xk_norm.to(input_dtype)

        








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

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)


        self.qk_norm_type = model_args.qk_norm_type

        if self.qk_norm_type == 'QKNormPlus':
            self.qk_norm = QKNormPlus(model_args)

        elif self.qk_norm_type == "RMSNorm":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=model_args.norm_eps, elementwise_affine=True)
        elif self.qk_norm_type == "None":
            pass
        else:
            raise UserWarning(f"Unknown QK norm type {model_args.qk_norm_type}")

        # Attention backend selection: FlashAttention > FlexAttention > SDPA
        self.use_flash_attn = model_args.use_flash_attn
        self.use_flex_attn = model_args.use_flex_attn
        if self.use_flash_attn:
            self.inner_attention = FlashAttentionWrapper()
        elif self.use_flex_attn:
            self.inner_attention = FlexAttentionWrapper()
        else:
            self.inner_attention = ScaledDotProductAttentionWrapper()

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        if self.qk_norm_type == 'QKNormPlus':
            self.qk_norm.init_weights()
        elif self.qk_norm_type == "RMSNorm":
            for norm in (self.q_norm, self.k_norm):
                norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            attention_masks: Optional attention masks (BlockMask for FlexAttention).
            cu_seqlens: Cumulative sequence lengths for Flash Attention document masking.
            max_seqlen: Maximum sequence length for Flash Attention.

        Returns:
            torch.Tensor: Output tensor after attention.

        """

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.qk_norm_type == 'QKNormPlus':
            xq, xk = self.qk_norm(xq, xk)
        elif self.qk_norm_type == "RMSNorm":
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        if self.use_flash_attn:
            # Flash Attention with optional document masking via cu_seqlens
            output = self.inner_attention(
                xq, xk, xv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                causal=True,
            )
        elif self.use_flex_attn:
            assert isinstance(attention_masks, BlockMask), attention_masks
            output = self.inner_attention(xq, xk, xv, block_mask=attention_masks)
        else:
            assert attention_masks is None
            output = self.inner_attention(xq, xk, xv)

        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    Unified FeedForward module supporting 2-4 layers with SwiGLU or SiLU activation.

    For SwiGLU: Uses gated activation F.silu(w1(x)) * w3(x)
    For SiLU: Uses simple activation F.silu(w1(x)) to keep parameter count matched

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.
    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()

        num_layers = model_args.mlp_layers
        activation = model_args.mlp_activation

        if num_layers not in [2, 3, 4]:
            raise ValueError(f"mlp_layers must be 2, 3, or 4, got {num_layers}")
        if activation not in ["swiglu", "silu"]:
            raise ValueError(f"mlp_activation must be 'swiglu' or 'silu', got {activation}")

        self.num_layers = num_layers
        self.activation = activation

        dim = model_args.dim
        ffn_dim_multiplier = model_args.ffn_dim_multiplier

        # Base hidden dim from 2-layer formula
        hidden_dim = int(ffn_dim_multiplier * dim)
        hidden_dim = model_args.multiple_of * ((hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of)

        # Calculate hidden dimension based on layer count to match parameter count
        if num_layers == 2:
            if activation == "swiglu":
                h = hidden_dim
            else:  # silu
                # SiLU needs 1.5x hidden_dim to match SwiGLU params
                # SwiGLU: 3*dim*h, SiLU: 2*dim*h_silu
                # To match: h_silu = 1.5 * h
                h = int(1.5 * hidden_dim)
                # Adaptive rounding based on model size
                if dim <= 2048:
                    multiple_of = 64
                else:
                    multiple_of = 128
                h = multiple_of * ((h + multiple_of - 1) // multiple_of)
        else:
            # For SwiGLU: solve for h to match 3*dim*hidden_dim parameters
            # For SiLU: solve for h to match 3*dim*hidden_dim parameters (but different structure)
            if activation == "swiglu":
                # n-layer params: 3*dim*h + 2*(n-2)*h²
                # Solving: 2*(n-2)*h² + 3*dim*h - 3*dim*hidden_dim = 0
                if num_layers == 3:
                    h_exact = (-3*dim + math.sqrt(9*dim**2 + 24*dim*hidden_dim)) / 4
                else:  # num_layers == 4
                    h_exact = (-3*dim + math.sqrt(9*dim**2 + 48*dim*hidden_dim)) / 8
            else:  # silu
                # For SiLU without gating: (n-1)*dim*h + (n-2)*h²
                # Solving to match 3*dim*hidden_dim
                if num_layers == 3:
                    # 2*dim*h + h² = 3*dim*hidden_dim
                    # h² + 2*dim*h - 3*dim*hidden_dim = 0
                    h_exact = (-2*dim + math.sqrt(4*dim**2 + 12*dim*hidden_dim)) / 2
                else:  # num_layers == 4
                    # 3*dim*h + 2*h² = 3*dim*hidden_dim
                    # 2*h² + 3*dim*h - 3*dim*hidden_dim = 0
                    h_exact = (-3*dim + math.sqrt(9*dim**2 + 24*dim*hidden_dim)) / 4

            # Adaptive rounding based on model size
            if dim <= 2048:
                multiple_of = 64
            else:
                multiple_of = 128

            h = int(h_exact)
            h = multiple_of * ((h + multiple_of - 1) // multiple_of)

        # Build layers
        self.input_layers = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()

        if activation == "swiglu":
            # First layer: dim -> h (with gate)
            self.input_layers.append(nn.Linear(dim, h, bias=False))  # w1
            self.input_layers.append(nn.Linear(dim, h, bias=False))  # w3 (gate)

            # Hidden layers: h -> h (with gate)
            for _ in range(num_layers - 2):
                self.hidden_layers.append(nn.Linear(h, h, bias=False))
                self.hidden_layers.append(nn.Linear(h, h, bias=False))  # gate
        else:  # silu
            # First layer: dim -> h (no gate)
            self.input_layers.append(nn.Linear(dim, h, bias=False))

            # Hidden layers: h -> h (no gate)
            for _ in range(num_layers - 2):
                self.hidden_layers.append(nn.Linear(h, h, bias=False))

        # Output layer: h -> dim
        self.output_layer = nn.Linear(h, dim, bias=False)

    def forward(self, x):
        if self.activation == "swiglu":
            # First layer with SwiGLU
            h = F.silu(self.input_layers[0](x)) * self.input_layers[1](x)

            # Hidden layers with SwiGLU
            for i in range(0, len(self.hidden_layers), 2):
                h = F.silu(self.hidden_layers[i](h)) * self.hidden_layers[i+1](h)
        else:  # silu
            # First layer with SiLU
            h = F.silu(self.input_layers[0](x))

            # Hidden layers with SiLU
            for layer in self.hidden_layers:
                h = F.silu(layer(h))

        # Output layer
        return self.output_layer(h)

    def init_weights(self, init_std: float):
        # Initialize all input layers
        for layer in self.input_layers:
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)

        # Initialize all hidden layers
        for layer in self.hidden_layers:
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)

        # Initialize output layer
        nn.init.trunc_normal_(self.output_layer.weight, mean=0.0, std=init_std)


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

    def __init__(self, layer_id: int, model_args: TransformerModelArgs):
        super().__init__()

        self.n_heads = model_args.n_heads
        self.dim = model_args.dim


        self.attention = Attention(model_args)

        self.feed_forward = FeedForward(model_args)
        
        self.attn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks: Optional attention masks (BlockMask for FlexAttention).
            cu_seqlens: Cumulative sequence lengths for Flash Attention document masking.
            max_seqlen: Maximum sequence length for Flash Attention.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attn_norm(x), freqs_cis, attention_masks, cu_seqlens, max_seqlen)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attn_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

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

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # Tie embedding and output weights if specified
        if model_args.tie_embedding:
            self.output.weight = self.tok_embeddings.weight

        self.init_weights()

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
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        # Only initialize output weights if they're not tied to embeddings
        if self.output is not None and not self.model_args.tie_embedding:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
            self.model_args.rope_scaling_args,
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        """
        Create attention masks for FlexAttention.

        Args:
            input_batch: Input token tensor with shape [batch, seqlen].
            tokenizer: Tokenizer with eos_id for document boundary detection.
            extra_inputs: Optional extra inputs (unused).

        Returns:
            BlockMask for FlexAttention with causal and optional document masking.
        """
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
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks: Optional attention masks (BlockMask for FlexAttention).
            cu_seqlens: Cumulative sequence lengths for Flash Attention document masking.
            max_seqlen: Maximum sequence length for Flash Attention.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, cu_seqlens, max_seqlen)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
