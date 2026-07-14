"""
Static FP8 weight quantization for inference.

Pre-quantizes nn.Linear weights to FP8 E4M3 with per-channel (row-wise) scales,
eliminating the dynamic amax computation that Float8Linear performs on every
forward pass.  Weights are stored in float8_e4m3fn (half the memory of bf16).

Usage:
    model.eval()
    quantize_weights_fp8(model)          # mutates model in-place
    output = model(input)                # inference with FP8 weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

E4M3_MAX: float = 448.0
E4M3_MIN_SCALE: float = 1e-12


class StaticFP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear with pre-quantized FP8 weights.

    Weights are stored in float8_e4m3fn.  Forward uses torch._scaled_mm for
    native FP8 tensor-core matmuls (SM89+, e.g. H100).

    Weight scale is per-tensor (scalar) for _scaled_mm compatibility.
    Input is dynamically quantized to FP8 per-tensor in the forward pass.
    """

    def __init__(
        self,
        weight_fp8: Tensor,
        weight_scale: Tensor,
        bias: Tensor | None = None,
    ):
        super().__init__()
        # Store weight in original layout [out, in] (row-major, contiguous).
        # In forward we call .t() to get a col-major (in, out) view — this is
        # what torch._scaled_mm requires for mat2.
        self.register_buffer("weight_fp8", weight_fp8.contiguous())  # (out, in)
        self.register_buffer("weight_scale", weight_scale.squeeze())   # scalar
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.out_features = weight_fp8.shape[0]
        self.in_features = weight_fp8.shape[1]

    @property
    def weight(self) -> Tensor:
        """Dequantized weight (for compatibility with code that reads .weight)."""
        return self.weight_fp8.float() * self.weight_scale

    def forward(self, x: Tensor) -> Tensor:
        # Flatten to 2D for _scaled_mm: (*, in) -> (M, in)
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # Dynamic per-tensor quantize input to FP8
        x_amax = x_2d.detach().float().abs().amax()
        x_scale = (x_amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)
        x_fp8 = (x_2d.float() / x_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

        # FP8 matmul on tensor cores: (M, in) @ (in, out) -> (M, out)
        # weight_fp8 is (out, in) row-major; .t() gives (in, out) col-major view
        out = torch._scaled_mm(
            x_fp8, self.weight_fp8.t(),
            scale_a=x_scale, scale_b=self.weight_scale,
            out_dtype=x.dtype,
        )

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype=float8_e4m3fn, "
                f"matmul=torch._scaled_mm")


def _quantize_weight_to_fp8(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a bf16/fp32 weight tensor to FP8 E4M3 with per-tensor scale.

    Per-tensor scale is required by torch._scaled_mm.

    Returns:
        (weight_fp8, weight_scale) where weight ≈ weight_fp8 * weight_scale
    """
    # Per-tensor amax → single scalar scale
    amax = weight.float().abs().amax()  # scalar
    scale = (amax / E4M3_MAX).clamp(min=E4M3_MIN_SCALE)

    weight_fp8 = (weight.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return weight_fp8, scale


def quantize_weights_fp8(
    model: nn.Module,
    min_dim: int = 16,
    skip_fqns: list[str] | None = None,
) -> int:
    """Replace eligible nn.Linear modules with StaticFP8Linear (in-place).

    Args:
        model: The model to quantize (should be in eval mode).
        min_dim: Only convert layers where both dims are >= min_dim and
                 divisible by 16 (FP8 tensor core requirement).
        skip_fqns: List of substrings; if any matches the fully-qualified
                   module name, the layer is skipped.

    Returns:
        Number of layers converted.
    """
    if skip_fqns is None:
        skip_fqns = []

    n_converted = 0
    # Collect (parent, attr_name, module, fqn) tuples first to avoid
    # modifying the module tree while iterating.
    replacements: list[tuple[nn.Module, str, nn.Linear, str]] = []

    for fqn, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        out_f, in_f = mod.weight.shape
        if out_f % 16 != 0 or in_f % 16 != 0:
            continue
        if out_f < min_dim or in_f < min_dim:
            continue
        if any(s in fqn for s in skip_fqns):
            continue

        # Find parent module and attribute name
        parts = fqn.rsplit(".", 1)
        if len(parts) == 2:
            parent = dict(model.named_modules())[parts[0]]
            attr = parts[1]
        else:
            parent = model
            attr = fqn

        replacements.append((parent, attr, mod, fqn))

    for parent, attr, linear, fqn in replacements:
        weight_fp8, weight_scale = _quantize_weight_to_fp8(linear.weight.data)
        bias = linear.bias.data if linear.bias is not None else None

        fp8_linear = StaticFP8Linear(weight_fp8, weight_scale, bias)
        fp8_linear.to(linear.weight.device)
        setattr(parent, attr, fp8_linear)
        n_converted += 1

    return n_converted
