"""
Bounded Spherical Adam: Adaptive optimisation for weight matrices
constrained to the unit L2 ball (||w|| <= 1) or unit sphere (||w|| = 1).

Supports four constraint modes:
- "bounded": Rows are bounded by max_norm (‖w_i‖ ≤ max_norm).
  Uses predictive activation: projects rows where the predicted Adam step
  would push the norm past max_norm (based on current momentum state).
- "normalized": Rows are normalized to unit norm (‖w_i‖ = 1).
  All rows are always projected onto the tangent space.
- "partial_orthogonal": Spectral norm + Newton-Schulz + row normalize (‖w_i‖ = 1).
  Decorrelates rows via NS iterations for more stable output norms.
- "partial_orthogonal_bounded": Same as partial_orthogonal but clamps ‖w_i‖ ≤ 1.0.

Designed for architectures like anGPT that bound (rather than normalise)
weight matrices.  The key insight: the unit ball is a manifold with
boundary, requiring conditional projection — only rows at/near the
boundary need tangent-space correction; interior rows get standard Adam.

Improvements over naive clipped Adam:
  1. Tangent-space projection of gradient BEFORE it enters momentum
  2. Momentum re-projection when constraint activates (clears phantom radial momentum)
  3. Optional element-wise v_t correction at activation
  4. Soft blending option to avoid hard transition discontinuity

Compatible with torchtitan's OptimizersContainer pattern.
"""

from __future__ import annotations

from typing import Optional

import itertools
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.ft import FTManager
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims


__all__ = [
    "BoundedSphericalAdamOptimizerContainer",
    "configure_bounded_spherical_adam",
    "make_build_bounded_spherical_adam_optimizers",
    "_bounded_spherical_adam_config",
]


# ======================================================================
#  torchtitan integration
# ======================================================================

_bounded_spherical_adam_config: dict = {
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "mode": "bounded",
    "max_norm": 1.0,
    "project_gradients": True,
    "correct_v": False,
    "soft_blend": False,
    "rotation_lr": None,
    "fallback_lr": None,
    "embedding_lr": None,
    "n_iter_spectral": 3,
    "n_iter_ns": 5,
    "n_iter": 1.0,
    "ffn_down_left_ns": False,
    "ns_alpha": 1.0,
    "ns_mode": "full",
    "kappa_target": 4.0,
    "lambda_max": 0.05,
    "kappa_ema_beta": 0.99,
    "ns_schedule_steps": 4000,
    "out_norm_dim_0": False,
}


def configure_bounded_spherical_adam(
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    mode: str = "bounded",
    max_norm: float = 1.0,
    project_gradients: bool = True,
    correct_v: bool = False,
    soft_blend: bool = False,
    rotation_lr: Optional[float] = None,
    fallback_lr: Optional[float] = None,
    embedding_lr: Optional[float] = None,
    n_iter_spectral: int = 3,
    n_iter_ns: int = 5,
    n_iter: float = 1.0,
    ffn_down_left_ns: bool = False,
    ns_alpha: float = 1.0,
    ns_mode: str = "full",
    kappa_target: float = 4.0,
    lambda_max: float = 0.05,
    kappa_ema_beta: float = 0.99,
    ns_schedule_steps: int = 4000,
    out_norm_dim_0: bool = False,
) -> None:
    """Configure BoundedSphericalAdam settings used by make_build_bounded_spherical_adam_optimizers().

    Args:
        betas: Adam beta coefficients.
        eps: Adam epsilon for numerical stability.
        weight_decay: Decoupled weight decay (AdamW style, non-bounded params only).
        mode: Constraint mode - "bounded", "normalized", "partial_orthogonal",
            or "partial_orthogonal_bounded".
        max_norm: Maximum row norm (for bounded mode).
        project_gradients: Project gradients/momentum to tangent space (True).
            Set False for pure Adam + constraint enforcement (equivalent to AdamW + GPTNormalizer).
        correct_v: Correct second-moment buffer at constraint activation.
        soft_blend: Use smooth blending in the delta-band.
        rotation_lr: Separate learning rate for constrained matrices (None = use optimizer.lr).
        fallback_lr: Learning rate for non-matrix parameters.
        embedding_lr: Learning rate for embedding parameters.
        n_iter_spectral: Power iteration steps for spectral norm (partial_orthogonal modes).
        n_iter_ns: Newton-Schulz iterations (partial_orthogonal modes).
        n_iter: Partial_orthogonal passes per step. 2.0=2 passes every step,
            1.0=1 pass every step, 0.5=1 pass every 2nd step, 0.2=every 5th.
        ffn_down_left_ns: Use left-sided Newton-Schulz for FFN down-projection (d_out < d_in).
            Requires model to implement get_param_ortho_metadata().
        ns_alpha: NS blending factor (used by "full" and "lerp" modes). 1.0=full NS, 0.0=skip.
        ns_mode: NS blending mode. "full"=standard NS with optional ns_alpha blend,
            "lerp"=explicit linear interpolation using ns_alpha,
            "adaptive"=augmented Lagrangian based on condition number.
        kappa_target: Target condition number for adaptive NS mode (default: 4.0).
        lambda_max: Maximum NS interpolation weight for adaptive mode (default: 0.05).
        kappa_ema_beta: EMA smoothing for per-parameter kappa tracking (default: 0.99).
        ns_schedule_steps: Steps to transition from full NS to adaptive (schedule mode, default: 4000).
        out_norm_dim_0: Normalize output projections (wo, w2) along dim=0 (column-wise).
            Default False = dim=-1 (row-wise).
    """
    _bounded_spherical_adam_config["betas"] = betas
    _bounded_spherical_adam_config["eps"] = eps
    _bounded_spherical_adam_config["weight_decay"] = weight_decay
    _bounded_spherical_adam_config["mode"] = mode
    _bounded_spherical_adam_config["max_norm"] = max_norm
    _bounded_spherical_adam_config["project_gradients"] = project_gradients
    _bounded_spherical_adam_config["correct_v"] = correct_v
    _bounded_spherical_adam_config["soft_blend"] = soft_blend
    _bounded_spherical_adam_config["rotation_lr"] = rotation_lr
    _bounded_spherical_adam_config["fallback_lr"] = fallback_lr
    _bounded_spherical_adam_config["embedding_lr"] = embedding_lr
    _bounded_spherical_adam_config["n_iter_spectral"] = n_iter_spectral
    _bounded_spherical_adam_config["n_iter_ns"] = n_iter_ns
    _bounded_spherical_adam_config["n_iter"] = n_iter
    _bounded_spherical_adam_config["ffn_down_left_ns"] = ffn_down_left_ns
    _bounded_spherical_adam_config["ns_alpha"] = ns_alpha
    _bounded_spherical_adam_config["ns_mode"] = ns_mode
    _bounded_spherical_adam_config["kappa_target"] = kappa_target
    _bounded_spherical_adam_config["lambda_max"] = lambda_max
    _bounded_spherical_adam_config["kappa_ema_beta"] = kappa_ema_beta
    _bounded_spherical_adam_config["ns_schedule_steps"] = ns_schedule_steps
    _bounded_spherical_adam_config["out_norm_dim_0"] = out_norm_dim_0


# ---------------------------------------------------------------------------
# torchtitan integration: build_optimizers factory
# ---------------------------------------------------------------------------


def make_build_bounded_spherical_adam_optimizers():
    """
    Factory function that creates a build_optimizers_fn for BoundedSphericalAdam.

    Uses module-level configuration set via configure_bounded_spherical_adam().
    Per-parameter constraint types are read from the model's get_param_constraints()
    method (with name-based fallback for models that don't implement it).

    Optimizer groups (all handled by a single fused AdamW per model part):
      - 2D hidden matrices: fused AdamW + pre-step projection + post-step constraint
      - Embeddings (tok_embeddings, output): fused AdamW + post-step constraint
      - 1D/scalars (biases, norms, gates): fused AdamW (no constraint)
    """

    def build_bounded_spherical_adam_optimizers_fn(
        model_parts: list[torch.nn.Module],
        optimizer_config: OptimizerConfig,
        parallel_dims: ParallelDims,
        ft_manager: FTManager,
    ):
        if optimizer_config.early_step_in_backward:
            raise NotImplementedError(
                "BoundedSphericalAdam does not support early_step_in_backward."
            )

        fallback_lr = _bounded_spherical_adam_config.get("fallback_lr")
        if fallback_lr is None or fallback_lr <= 0:
            fallback_lr = optimizer_config.lr

        embedding_lr = _bounded_spherical_adam_config.get("embedding_lr")
        if embedding_lr is None or embedding_lr <= 0:
            embedding_lr = fallback_lr

        return BoundedSphericalAdamOptimizerContainer(
            model_parts=model_parts,
            constrained_lr=optimizer_config.lr,
            embedding_lr=embedding_lr,
            scalar_lr=fallback_lr,
            bounded_spherical_config=_bounded_spherical_adam_config,
        )

    return build_bounded_spherical_adam_optimizers_fn


class BoundedSphericalAdamOptimizerContainer(OptimizersContainer):
    """
    Fused AdamW container with pre-step tangent-space projection and post-step
    constraint enforcement for bounded/normalized weight matrices.

    Uses a single torch.optim.AdamW(fused=True) per model part for ALL params.
    This ensures:
      - Bit-exact Adam step matching the AdamW + GPTNormalizer baseline
      - Correct LR scheduler propagation (no wrapper indirection)
      - Standard PyTorch state dict format for checkpointing

    Per-parameter constraint types are read from the model's get_param_constraints()
    method. Each param gets one of:
      - "normalized" — always ‖w‖=1 (e.g., embeddings, output head)
      - "bounded"    — always ‖w‖≤max_norm
      - "none"       — no constraint (1D scalars)
    Params classified as "default" by the model are resolved to the global mode.

    Step order:
      1. Pre-step: project gradients and momentum for constrained params (if project_gradients)
      2. Fused AdamW step for all params
      3. Post-step: enforce per-param constraint (normalized or bounded)
    """

    def __init__(
        self,
        model_parts: list[torch.nn.Module],
        constrained_lr: float,
        embedding_lr: float,
        scalar_lr: float,
        bounded_spherical_config: dict,
    ):
        from torchtitan.tools.logging import logger

        all_params = []
        self.optimizers: list[torch.optim.AdamW] = []
        self.model_parts = model_parts

        # Pre-grouped param lists (no per-param branching in hot path)
        self._normalized_params_per_optimizer: list[list[torch.nn.Parameter]] = []
        self._bounded_params_per_optimizer: list[list[torch.nn.Parameter]] = []
        self._partial_ortho_params_per_optimizer: list[list[torch.nn.Parameter]] = []
        self._partial_ortho_bounded_params_per_optimizer: list[list[torch.nn.Parameter]] = []

        # Store projection/constraint config
        self._mode = bounded_spherical_config["mode"]
        self._max_norm = bounded_spherical_config["max_norm"]
        self._eps = bounded_spherical_config["eps"]
        self._project_gradients = bounded_spherical_config["project_gradients"]
        self._correct_v = bounded_spherical_config["correct_v"]
        self._soft_blend = bounded_spherical_config["soft_blend"]

        # Partial orthogonal config
        self._n_iter_spectral = bounded_spherical_config["n_iter_spectral"]
        self._n_iter_ns = bounded_spherical_config["n_iter_ns"]
        self._n_iter = bounded_spherical_config["n_iter"]
        self._ffn_down_left_ns = bounded_spherical_config.get("ffn_down_left_ns", False)
        self._ns_alpha = bounded_spherical_config.get("ns_alpha", 1.0)
        self._ns_mode = bounded_spherical_config.get("ns_mode", "full")
        self._kappa_target = bounded_spherical_config.get("kappa_target", 4.0)
        self._lambda_max = bounded_spherical_config.get("lambda_max", 0.05)
        self._kappa_ema_beta = bounded_spherical_config.get("kappa_ema_beta", 0.99)
        self._ns_schedule_steps = bounded_spherical_config.get("ns_schedule_steps", 4000)
        self._step_count = 0
        self._power_iter_state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._eye_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        # Per-param metadata for FFN down left-NS: id(p) → {"po_type": str}
        self._po_metadata: dict[int, dict] = {}
        # Per-param names for logging: id(p) → name
        self._param_names: dict[int, str] = {}
        # Per-param EMA of kappa for adaptive NS mode: id(p) → float
        self._kappa_ema: dict[int, float] = {}
        # Per-param adaptive stats for logging: id(p) → {ns_lambda, kappa_ema, violation}
        self._adaptive_stats: dict[int, dict[str, float]] = {}
        # Output projection handling (wo, w2)
        self._out_norm_dim_0 = bounded_spherical_config.get("out_norm_dim_0", False)
        self._normalized_dim0_params_per_optimizer: list[list[torch.nn.Parameter]] = []
        self._bounded_dim0_params_per_optimizer: list[list[torch.nn.Parameter]] = []

        betas = bounded_spherical_config["betas"]
        eps = bounded_spherical_config["eps"]

        total_constrained = 0
        total_unconstrained = 0
        constraint_counts: dict[str, int] = {
            "normalized": 0, "bounded": 0,
            "partial_orthogonal": 0, "partial_orthogonal_bounded": 0,
        }

        for model in model_parts:
            # Get per-param constraint classification from model
            constraint_map = self._get_constraint_map(model)
            ortho_metadata = self._get_ortho_metadata(model) if self._ffn_down_left_ns else {}
            mup_scales = self._get_mup_lr_scales(model)

            matrix_group = []
            down_group = []  # FFN down projections (µP: extra 1/√R lr factor)
            embed_group = []
            scalar_group = []
            normalized_params: list[torch.nn.Parameter] = []
            bounded_params: list[torch.nn.Parameter] = []
            partial_ortho_params: list[torch.nn.Parameter] = []
            partial_ortho_bounded_params: list[torch.nn.Parameter] = []
            normalized_dim0: list[torch.nn.Parameter] = []
            bounded_dim0: list[torch.nn.Parameter] = []

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                all_params.append(p)

                # LR grouping (name-based)
                if p.ndim >= 2:
                    if "tok_embeddings" in name or "output" in name:
                        embed_group.append(p)
                    elif mup_scales and mup_scales.get(name, 1.0) < 1.0:
                        down_group.append(p)
                    else:
                        matrix_group.append(p)
                else:
                    scalar_group.append(p)

                # Constraint classification
                constraint = constraint_map.get(name, None)
                if constraint is None:
                    # Fallback for models without get_param_constraints
                    if p.ndim < 2:
                        constraint = "none"
                    elif "tok_embeddings" in name or "output" in name:
                        constraint = "normalized"
                    else:
                        constraint = "default"

                # Resolve "default" to global mode
                if constraint == "default":
                    constraint = self._mode

                # Identify output projections (wo, w2) for special handling
                is_output_proj = p.ndim >= 2 and (".wo." in name or ".w2." in name)

                if constraint == "normalized":
                    if is_output_proj and self._out_norm_dim_0:
                        normalized_dim0.append(p)
                    else:
                        normalized_params.append(p)
                    total_constrained += 1
                    constraint_counts["normalized"] += 1
                elif constraint == "bounded":
                    if is_output_proj and self._out_norm_dim_0:
                        bounded_dim0.append(p)
                    else:
                        bounded_params.append(p)
                    total_constrained += 1
                    constraint_counts["bounded"] += 1
                elif constraint == "partial_orthogonal":
                    partial_ortho_params.append(p)
                    total_constrained += 1
                    constraint_counts["partial_orthogonal"] += 1
                elif constraint == "partial_orthogonal_bounded":
                    partial_ortho_bounded_params.append(p)
                    total_constrained += 1
                    constraint_counts["partial_orthogonal_bounded"] += 1
                else:
                    total_unconstrained += 1

                # Store param name for spectral stats logging
                if constraint in ("partial_orthogonal", "partial_orthogonal_bounded"):
                    self._param_names[id(p)] = name

                # Store structured PO metadata for this param
                if self._ffn_down_left_ns and name in ortho_metadata:
                    self._po_metadata[id(p)] = ortho_metadata[name]

            self._normalized_dim0_params_per_optimizer.append(normalized_dim0)
            self._bounded_dim0_params_per_optimizer.append(bounded_dim0)
            self._normalized_params_per_optimizer.append(normalized_params)
            self._bounded_params_per_optimizer.append(bounded_params)
            self._partial_ortho_params_per_optimizer.append(partial_ortho_params)
            self._partial_ortho_bounded_params_per_optimizer.append(partial_ortho_bounded_params)

            # Build param groups for a single fused AdamW
            # µP: down projections get a reduced lr (constrained_lr * scale)
            down_lr = constrained_lr
            if down_group and mup_scales:
                # All down-group params share the same scale (1/√R)
                down_scale = next(iter(set(
                    mup_scales.get(n, 1.0) for n, p in model.named_parameters()
                    if p.requires_grad and id(p) in {id(dp) for dp in down_group}
                )))
                down_lr = constrained_lr * down_scale

            param_groups = []
            if matrix_group:
                param_groups.append(dict(params=matrix_group, lr=constrained_lr))
            if down_group:
                param_groups.append(dict(params=down_group, lr=down_lr))
            if embed_group:
                param_groups.append(dict(params=embed_group, lr=embedding_lr))
            if scalar_group:
                param_groups.append(dict(params=scalar_group, lr=scalar_lr))

            # Single fused AdamW for ALL params in this model part
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=constrained_lr,
                betas=betas,
                eps=eps,
                weight_decay=0.0,
                fused=True,
            )
            self.optimizers.append(optimizer)

        self._validate_length(len(self.model_parts))
        self._post_init(all_params, {"lr": constrained_lr})

        # Force-initialize optimizer state (exp_avg, exp_avg_sq) so that
        # DCP checkpoint loading can match saved state against the structure
        # returned by state_dict().  Without this, state_dict() returns an
        # empty dict before the first step(), and DCP silently skips the
        # optimizer state from the checkpoint.
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }

        # Build shape groups for batched constraint/projection operations
        self._bounded_shape_groups = self._build_shape_groups(
            self._bounded_params_per_optimizer
        )

        mode_descs = {
            "bounded": f"‖w‖≤{self._max_norm}",
            "normalized": "‖w‖=1",
            "partial_orthogonal": "spectral+NS+‖w‖=1",
            "partial_orthogonal_bounded": "spectral+NS+‖w‖≤1",
        }
        default_desc = mode_descs.get(self._mode, self._mode)
        po_info = ""
        if self._mode.startswith("partial_orthogonal"):
            po_info = (
                f"\n  PO config: n_iter_spectral={self._n_iter_spectral}, "
                f"n_iter_ns={self._n_iter_ns}, n_iter={self._n_iter}, "
                f"ffn_down_left_ns={self._ffn_down_left_ns}, "
                f"ns_mode={self._ns_mode}, ns_alpha={self._ns_alpha}"
            )
            if self._ns_mode in ("adaptive", "schedule"):
                po_info += (
                    f"\n  Adaptive NS: kappa_target={self._kappa_target}, "
                    f"lambda_max={self._lambda_max}, kappa_ema_beta={self._kappa_ema_beta}"
                )
            if self._ns_mode == "schedule":
                po_info += f"\n  Schedule: full NS → adaptive over {self._ns_schedule_steps} steps"
            if self._ffn_down_left_ns:
                n_ffn = sum(1 for m in self._po_metadata.values() if m["po_type"] == "ffn")
                po_info += f"\n  FFN down left-NS: {n_ffn} FFN params"
        logger.info(
            f"BoundedSphericalAdamOptimizerContainer created:\n"
            f"  Model parts: {len(model_parts)}\n"
            f"  Global mode: {self._mode} ({default_desc})\n"
            f"  Constrained params: {total_constrained} "
            f"(normalized={constraint_counts['normalized']}, bounded={constraint_counts['bounded']}, "
            f"partial_ortho={constraint_counts['partial_orthogonal']}, "
            f"partial_ortho_bounded={constraint_counts['partial_orthogonal_bounded']})\n"
            f"  Unconstrained params: {total_unconstrained}\n"
            f"  Output proj (wo/w2): out_norm_dim_0={self._out_norm_dim_0}\n"
            f"  Dim=0 params (column-wise, full-tensor ops): "
            f"{sum(len(ps) for ps in self._normalized_dim0_params_per_optimizer) + sum(len(ps) for ps in self._bounded_dim0_params_per_optimizer)}\n"
            f"  LR: matrices={constrained_lr}, down={down_lr}, embeddings={embedding_lr}, scalars={scalar_lr}\n"
            f"  Config: max_norm={self._max_norm}, "
            f"project_gradients={self._project_gradients}, "
            f"correct_v={self._correct_v}, "
            f"soft_blend={self._soft_blend}"
            f"{po_info}"
        )
        for opt_idx, shape_groups in enumerate(self._bounded_shape_groups):
            if shape_groups:
                group_info = ", ".join(
                    f"{s}: {len(ps)} params" for s, ps in shape_groups.items()
                )
                logger.info(f"  Bounded shape groups (part {opt_idx}): {group_info}")

    @staticmethod
    def _get_constraint_map(model: torch.nn.Module) -> dict[str, str]:
        """Get per-param constraint classification from the model."""
        m = model
        if hasattr(m, 'module'):
            m = m.module
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        if hasattr(m, 'get_param_constraints'):
            return m.get_param_constraints()
        return {}

    @staticmethod
    def _get_ortho_metadata(model: torch.nn.Module) -> dict[str, dict]:
        """Get per-param structural metadata for structured partial orthogonal."""
        m = model
        if hasattr(m, 'module'):
            m = m.module
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        if hasattr(m, 'get_param_ortho_metadata'):
            return m.get_param_ortho_metadata()
        return {}

    @staticmethod
    def _get_mup_lr_scales(model: torch.nn.Module) -> dict[str, float]:
        """Get per-param µP lr scale factors from the model (empty = disabled)."""
        m = model
        if hasattr(m, 'module'):
            m = m.module
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        if hasattr(m, 'get_mup_lr_scales'):
            return m.get_mup_lr_scales()
        return {}

    @staticmethod
    def _get_local(t: torch.Tensor) -> torch.Tensor:
        """Extract local tensor from DTensor, or return as-is for plain tensors.

        Needed because torch.stack() does not support DTensors. Per-row
        operations (dim=-1) are fully local with FSDP2 Shard(0), so working
        on the local shard is both safe and correct.
        """
        if hasattr(t, "_local_tensor"):
            return t._local_tensor
        return t

    def _build_shape_groups(
        self, params_per_optimizer: list[list[torch.nn.Parameter]],
    ) -> list[dict[tuple, list[torch.nn.Parameter]]]:
        """Group parameters by local shard shape for batched operations.

        Returns one dict per optimizer, mapping shape tuple -> param list.
        """
        result = []
        for params in params_per_optimizer:
            groups: dict[tuple, list[torch.nn.Parameter]] = {}
            for p in params:
                local_shape = tuple(self._get_local(p.data).shape)
                groups.setdefault(local_shape, []).append(p)
            result.append(groups)
        return result

    def step(self, *args, **kwargs):
        """Pre-step projection -> fused Adam step -> constraint enforcement."""
        # 1. Pre-step: project gradients and momentum for constrained params
        if self._project_gradients:
            self._pre_step_projections()

        # 2. Fused AdamW step for ALL params
        super().step(*args, **kwargs)

        # 3. Post-step: enforce per-param constraint
        self._step_count += 1
        self._enforce_constraints()

    @torch.no_grad()
    def _pre_step_projections(self):
        """Project gradients and momentum onto tangent space for constrained params.

        Modifies p.grad and optimizer.state[p]["exp_avg"] in-place before the
        fused AdamW kernel runs. Separate loops for normalized vs bounded params
        eliminate per-param branching.
        """
        for opt_idx, (opt, normalized_params, _, po_params, pob_params, norm_dim0, bound_dim0) in enumerate(zip(
            self.optimizers,
            self._normalized_params_per_optimizer,
            self._bounded_params_per_optimizer,
            self._partial_ortho_params_per_optimizer,
            self._partial_ortho_bounded_params_per_optimizer,
            self._normalized_dim0_params_per_optimizer,
            self._bounded_dim0_params_per_optimizer,
        )):
            # ---- Normalized + partial_orthogonal params: ALL rows projected ----
            for p in itertools.chain(normalized_params, po_params):
                if p.grad is None:
                    continue
                grad = p.grad
                w = p.data
                w_f = w.float()
                row_norms = w_f.norm(dim=-1, keepdim=True)
                w_hat = w_f / row_norms.clamp(min=1e-8)  # keep in f32
                del w_f

                state = opt.state.get(p, {})
                exp_avg = state.get("exp_avg", None)
                exp_avg_sq = state.get("exp_avg_sq", None)

                radial_dot_g = (grad.float() * w_hat).sum(dim=-1, keepdim=True)
                p.grad = (grad.float() - radial_dot_g * w_hat).to(grad.dtype)

                if exp_avg is not None:
                    radial_dot_m = (exp_avg.float() * w_hat).sum(dim=-1, keepdim=True)
                    exp_avg.sub_((radial_dot_m * w_hat).to(exp_avg.dtype))

                if self._correct_v and exp_avg_sq is not None:
                    exp_avg_sq.mul_((1.0 - w_hat * w_hat).to(exp_avg_sq.dtype))

            # ---- Bounded + partial_orthogonal_bounded params: batched by shape group ----
            # Also process pob_params (rare in bounded mode) via per-param fallback

            # Build param→lr mapping (LR may change with schedulers)
            param_lr: dict[int, float] = {}
            for pg in opt.param_groups:
                lr_val = pg['lr']
                for pp in pg['params']:
                    param_lr[id(pp)] = lr_val

            for p in pob_params:
                if p.grad is None:
                    continue
                self._project_single_bounded(p, opt, param_lr)

            _gl = self._get_local

            for shape_key, group in self._bounded_shape_groups[opt_idx].items():
                # Filter params with gradients
                active_params = [p for p in group if p.grad is not None]
                if not active_params:
                    continue

                if len(active_params) < 2:
                    # Single param — no stacking overhead
                    self._project_single_bounded(active_params[0], opt, param_lr)
                    continue

                # Stack local tensors into [N, R, C] batch
                g_locals = [_gl(p.grad) for p in active_params]
                W = torch.stack([_gl(p.data) for p in active_params])  # [N, R, C] bf16
                G = torch.stack(g_locals)   # [N, R, C] bf16

                # f32 norm computation (REQUIRED: bf16 gives ~20% error at dim=576)
                W_f = W.float()             # [N, R, C] f32
                row_norms = W_f.norm(dim=-1, keepdim=True)  # [N, R, 1] f32
                w_hat = W_f / row_norms.clamp(min=1e-8)  # [N, R, C] keep f32
                del W_f

                # ---- Predictive activation: predict whether Adam step violates boundary ----
                has_state = opt.state.get(active_params[0], {}).get("exp_avg") is not None
                lrs = torch.tensor(
                    [param_lr.get(id(p), opt.defaults['lr']) for p in active_params],
                    dtype=torch.float32, device=W.device,
                ).view(-1, 1, 1)  # [N, 1, 1]

                if has_state:
                    M = torch.stack([_gl(opt.state[p]["exp_avg"]) for p in active_params]).float()
                    V = torch.stack([_gl(opt.state[p]["exp_avg_sq"]) for p in active_params]).float()
                    beta1 = opt.defaults['betas'][0]
                    m_pred = beta1 * M + (1 - beta1) * G.float()
                    adam_dir = m_pred / (V.sqrt() + self._eps)  # [N, R, C]
                    step_radial = (adam_dir * w_hat).sum(dim=-1, keepdim=True)  # [N, R, 1]
                    # delta_r > 0 means step pushes outward (norm increases)
                    delta_r = -lrs * step_radial
                else:
                    # Step 0: use gradient magnitude as conservative proxy
                    radial_g = (G.float() * w_hat).sum(dim=-1, keepdim=True)
                    delta_r = lrs * radial_g.abs()

                predicted_norm = row_norms + delta_r.clamp(min=0)

                G_f = G.float()
                radial_dot_g = (G_f * w_hat).sum(dim=-1, keepdim=True)

                if self._soft_blend:
                    # Smooth blending based on predicted overshoot
                    overshoot = (predicted_norm - self._max_norm).clamp(min=0)
                    sigma = (overshoot / delta_r.abs().clamp(min=1e-8)).clamp(0.0, 1.0)  # f32

                    # Gradient projection (f32 dot products)
                    project_g = (radial_dot_g < 0).float()
                    G_new = (G_f - sigma * project_g * (radial_dot_g * w_hat)).to(G.dtype)
                    for i, gl in enumerate(g_locals):
                        gl.copy_(G_new[i])

                    # Momentum projection (if optimizer state exists)
                    if has_state:
                        m_locals = [_gl(opt.state[p]["exp_avg"]) for p in active_params]
                        M_f = torch.stack(m_locals).float()
                        radial_dot_m = (M_f * w_hat).sum(dim=-1, keepdim=True)
                        project_m = (radial_dot_m < 0).float()
                        M_delta = (sigma * project_m * (radial_dot_m * w_hat)).to(m_locals[0].dtype)
                        for i, ml in enumerate(m_locals):
                            ml.sub_(M_delta[i])

                        if self._correct_v:
                            v_locals = [_gl(opt.state[p]["exp_avg_sq"]) for p in active_params]
                            V_scale = (1.0 - sigma * (w_hat * w_hat)).to(W.dtype)  # [N, R, C]
                            for i, vl in enumerate(v_locals):
                                vl.mul_(V_scale[i])
                else:
                    # Hard: binary activation based on predicted violation
                    active_bool = (predicted_norm > self._max_norm)  # [N, R, 1] bool

                    project_g = active_bool.float() * (radial_dot_g < 0).float()
                    G_new = (G_f - project_g * (radial_dot_g * w_hat)).to(G.dtype)
                    for i, gl in enumerate(g_locals):
                        gl.copy_(G_new[i])

                    if has_state:
                        m_locals = [_gl(opt.state[p]["exp_avg"]) for p in active_params]
                        M_f = torch.stack(m_locals).float()
                        radial_dot_m = (M_f * w_hat).sum(dim=-1, keepdim=True)
                        project_m = active_bool.float() * (radial_dot_m < 0).float()
                        M_delta = (project_m * radial_dot_m * w_hat).to(m_locals[0].dtype)
                        for i, ml in enumerate(m_locals):
                            ml.sub_(M_delta[i])

                        if self._correct_v:
                            v_locals = [_gl(opt.state[p]["exp_avg_sq"]) for p in active_params]
                            V_scale = torch.where(
                                active_bool.expand_as(w_hat), 1.0 - w_hat * w_hat,
                                torch.ones_like(w_hat),
                            ).to(W.dtype)  # [N, R, C]
                            for i, vl in enumerate(v_locals):
                                vl.mul_(V_scale[i])

            # ---- dim=0 (column-wise) normalized params: full-tensor projection ----
            for p in norm_dim0:
                if p.grad is None:
                    continue
                full_w, shard_info = self._get_full_weight(p)
                w_f = full_w.float()
                col_norms = w_f.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, C]
                w_hat = w_f / col_norms  # [R_full, C] unit-norm columns

                grad_local = self._get_local(p.grad)
                if shard_info is not None:
                    _, offset, local_size = shard_info
                    w_hat_local = w_hat[offset:offset + local_size]
                else:
                    w_hat_local = w_hat

                # Column-wise radial dot product (needs all-reduce for FSDP)
                local_dot = (grad_local.float() * w_hat_local).sum(dim=0, keepdim=True)  # [1, C]
                fsdp_pg = self._get_fsdp_process_group(p)
                if fsdp_pg is not None:
                    dist.all_reduce(local_dot, group=fsdp_pg)

                # Project gradient: remove radial component per column
                self._get_local(p.grad).copy_(
                    (grad_local.float() - local_dot * w_hat_local).to(p.grad.dtype)
                )

                # Project momentum
                state = opt.state.get(p, {})
                exp_avg = state.get("exp_avg", None)
                if exp_avg is not None:
                    m_local = self._get_local(exp_avg)
                    local_dot_m = (m_local.float() * w_hat_local).sum(dim=0, keepdim=True)
                    if fsdp_pg is not None:
                        dist.all_reduce(local_dot_m, group=fsdp_pg)
                    self._get_local(exp_avg).copy_(
                        (m_local.float() - local_dot_m * w_hat_local).to(exp_avg.dtype)
                    )

                # correct_v
                if self._correct_v:
                    exp_avg_sq = state.get("exp_avg_sq", None)
                    if exp_avg_sq is not None:
                        self._get_local(exp_avg_sq).mul_(
                            (1.0 - w_hat_local * w_hat_local).to(exp_avg_sq.dtype)
                        )

            # ---- dim=0 (column-wise) bounded params: predictive activation per column ----
            for p in bound_dim0:
                if p.grad is None:
                    continue
                full_w, shard_info = self._get_full_weight(p)
                w_f = full_w.float()
                col_norms = w_f.norm(dim=0, keepdim=True).clamp(min=1e-8)  # [1, C]
                w_hat = w_f / col_norms

                grad_local = self._get_local(p.grad)
                lr = param_lr.get(id(p), opt.defaults['lr'])
                if shard_info is not None:
                    _, offset, local_size = shard_info
                    w_hat_local = w_hat[offset:offset + local_size]
                else:
                    w_hat_local = w_hat

                # Column-wise radial dot product
                local_dot = (grad_local.float() * w_hat_local).sum(dim=0, keepdim=True)
                fsdp_pg = self._get_fsdp_process_group(p)
                if fsdp_pg is not None:
                    dist.all_reduce(local_dot, group=fsdp_pg)

                # Predict radial displacement per column
                state = opt.state.get(p, {})
                exp_avg = state.get("exp_avg", None)
                exp_avg_sq = state.get("exp_avg_sq", None)

                if exp_avg is not None and exp_avg_sq is not None:
                    m_local = self._get_local(exp_avg)
                    v_local = self._get_local(exp_avg_sq)
                    beta1 = opt.defaults['betas'][0]
                    m_pred = beta1 * m_local.float() + (1 - beta1) * grad_local.float()
                    adam_dir = m_pred / (v_local.float().sqrt() + self._eps)
                    step_radial = (adam_dir * w_hat_local).sum(dim=0, keepdim=True)
                    if fsdp_pg is not None:
                        dist.all_reduce(step_radial, group=fsdp_pg)
                    delta_r = -lr * step_radial
                else:
                    delta_r = lr * local_dot.abs()

                predicted_norm = col_norms + delta_r.clamp(min=0)

                if self._soft_blend:
                    overshoot = (predicted_norm - self._max_norm).clamp(min=0)
                    sigma = (overshoot / delta_r.abs().clamp(min=1e-8)).clamp(0.0, 1.0)
                    project_g = (local_dot < 0).float()
                    self._get_local(p.grad).copy_(
                        (grad_local.float() - sigma * project_g * (local_dot * w_hat_local)).to(p.grad.dtype)
                    )
                else:
                    active = (predicted_norm > self._max_norm)
                    project_g = active.float() * (local_dot < 0).float()
                    self._get_local(p.grad).copy_(
                        (grad_local.float() - project_g * (local_dot * w_hat_local)).to(p.grad.dtype)
                    )

                # Momentum projection
                if exp_avg is not None:
                    m_local = self._get_local(exp_avg)
                    local_dot_m = (m_local.float() * w_hat_local).sum(dim=0, keepdim=True)
                    if fsdp_pg is not None:
                        dist.all_reduce(local_dot_m, group=fsdp_pg)
                    if self._soft_blend:
                        project_m = (local_dot_m < 0).float()
                        self._get_local(exp_avg).sub_(
                            (sigma * project_m * (local_dot_m * w_hat_local)).to(exp_avg.dtype)
                        )
                    else:
                        project_m = active.float() * (local_dot_m < 0).float()
                        self._get_local(exp_avg).sub_(
                            (project_m * local_dot_m * w_hat_local).to(exp_avg.dtype)
                        )

                # correct_v
                if self._correct_v and exp_avg_sq is not None:
                    if self._soft_blend:
                        self._get_local(exp_avg_sq).mul_(
                            (1.0 - sigma * (w_hat_local * w_hat_local)).to(exp_avg_sq.dtype)
                        )
                    else:
                        self._get_local(exp_avg_sq).mul_(torch.where(
                            active.expand_as(w_hat_local),
                            1.0 - w_hat_local * w_hat_local,
                            torch.ones_like(w_hat_local),
                        ).to(exp_avg_sq.dtype))

    def _get_fsdp_process_group(self, p: torch.nn.Parameter):
        """Return FSDP process group for all-reduce, or None for plain tensors."""
        try:
            from torch.distributed._tensor import DTensor
            if not isinstance(p.data, DTensor):
                return None
        except ImportError:
            return None
        dt = p.data
        from torch.distributed._tensor.placement_types import Shard as ShardPlacement
        mesh_dim = 0
        for i, pl in enumerate(dt.placements):
            if isinstance(pl, ShardPlacement) and pl.dim == 0:
                mesh_dim = i
                break
        return dt.device_mesh.get_group(mesh_dim)

    def _project_single_bounded(
        self, p: torch.nn.Parameter, opt: torch.optim.AdamW,
        param_lr: dict[int, float],
    ):
        """Project a single bounded param's gradient/momentum using predictive activation."""
        grad = p.grad
        w = p.data
        w_f = w.float()
        row_norms = w_f.norm(dim=-1, keepdim=True)
        w_hat = w_f / row_norms.clamp(min=1e-8)  # keep in f32
        del w_f

        state = opt.state.get(p, {})
        exp_avg = state.get("exp_avg", None)
        exp_avg_sq = state.get("exp_avg_sq", None)
        lr = param_lr.get(id(p), opt.defaults['lr'])

        # Predictive activation
        if exp_avg is not None and exp_avg_sq is not None:
            beta1 = opt.defaults['betas'][0]
            m_pred = beta1 * exp_avg.float() + (1 - beta1) * grad.float()
            adam_dir = m_pred / (exp_avg_sq.float().sqrt() + self._eps)
            step_radial = (adam_dir * w_hat).sum(dim=-1, keepdim=True)
            delta_r = -lr * step_radial
        else:
            radial_g = (grad.float() * w_hat).sum(dim=-1, keepdim=True)
            delta_r = lr * radial_g.abs()

        predicted_norm = row_norms + delta_r.clamp(min=0)
        radial_dot_g = (grad.float() * w_hat).sum(dim=-1, keepdim=True)

        if self._soft_blend:
            overshoot = (predicted_norm - self._max_norm).clamp(min=0)
            sigma = (overshoot / delta_r.abs().clamp(min=1e-8)).clamp(0.0, 1.0)  # f32
            project_g = (radial_dot_g < 0).float()
            p.grad = (grad.float() - sigma * project_g * (radial_dot_g * w_hat)).to(grad.dtype)

            if exp_avg is not None:
                radial_dot_m = (exp_avg.float() * w_hat).sum(dim=-1, keepdim=True)
                project_m = (radial_dot_m < 0).float()
                exp_avg.sub_((sigma * project_m * (radial_dot_m * w_hat)).to(exp_avg.dtype))

            if self._correct_v and exp_avg_sq is not None:
                exp_avg_sq.mul_((1.0 - sigma * (w_hat * w_hat)).to(exp_avg_sq.dtype))
        else:
            active = (predicted_norm > self._max_norm)
            project_g = (active & (radial_dot_g < 0)).float()
            p.grad = (grad.float() - project_g * (radial_dot_g * w_hat)).to(grad.dtype)

            if exp_avg is not None:
                radial_dot_m = (exp_avg.float() * w_hat).sum(dim=-1, keepdim=True)
                project_m = (active & (radial_dot_m < 0)).float()
                exp_avg.sub_((project_m * radial_dot_m * w_hat).to(exp_avg.dtype))

            if self._correct_v and exp_avg_sq is not None:
                exp_avg_sq.mul_(torch.where(
                    active.expand_as(w_hat), 1.0 - w_hat * w_hat,
                    torch.ones_like(w_hat),
                ).to(exp_avg_sq.dtype))

    def _get_eye(self, d_in: int, device: torch.device) -> torch.Tensor:
        """Return cached identity matrix for Newton-Schulz iterations."""
        key = (d_in, device)
        if key not in self._eye_cache:
            self._eye_cache[key] = torch.eye(d_in, device=device, dtype=torch.float32)
        return self._eye_cache[key]

    def _should_run_partial_orthogonal(self) -> int:
        """Return number of partial_orthogonal passes to run this step.

        n_iter=2.0 -> 2 passes every step, 1.0 -> 1 pass every step,
        0.5 -> 1 pass every 2nd step, 0.2 -> 1 pass every 5th step.
        Returns 0 when this step should be skipped.
        """
        if self._n_iter >= 1.0:
            return int(self._n_iter)
        if self._n_iter <= 0:
            return 0
        period = int(round(1.0 / self._n_iter))
        return 1 if (self._step_count % period) == 0 else 0

    @torch.no_grad()
    def _apply_partial_orthogonal(self, p: torch.nn.Parameter, bounded: bool = False):
        """Apply partial orthogonal constraint, dispatching to structured variants if enabled."""
        if self._ffn_down_left_ns:
            meta = self._po_metadata.get(id(p))
            if meta is not None and meta["po_type"] == "ffn":
                return self._apply_po_ffn_adaptive(p, bounded)
        self._apply_po_generic(p, bounded)

    def _get_full_weight(self, p: torch.nn.Parameter):
        """Get full (unsharded) weight and shard write-back info.

        For DTensors (FSDP2), all-gathers the full weight so PO operates on
        the complete matrix.  Returns (full_W, shard_info) where shard_info is
        None for plain tensors or (local_tensor, offset, local_size) for DTensors.
        """
        try:
            from torch.distributed._tensor import DTensor
            is_dtensor = isinstance(p.data, DTensor)
        except ImportError:
            is_dtensor = False

        if not is_dtensor:
            return p.data, None

        dt = p.data
        full_W = dt.full_tensor()
        local_tensor = dt._local_tensor
        local_size = local_tensor.shape[0]
        full_size = full_W.shape[0]

        # Find FSDP shard mesh dimension
        from torch.distributed._tensor.placement_types import Shard as ShardPlacement
        mesh_dim = 0
        for i, pl in enumerate(dt.placements):
            if isinstance(pl, ShardPlacement) and pl.dim == 0:
                mesh_dim = i
                break

        local_rank = dt.device_mesh.get_local_rank(mesh_dim)
        chunk_size = math.ceil(full_size / dt.device_mesh.size(mesh_dim))
        offset = local_rank * chunk_size

        return full_W, (local_tensor, offset, local_size)

    def _write_back_shard(
        self, result: torch.Tensor, p: torch.nn.Parameter, shard_info,
    ):
        """Write PO result back, extracting local shard for DTensors."""
        if shard_info is None:
            p.data.copy_(result)
        else:
            local_tensor, offset, local_size = shard_info
            local_tensor.copy_(result[offset:offset + local_size])

    def _deterministic_power_init(self, W_f: torch.Tensor):
        """Deterministic power iteration init from weight data.

        Uses column/row sums instead of torch.randn so all FSDP ranks
        compute identical u, v from the same full_tensor().
        """
        u = F.normalize(W_f.sum(dim=1), dim=0)  # (d_out,)
        v = F.normalize(W_f.sum(dim=0), dim=0)  # (d_in,)
        return u, v

    def _run_ns_iterations(self, X: torch.Tensor, use_left: bool = False) -> torch.Tensor:
        """Run Newton-Schulz iterations (left or right sided)."""
        if use_left:
            I = self._get_eye(X.shape[0], X.device)
            for _ in range(self._n_iter_ns):
                X = 0.5 * (3 * I - X @ X.T) @ X
        else:
            I = self._get_eye(X.shape[1], X.device)
            for _ in range(self._n_iter_ns):
                X = 0.5 * X @ (3 * I - X.T @ X)
        return X

    def _apply_ns_blend(
        self, X_spectral: torch.Tensor, p: torch.nn.Parameter, use_left: bool = False,
    ) -> torch.Tensor:
        """Apply Newton-Schulz with mode-dependent blending.

        Args:
            X_spectral: Weight matrix after spectral normalization (W/sigma_max), float32.
            p: The parameter (for per-param EMA tracking in adaptive mode).
            use_left: If True, use left-sided NS (d_out x d_out Gram).
        """
        pid = id(p)

        if self._ns_mode == "full":
            # Full NS, no blending
            if self._n_iter_ns <= 0:
                return X_spectral
            return self._run_ns_iterations(X_spectral, use_left)

        elif self._ns_mode == "lerp":
            # Explicit linear interpolation using ns_alpha
            if self._n_iter_ns <= 0 or self._ns_alpha <= 0.0:
                return X_spectral
            X_ns = self._run_ns_iterations(X_spectral.clone(), use_left)
            if self._ns_alpha >= 1.0:
                return X_ns
            return (1.0 - self._ns_alpha) * X_spectral + self._ns_alpha * X_ns

        elif self._ns_mode == "adaptive":
            # Augmented Lagrangian: NS proportional to kappa violation
            if self._n_iter_ns <= 0:
                return X_spectral

            # kappa_est: after spectral norm sigma_max=1, so kappa = 1/sigma_rms
            d = min(X_spectral.shape)
            sigma_rms = X_spectral.norm(p='fro').item() / math.sqrt(d)
            kappa_est = 1.0 / max(sigma_rms, 1e-8)

            # EMA update
            if pid in self._kappa_ema:
                kappa_ema = (
                    self._kappa_ema_beta * self._kappa_ema[pid]
                    + (1.0 - self._kappa_ema_beta) * kappa_est
                )
            else:
                kappa_ema = kappa_est
            self._kappa_ema[pid] = kappa_ema

            # Violation and adaptive lambda
            violation = max(kappa_ema - self._kappa_target, 0.0) / self._kappa_target
            ns_lambda = min(violation * self._lambda_max, self._lambda_max)

            # Store for logging
            self._adaptive_stats[pid] = {
                "ns_lambda": ns_lambda,
                "kappa_ema": kappa_ema,
                "violation": violation,
            }

            if ns_lambda < 1e-6:
                return X_spectral

            X_ns = self._run_ns_iterations(X_spectral.clone(), use_left)
            return (1.0 - ns_lambda) * X_spectral + ns_lambda * X_ns

        elif self._ns_mode == "schedule":
            # Full NS → adaptive transition over ns_schedule_steps
            if self._n_iter_ns <= 0:
                return X_spectral

            # Schedule progress: 0 → full NS, 1 → adaptive
            t = min(self._step_count / max(self._ns_schedule_steps, 1), 1.0)

            # Compute adaptive lambda (same as adaptive mode)
            d = min(X_spectral.shape)
            sigma_rms = X_spectral.norm(p='fro').item() / math.sqrt(d)
            kappa_est = 1.0 / max(sigma_rms, 1e-8)

            if pid in self._kappa_ema:
                kappa_ema = (
                    self._kappa_ema_beta * self._kappa_ema[pid]
                    + (1.0 - self._kappa_ema_beta) * kappa_est
                )
            else:
                kappa_ema = kappa_est
            self._kappa_ema[pid] = kappa_ema

            violation = max(kappa_ema - self._kappa_target, 0.0) / self._kappa_target
            adaptive_lambda = min(violation * self._lambda_max, self._lambda_max)

            # Blend: full NS at t=0, adaptive at t=1
            effective_lambda = (1.0 - t) * 1.0 + t * adaptive_lambda

            self._adaptive_stats[pid] = {
                "ns_lambda": effective_lambda,
                "kappa_ema": kappa_ema,
                "violation": violation,
            }

            X_ns = self._run_ns_iterations(X_spectral.clone(), use_left)
            if effective_lambda >= 1.0 - 1e-6:
                return X_ns
            return (1.0 - effective_lambda) * X_spectral + effective_lambda * X_ns

        else:
            raise ValueError(
                f"Unknown ns_mode: {self._ns_mode!r}. "
                "Use 'full', 'lerp', 'adaptive', or 'schedule'."
            )

    @torch.no_grad()
    def _apply_po_generic(self, p: torch.nn.Parameter, bounded: bool = False):
        """Generic partial orthogonal: right-sided Newton-Schulz on d_in x d_in Gram.

        Uses full_tensor() for DTensors so NS decorrelates all rows globally.
        """
        full_W, shard_info = self._get_full_weight(p)
        dtype = full_W.dtype
        W_f = full_W.float()
        d_out, d_in = W_f.shape

        # Step 1: Power iteration spectral normalization (warm-started)
        pid = id(p)
        cached = self._power_iter_state.get(pid)
        if cached is not None:
            u, v = cached[0].float().to(W_f.device), cached[1].float().to(W_f.device)
            if u.shape[0] != d_out or v.shape[0] != d_in:
                u, v = self._deterministic_power_init(W_f)
        else:
            u, v = self._deterministic_power_init(W_f)

        for _ in range(self._n_iter_spectral):
            v = F.normalize(W_f.T @ u, dim=0)
            u = F.normalize(W_f @ v, dim=0)

        sigma_max = (u @ W_f @ v).abs().clamp(min=1e-5)
        X = W_f / sigma_max

        self._power_iter_state[pid] = (u.to(dtype), v.to(dtype))

        # Step 2: Newton-Schulz with mode-dependent blending
        X = self._apply_ns_blend(X, p, use_left=False)

        # Step 3: Row normalization
        if bounded:
            row_norms = X.norm(dim=1, keepdim=True)
            X = X / row_norms.clamp(min=1.0)
        else:
            X = F.normalize(X, dim=1)

        self._write_back_shard(X.to(dtype), p, shard_info)

    @torch.no_grad()
    def _apply_po_ffn_adaptive(self, p: torch.nn.Parameter, bounded: bool = False):
        """Adaptive-direction partial orthogonal for FFN weights.

        Uses full_tensor() for DTensors.  Left-sided NS when d_out < d_in
        (downscale: cheaper d_out x d_out Gram), right-sided otherwise.
        """
        full_W, shard_info = self._get_full_weight(p)
        dtype = full_W.dtype
        W_f = full_W.float()
        d_out, d_in = W_f.shape
        use_left = d_out < d_in

        # Power iteration (single u, v pair, warm-started, deterministic init)
        pid = id(p)
        cached = self._power_iter_state.get(pid)
        if cached is not None:
            u, v = cached[0].float().to(W_f.device), cached[1].float().to(W_f.device)
            if u.shape[0] != d_out or v.shape[0] != d_in:
                u, v = self._deterministic_power_init(W_f)
        else:
            u, v = self._deterministic_power_init(W_f)

        for _ in range(self._n_iter_spectral):
            v = F.normalize(W_f.T @ u, dim=0)
            u = F.normalize(W_f @ v, dim=0)

        sigma_max = (u @ W_f @ v).abs().clamp(min=1e-5)
        X = W_f / sigma_max

        self._power_iter_state[pid] = (u.to(dtype), v.to(dtype))

        # Newton-Schulz with mode-dependent blending
        X = self._apply_ns_blend(X, p, use_left=use_left)

        # Row normalization
        if bounded:
            row_norms = X.norm(dim=1, keepdim=True).clamp(min=1.0)
            X = X / row_norms
        else:
            X = F.normalize(X, dim=1)

        self._write_back_shard(X.to(dtype), p, shard_info)

    @torch.no_grad()
    def _enforce_constraints(self):
        """Apply per-param constraint in float32 (matching GPTNormalizer precision)."""
        po_passes = self._should_run_partial_orthogonal()

        for opt_idx, (normalized_params, _, po_params, pob_params, norm_dim0, bound_dim0) in enumerate(zip(
            self._normalized_params_per_optimizer,
            self._bounded_params_per_optimizer,
            self._partial_ortho_params_per_optimizer,
            self._partial_ortho_bounded_params_per_optimizer,
            self._normalized_dim0_params_per_optimizer,
            self._bounded_dim0_params_per_optimizer,
        )):
            # ---- dim=-1 (row-wise) normalized params: local ops ----
            for p in normalized_params:
                p_f = p.float()
                norms = p_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                p.copy_((p_f / norms).to(p.dtype))

            # ---- dim=-1 bounded params: batched by shape group ----
            for shape_key, group in self._bounded_shape_groups[opt_idx].items():
                if len(group) < 2:
                    # Single param — no stacking overhead
                    p = group[0]
                    p_f = p.float()
                    norms = p_f.norm(dim=-1, keepdim=True).clamp(min=self._max_norm)
                    p.copy_((p_f / norms * self._max_norm).to(p.dtype))
                    continue
                # Stack local tensors into [N, R, C] batch
                _gl = self._get_local
                locals_list = [_gl(p.data) for p in group]
                stacked_f = torch.stack(locals_list).float()  # [N, R, C] f32
                norms = stacked_f.norm(dim=-1, keepdim=True).clamp(min=self._max_norm)
                result = (stacked_f / norms * self._max_norm).to(locals_list[0].dtype)
                for i, lp in enumerate(locals_list):
                    lp.copy_(result[i])

            # ---- dim=0 (column-wise) params: full-tensor ops ----
            for p in norm_dim0:
                full_w, shard_info = self._get_full_weight(p)
                w_f = full_w.float()
                col_norms = w_f.norm(dim=0, keepdim=True).clamp(min=1e-8)
                self._write_back_shard((w_f / col_norms).to(full_w.dtype), p, shard_info)

            for p in bound_dim0:
                full_w, shard_info = self._get_full_weight(p)
                w_f = full_w.float()
                col_norms = w_f.norm(dim=0, keepdim=True).clamp(min=self._max_norm)
                self._write_back_shard((w_f / col_norms * self._max_norm).to(full_w.dtype), p, shard_info)

            if po_passes > 0:
                for _ in range(po_passes):
                    for p in po_params:
                        self._apply_partial_orthogonal(p, bounded=False)
                    for p in pob_params:
                        self._apply_partial_orthogonal(p, bounded=True)
            else:
                # Cheap fallback: row-norm to keep on manifold between full PO passes
                for p in po_params:
                    p_f = p.float()
                    norms = p_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    p.copy_((p_f / norms).to(p.dtype))
                for p in pob_params:
                    p_f = p.float()
                    norms = p_f.norm(dim=-1, keepdim=True).clamp(min=1.0)
                    p.copy_((p_f / norms).to(p.dtype))

    @torch.no_grad()
    def compute_spectral_stats(self) -> dict[str, float]:
        """Compute spectral stats (σ_max, σ_rms, κ_est) for all PO params.

        Returns a dict of tensorboard-ready metrics keyed by param name.
        """
        stats: dict[str, float] = {}
        for po_params, pob_params in zip(
            self._partial_ortho_params_per_optimizer,
            self._partial_ortho_bounded_params_per_optimizer,
        ):
            for p in itertools.chain(po_params, pob_params):
                full_W, _ = self._get_full_weight(p)
                W = full_W.float()

                # σ_max via power iteration (reuse cached vectors)
                pid = id(p)
                cached = self._power_iter_state.get(pid)
                if cached is not None:
                    u, v = cached[0].float().to(W.device), cached[1].float().to(W.device)
                else:
                    u = F.normalize(W.sum(dim=1), dim=0)
                    v = F.normalize(W.sum(dim=0), dim=0)

                for _ in range(10):
                    v = F.normalize(W.T @ u, dim=0)
                    u = F.normalize(W @ v, dim=0)
                sigma_max = (u @ W @ v).abs().item()

                # σ_rms from Frobenius norm (cheap condition number estimate)
                sigma_fro = W.norm(p='fro').item()
                d = min(W.shape)
                sigma_rms = sigma_fro / math.sqrt(d)
                condition_est = sigma_max / max(sigma_rms, 1e-8)

                name = self._param_names.get(pid, f"param_{pid}")
                clean = name.replace(".", "/")
                stats[f"spectral/{clean}/sigma_max"] = sigma_max
                stats[f"spectral/{clean}/sigma_rms"] = sigma_rms
                stats[f"spectral/{clean}/kappa_est"] = condition_est

                # Adaptive/schedule mode per-param stats
                if self._ns_mode in ("adaptive", "schedule") and pid in self._adaptive_stats:
                    adaptive = self._adaptive_stats[pid]
                    stats[f"spectral/{clean}/ns_lambda"] = adaptive["ns_lambda"]
                    stats[f"spectral/{clean}/kappa_ema"] = adaptive["kappa_ema"]
                    stats[f"spectral/{clean}/violation"] = adaptive["violation"]
        return stats
