"""
ConstrainedAdam: Unified Adam optimizer for row-constrained weight matrices.

Supports two constraint modes:
- "bounded": Rows are bounded by max_norm (‖w_i‖ ≤ max_norm)
- "normalized": Rows are normalized to unit norm (‖w_i‖ = 1)

Both modes use tangent-space gradient projection to prevent momentum
corruption from the constraint projection. This is more principled than
applying standard Adam followed by post-hoc normalization.

Compatible with torchtitan's OptimizersContainer pattern.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.ft import FTManager
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims


__all__ = [
    "ConstrainedAdam",
    "ConstrainedAdamWithFallback",
    "ConstrainedAdamOptimizerContainer",
    "configure_constrained_adam",
    "make_build_constrained_adam_optimizers",
]


_constrained_adam_config: dict = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "mode": "normalized",  # "bounded" or "normalized"
    "max_norm": 1.0,
    "delta": 0.0,  # only for bounded mode
    "project_momentum": True,
    "parallel_transport": True,  # only for normalized mode
    "fallback_lr": None,
    "embedding_lr": None,
    "embedding_norm": True,  # always normalize embeddings
}


def configure_constrained_adam(
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    mode: str = "bounded",
    max_norm: float = 1.0,
    delta: float = 0.0,
    project_momentum: bool = True,
    parallel_transport: bool = True,
    fallback_lr: Optional[float] = None,
    embedding_lr: Optional[float] = None,
    embedding_norm: bool = False,
) -> None:
    """Configure ConstrainedAdam settings used by make_build_constrained_adam_optimizers().

    Args:
        betas: Adam beta coefficients.
        eps: Adam epsilon for numerical stability.
        weight_decay: Decoupled weight decay (AdamW style).
        mode: Constraint mode - "bounded" or "normalized".
        max_norm: Maximum row norm (for bounded mode, ignored in normalized mode).
        delta: Tolerance for constraint activation in bounded mode.
        project_momentum: Project first-moment buffer onto tangent space.
        parallel_transport: Re-project momentum after update (normalized mode only).
        fallback_lr: Learning rate for non-matrix parameters.
        embedding_lr: Learning rate for embedding parameters.
        embedding_norm: Apply L2 normalization to embeddings after step.
    """
    _constrained_adam_config["betas"] = betas
    _constrained_adam_config["eps"] = eps
    _constrained_adam_config["weight_decay"] = weight_decay
    _constrained_adam_config["mode"] = mode
    _constrained_adam_config["max_norm"] = max_norm
    _constrained_adam_config["delta"] = delta
    _constrained_adam_config["project_momentum"] = project_momentum
    _constrained_adam_config["parallel_transport"] = parallel_transport
    _constrained_adam_config["fallback_lr"] = fallback_lr
    _constrained_adam_config["embedding_lr"] = embedding_lr
    _constrained_adam_config["embedding_norm"] = embedding_norm


class ConstrainedAdam(Optimizer):
    """Adam optimizer with row-wise constraints on weight matrices.

    Supports two constraint modes:

    **Bounded mode** (mode="bounded"):
        Each row is projected onto the ℓ₂ ball: ‖w_i‖ ≤ max_norm.
        Tangent-space projection is applied only to rows near the boundary
        (where ‖w_i‖ ≥ max_norm - delta).

    **Normalized mode** (mode="normalized"):
        Each row is normalized to unit norm: ‖w_i‖ = 1.
        All rows are treated as being on the unit sphere, so tangent-space
        projection is always applied. Optionally uses parallel transport
        to re-project momentum onto the new tangent space after the update.

    For non-matrix parameters (biases, norms, etc.), standard Adam is used.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running averages (default: (0.9, 0.999)).
        eps: Term for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay (default: 0).
        mode: Constraint mode - "bounded" or "normalized" (default: "bounded").
        max_norm: Maximum row norm for bounded mode (default: 1.0).
        delta: Tolerance for constraint activation in bounded mode (default: 0.0).
        project_momentum: Project first-moment buffer (default: True).
        parallel_transport: Re-project momentum in normalized mode (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        mode: str = "bounded",
        max_norm: float = 1.0,
        delta: float = 0.0,
        project_momentum: bool = True,
        parallel_transport: bool = True,
    ):
        if mode not in ("bounded", "normalized"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'bounded' or 'normalized'.")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if max_norm <= 0:
            raise ValueError(f"Invalid max_norm: {max_norm}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mode=mode,
            max_norm=max_norm,
            delta=delta,
            project_momentum=project_momentum,
            parallel_transport=parallel_transport,
        )
        super().__init__(params, defaults)

    @staticmethod
    @torch.no_grad()
    def _project_to_tangent(vecs: Tensor, w_hat: Tensor) -> Tensor:
        """Remove radial component: v_tan = v - (v · ŵ) * ŵ"""
        dots = (vecs * w_hat).sum(dim=-1, keepdim=True)
        return vecs - dots * w_hat

    @staticmethod
    @torch.no_grad()
    def _project_to_ball(w: Tensor, max_norm: float) -> None:
        """Project rows onto ℓ₂ ball: ‖w_i‖ ≤ max_norm (in-place)."""
        norms = w.norm(dim=-1, keepdim=True)
        scale = norms.clamp(min=max_norm)
        w.mul_(max_norm / scale)

    @staticmethod
    @torch.no_grad()
    def _normalize_rows(w: Tensor) -> None:
        """Normalize rows to unit norm: ‖w_i‖ = 1 (in-place)."""
        norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        w.div_(norms)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            mode = group["mode"]
            max_norm = group["max_norm"]
            delta = group["delta"]
            proj_m = group["project_momentum"]
            parallel_transport = group["parallel_transport"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ConstrainedAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                is_matrix = p.dim() >= 2

                # Decoupled weight decay (skip for normalized since norm is fixed)
                if wd != 0 and not (is_matrix and mode == "normalized"):
                    p.mul_(1.0 - lr * wd)

                if is_matrix:
                    orig_shape = p.shape
                    w_2d = p.view(-1, orig_shape[-1])
                    g_2d = grad.view(-1, orig_shape[-1])
                    m_2d = m.view(-1, orig_shape[-1])

                    if mode == "bounded":
                        # Bounded mode: project only active rows (near boundary)
                        # Use static-shape operations to be compatible with torch.compile/FSDP
                        row_norms = w_2d.norm(dim=-1, keepdim=True)
                        active = (row_norms >= (max_norm - delta))  # shape: [N, 1]

                        # Compute w_hat for all rows (safe normalization)
                        w_hat = w_2d / row_norms.clamp(min=1e-12)

                        # Project gradient: remove radial component for active rows
                        g_dot = (g_2d * w_hat).sum(dim=-1, keepdim=True)
                        g_projected = g_2d - g_dot * w_hat
                        # Apply only to active rows using where (static shape)
                        g_2d_new = torch.where(active, g_projected, g_2d)
                        # Copy back to grad view
                        grad.view_as(g_2d_new).copy_(g_2d_new)

                        if proj_m:
                            # Project momentum for active rows
                            m_dot = (m_2d * w_hat).sum(dim=-1, keepdim=True)
                            m_projected = m_2d - m_dot * w_hat
                            m_2d_new = torch.where(active, m_projected, m_2d)
                            m.view_as(m_2d_new).copy_(m_2d_new)

                        # Standard Adam moment updates (use potentially modified grad)
                        m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                        # Bias correction and update
                        bc1 = 1.0 - beta1 ** step
                        bc2 = 1.0 - beta2 ** step
                        step_size = lr / bc1
                        denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                        p.addcdiv_(m, denom, value=-step_size)

                        # Project to ball
                        w_2d = p.view(-1, orig_shape[-1])
                        self._project_to_ball(w_2d, max_norm)

                    else:  # mode == "normalized"
                        # Normalized mode: all rows on unit sphere
                        w = p.data

                        # Project gradient onto tangent space
                        dot = (grad * w).sum(dim=-1, keepdim=True)
                        g_tan = grad - dot * w

                        # Update moments with tangential gradient
                        m.mul_(beta1).add_(g_tan, alpha=1.0 - beta1)
                        v.mul_(beta2).addcmul_(g_tan, g_tan, value=1.0 - beta2)

                        # Parallel transport: re-project momentum onto tangent space
                        if parallel_transport and proj_m:
                            m_dot = (m * w).sum(dim=-1, keepdim=True)
                            m.sub_(m_dot * w)

                        # Bias correction and update
                        bc1 = 1.0 - beta1 ** step
                        bc2 = 1.0 - beta2 ** step
                        m_hat = m / bc1
                        v_hat = v / bc2
                        p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

                        # Normalize to unit sphere
                        self._normalize_rows(p.data.view(-1, orig_shape[-1]))

                else:
                    # Non-matrix params: standard Adam
                    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bc1 = 1.0 - beta1 ** step
                    bc2 = 1.0 - beta2 ** step
                    step_size = lr / bc1
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    p.addcdiv_(m, denom, value=-step_size)

        return loss


# ---------------------------------------------------------------------------
# ConstrainedAdamWithFallback: handles all param types
# ---------------------------------------------------------------------------


class ConstrainedAdamWithFallback(Optimizer):
    """
    ConstrainedAdam with a fallback optimizer for non-matrix parameters.

    Routes 2D weight matrices to ConstrainedAdam (with row constraints)
    and delegates everything else (embeddings, biases, norms) to a
    fallback optimizer (default: AdamW).

    Args:
        params: Param groups with ``use_constrained`` flag.
        lr: Default learning rate.
        betas: Adam beta coefficients.
        eps: Adam epsilon.
        weight_decay: Decoupled weight decay.
        mode: Constraint mode - "bounded" or "normalized".
        max_norm: Maximum row norm for bounded mode.
        delta: Tolerance for constraint activation.
        project_momentum: Project first-moment buffer.
        parallel_transport: Re-project momentum (normalized mode).
        fallback_cls: Fallback optimizer class (default: AdamW).
        fallback_defaults: Default kwargs for the fallback optimizer.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        mode: str = "bounded",
        max_norm: float = 1.0,
        delta: float = 0.0,
        project_momentum: bool = True,
        parallel_transport: bool = True,
        fallback_cls: type = None,
        fallback_defaults: Optional[dict] = None,
    ):
        if fallback_cls is None:
            fallback_cls = torch.optim.AdamW
        if fallback_defaults is None:
            fallback_defaults = dict(
                betas=betas, eps=eps, weight_decay=weight_decay
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mode=mode,
            max_norm=max_norm,
            delta=delta,
            project_momentum=project_momentum,
            parallel_transport=parallel_transport,
            use_constrained=True,
        )
        super().__init__(params, defaults)

        # Separate params into constrained and fallback groups
        constrained_params = []
        fallback_groups = []
        for group in self.param_groups:
            use_constrained = group.get("use_constrained", True)
            group_lr = group.get("lr", lr)
            group_constrained = []
            group_fallback = []
            for p in group["params"]:
                if use_constrained and p.ndim >= 2:
                    group_constrained.append(p)
                else:
                    group_fallback.append(p)
            constrained_params.extend(group_constrained)
            if group_fallback:
                fallback_groups.append({"params": group_fallback, "lr": group_lr})

        # Internal ConstrainedAdam optimizer
        self._constrained = ConstrainedAdam(
            constrained_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mode=mode,
            max_norm=max_norm,
            delta=delta,
            project_momentum=project_momentum,
            parallel_transport=parallel_transport,
        ) if constrained_params else None

        # Internal fallback optimizer
        fallback_defaults.pop("lr", None)
        self._fallback = fallback_cls(
            fallback_groups,
            lr=lr,
            **fallback_defaults,
        ) if fallback_groups else None

    @torch.no_grad()
    def step(self, closure=None):
        """Step both internal optimizers."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._constrained is not None:
            self._constrained.step()
        if self._fallback is not None:
            self._fallback.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both internal optimizers."""
        if self._constrained is not None:
            self._constrained.zero_grad(set_to_none=set_to_none)
        if self._fallback is not None:
            self._fallback.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "constrained": self._constrained.state_dict() if self._constrained else None,
            "fallback": self._fallback.state_dict() if self._fallback else None,
        }

    def load_state_dict(self, state_dict):
        if self._constrained is not None and state_dict.get("constrained") is not None:
            self._constrained.load_state_dict(state_dict["constrained"])
        if self._fallback is not None and state_dict.get("fallback") is not None:
            self._fallback.load_state_dict(state_dict["fallback"])


# ---------------------------------------------------------------------------
# torchtitan integration: build_optimizers factory
# ---------------------------------------------------------------------------


def build_constrained_adam_optimizers(
    model_parts: list[torch.nn.Module],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    mode: str = "bounded",
    max_norm: float = 1.0,
    delta: float = 0.0,
    project_momentum: bool = True,
    parallel_transport: bool = True,
    fallback_lr: float = 3e-4,
) -> list[ConstrainedAdamWithFallback]:
    """
    Factory function following torchtitan's build_optimizers pattern.

    Creates one ConstrainedAdamWithFallback optimizer per model part.

    Args:
        model_parts: List of nn.Module (one per pipeline stage).
        lr: Learning rate for constrained 2D weights.
        betas: Adam beta coefficients.
        eps: Adam epsilon.
        weight_decay: Decoupled weight decay.
        mode: Constraint mode - "bounded" or "normalized".
        max_norm: Maximum row norm for bounded mode.
        delta: Tolerance for constraint activation.
        project_momentum: Project first-moment buffer.
        parallel_transport: Re-project momentum (normalized mode).
        fallback_lr: Learning rate for fallback optimizer.

    Returns:
        List of ConstrainedAdamWithFallback optimizers.
    """
    optimizers = []
    for model in model_parts:
        matrix_params = []
        other_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_hidden_matrix = (
                p.ndim >= 2
                and "embed" not in name
                and "head" not in name
                and "norm" not in name
                and "output" not in name
            )
            if is_hidden_matrix:
                matrix_params.append(p)
            else:
                other_params.append(p)

        param_groups = [
            dict(params=matrix_params, use_constrained=True, lr=lr),
            dict(params=other_params, use_constrained=False, lr=fallback_lr),
        ]

        optimizer = ConstrainedAdamWithFallback(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            mode=mode,
            max_norm=max_norm,
            delta=delta,
            project_momentum=project_momentum,
            parallel_transport=parallel_transport,
            fallback_defaults=dict(
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            ),
        )
        optimizers.append(optimizer)

    return optimizers


def make_build_constrained_adam_optimizers(enable_embedding_norm: bool = False):
    """
    Factory function that creates a build_optimizers_fn for ConstrainedAdam.

    Uses module-level configuration set via configure_constrained_adam().

    Args:
        enable_embedding_norm: If True, apply L2 normalization to embeddings.

    Optimizer groups:
      - 2D hidden matrices: ConstrainedAdam (bounded or normalized)
      - Embeddings (tok_embeddings, output): Adam, optionally L2-normalized
      - 1D/scalars (biases, norms, gates): Adam (no weight decay)
    """

    def build_constrained_adam_optimizers_fn(
        model_parts: list[torch.nn.Module],
        optimizer_config: OptimizerConfig,
        parallel_dims: ParallelDims,
        ft_manager: FTManager,
    ):
        if optimizer_config.early_step_in_backward:
            raise NotImplementedError(
                "ConstrainedAdam does not support early_step_in_backward."
            )

        fallback_lr = _constrained_adam_config.get("fallback_lr")
        if fallback_lr is None or fallback_lr <= 0:
            fallback_lr = optimizer_config.lr

        embedding_lr = _constrained_adam_config.get("embedding_lr")
        if embedding_lr is None or embedding_lr <= 0:
            embedding_lr = fallback_lr

        return ConstrainedAdamOptimizerContainer(
            model_parts=model_parts,
            constrained_lr=optimizer_config.lr,
            embedding_lr=embedding_lr,
            scalar_lr=fallback_lr,
            constrained_config=_constrained_adam_config,
            enable_embedding_norm=enable_embedding_norm,
        )

    return build_constrained_adam_optimizers_fn


class ConstrainedAdamOptimizerContainer(OptimizersContainer):
    """
    Container for ConstrainedAdam + Adam optimizers with optional embedding normalization.

    Extends TorchTitan's OptimizersContainer to create one ConstrainedAdamWithFallback
    per model part. Each optimizer internally routes:
      - 2D hidden matrices → ConstrainedAdam (bounded or normalized)
      - Embeddings → Adam (separate LR, optionally L2-normalized after step)
      - 1D/scalars → Adam (no weight decay)

    Fully compatible with torchtitan's LR scheduler, checkpointing, and
    pipeline parallelism patterns.
    """

    def __init__(
        self,
        model_parts: list[torch.nn.Module],
        constrained_lr: float,
        embedding_lr: float,
        scalar_lr: float,
        constrained_config: dict,
        enable_embedding_norm: bool = False,
    ):
        from torchtitan.tools.logging import logger

        all_params = []
        self.optimizers: list[ConstrainedAdamWithFallback] = []
        self.model_parts = model_parts

        total_matrix = 0
        total_embed = 0
        total_scalar = 0

        mode = constrained_config["mode"]
        max_norm = constrained_config["max_norm"]

        for model in model_parts:
            mup_scales = self._get_mup_lr_scales(model)

            matrix_group = []
            down_group = []  # FFN down projections (µP: extra 1/√R lr factor)
            embed_group = []
            scalar_group = []

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                all_params.append(p)

                if p.ndim >= 2:
                    if "tok_embeddings" in name or "output" in name:
                        embed_group.append(p)
                    elif mup_scales and mup_scales.get(name, 1.0) < 1.0:
                        down_group.append(p)
                    else:
                        matrix_group.append(p)
                else:
                    scalar_group.append(p)

            total_matrix += len(matrix_group)
            total_embed += len(embed_group)
            total_scalar += len(scalar_group)

            # µP: down projections get a reduced lr
            down_lr = constrained_lr
            if down_group and mup_scales:
                down_scale = next(iter(set(
                    mup_scales.get(n, 1.0) for n, p in model.named_parameters()
                    if p.requires_grad and id(p) in {id(dp) for dp in down_group}
                )))
                down_lr = constrained_lr * down_scale

            param_groups = []
            if matrix_group:
                param_groups.append(dict(params=matrix_group, use_constrained=True, lr=constrained_lr))
            if down_group:
                param_groups.append(dict(params=down_group, use_constrained=True, lr=down_lr))
            if embed_group:
                param_groups.append(dict(params=embed_group, use_constrained=False, lr=embedding_lr))
            if scalar_group:
                param_groups.append(dict(params=scalar_group, use_constrained=False, lr=scalar_lr))

            optimizer = ConstrainedAdamWithFallback(
                param_groups,
                lr=constrained_lr,
                betas=constrained_config["betas"],
                eps=constrained_config["eps"],
                weight_decay=constrained_config["weight_decay"],
                mode=constrained_config["mode"],
                max_norm=constrained_config["max_norm"],
                delta=constrained_config["delta"],
                project_momentum=constrained_config["project_momentum"],
                parallel_transport=constrained_config["parallel_transport"],
                fallback_cls=torch.optim.Adam,
                fallback_defaults=dict(
                    betas=constrained_config["betas"],
                    eps=constrained_config["eps"],
                    weight_decay=0.0,
                ),
            )
            self.optimizers.append(optimizer)

        self._validate_length(len(self.model_parts))
        self._post_init(all_params, {"lr": constrained_lr})

        # Embedding normalization
        self._enable_embedding_norm = enable_embedding_norm
        if enable_embedding_norm:
            from titan_oellm.components.gpt_normalizer import GPTNormalizer
            self._embedding_normalizer = GPTNormalizer(
                model_parts=model_parts,
                out_norm_dim_0=False,
                bounded=False,
                normalize_full_tensor=False,
                rounding_enabled=False,
                normalize_every_n_steps=1,
            )
        else:
            self._embedding_normalizer = None

        constraint_desc = f"‖w‖≤{max_norm}" if mode == "bounded" else "‖w‖=1"
        logger.info(
            f"ConstrainedAdamOptimizerContainer created:\n"
            f"  Model parts: {len(model_parts)}\n"
            f"  Mode: {mode} ({constraint_desc})\n"
            f"  LR: matrices={constrained_lr}, down={down_lr}, embeddings={embedding_lr}, scalars={scalar_lr}\n"
            f"  2D matrices (ConstrainedAdam): {total_matrix} params\n"
            f"  Embeddings (Adam+Norm):        {total_embed} params, "
            f"norm={'enabled' if enable_embedding_norm else 'disabled'}\n"
            f"  1D/scalars (Adam):             {total_scalar} params"
        )

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

    def step(self, *args, **kwargs):
        """Perform optimizer step, then optionally normalize embeddings."""
        super().step(*args, **kwargs)

        if self._enable_embedding_norm and self._embedding_normalizer is not None:
            self._normalize_embeddings()

    @torch.no_grad()
    def _normalize_embeddings(self):
        """Apply L2 normalization to embedding weights (norm = 1 per row)."""
        norm_fn = self._embedding_normalizer.get_normalization_fn()
        for model in self.model_parts:
            m = model
            if hasattr(m, 'module'):
                m = m.module
            if hasattr(m, '_orig_mod'):
                m = m._orig_mod

            if hasattr(m, 'tok_embeddings') and m.tok_embeddings is not None:
                m.tok_embeddings.weight.data.copy_(
                    norm_fn(m.tok_embeddings.weight.data, dim=1, use_justnorm=True)
                )
