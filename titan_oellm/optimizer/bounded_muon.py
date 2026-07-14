"""
Bounded Muon Optimizer.

Combines BSA's gradient projection with true Muon semantics:
  - Pre-step: project gradients to tangent space (BSA, inherited)
  - Momentum: accumulate projected gradients into a full momentum buffer
  - NS: orthogonalize momentum via Newton-Schulz on the global gram (all-gather)
  - Step: direct SGD update using the orthogonalized momentum (no Adam v_t)
  - Post-step: enforce per-row norm constraint (BSA, inherited)

For non-constrained params (scalars, embeddings): standard fused AdamW.

Compared to SharedBoundMuon (BSM):
  BSM: project_grad → NS(raw_g)          → Adam(NS(g)) → constraint
  BM:  project_grad → m = EMA(g_proj)    → NS(m)       → SGD(NS(m)) → constraint

This fixes BSM's second-moment mismatch: BSM's v_t tracks g² while the update
direction is NS(g), creating a miscalibrated adaptive step.  Muon on momentum
is also the original Muon design known to drive weight matrices toward the
Stiefel manifold.

Supports Polar Express NS (arxiv:2505.16932): per-iteration optimal (a_t,b_t,c_t)
coefficients that consistently outperform fixed Jordan coefficients by solving a
minimax polynomial problem at each step.  Use muon_ns_mode="polar_express".

Supports Gram Newton-Schulz (Dao & Gu, 2026): reformulates NS to operate on the
n x n Gram matrix XX^T instead of the n x m rectangular X, shifting per-iteration
cost from O(mn^2) to O(n^3).  Uses float16, a restart at iteration 3, and a
larger safety factor (1.05) for numerical stability.  Mathematically equivalent
to standard PE but 25-50% faster.  Use muon_ns_mode="gram_polar_express".

Supports distributed NS ownership ("dist_*" modes) for FSDP2 scaling.  Parameters are
assigned to owner ranks via LPT-greedy scheduling (balanced by NS gram cost = 2*n^2*m + 20*n^3).
Communication is gather-to-owner + scatter-to-shards: non-owners only send/receive their local
(local_size × d_in) shard, never holding the full matrix.  The owner runs full NS and holds the
full momentum buffer.  Reduces per non-owner comm and peak memory by ~K× vs all_gather+broadcast.
The inner NS algorithm is encoded in the mode name suffix:
  "dist_muon"               — fast fixed-coefficient bf16 NS (recommended)
  "dist_gram_polar_express" — Gram NS with Polar Express coefficients
  "dist_polar_express"      — Polar Express NS
muon_surprise / adam_scale not supported in dist_* modes.

Update scaling follows Moonlight (arxiv:2502.16982): scale = 0.2*sqrt(max(R,C))
to keep the effective LR comparable to AdamW regardless of matrix shape.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.distributed as dist

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.components.ft import FTManager

from titan_oellm.optimizer.shared_bound_muon import SharedBoundMuonOptimizerContainer


__all__ = [
    "BoundedMuonOptimizerContainer",
    "configure_bounded_muon",
    "make_build_bounded_muon_optimizers",
    "_bounded_muon_config",
]


# ======================================================================
#  Polar Express coefficients — Algorithm 1, ℓ=10⁻³, k=5 iterations
#  Source: arxiv:2505.16932, "The Polar Express"
#  Raw (before safety factor) from the paper's precomputed table.
#  Safety factor: a /= f, b /= f³, c /= f⁵  with f=1.01, for all
#  iterations except the last (which is left unchanged for full convergence).
# ======================================================================

_PE_RAW_COEFFS_5 = [
    (8.287212, -23.595887, 17.300387),
    (4.107059,  -2.947850,  0.544843),
    (3.948691,  -2.908902,  0.551819),
    (3.318420,  -2.488488,  0.510049),
    (2.300652,  -1.668904,  0.418807),
]

def _compute_polar_express_coeffs(raw, safety_factor=1.01):
    """Apply per-power safety factor to all-but-last iteration."""
    result = []
    n = len(raw)
    for i, (a, b, c) in enumerate(raw):
        if i < n - 1:
            result.append((
                a / safety_factor,
                b / safety_factor ** 3,
                c / safety_factor ** 5,
            ))
        else:
            result.append((a, b, c))
    return result

_POLAR_EXPRESS_COEFFS_5 = _compute_polar_express_coeffs(_PE_RAW_COEFFS_5)

# Gram NS uses a larger safety factor (1.05) for float16 stability in the
# Gram domain, where eigenvalue errors accumulate quadratically.
_GRAM_POLAR_EXPRESS_COEFFS_5 = _compute_polar_express_coeffs(
    _PE_RAW_COEFFS_5, safety_factor=1.05
)


# ======================================================================
#  Distributed gram NS — LPT parameter ownership assignment
# ======================================================================


def _compute_dist_gram_ownership(
    eligible: list,
    world_size: int,
) -> dict[int, int]:
    """LPT-greedy parameter-to-rank assignment for dist_gram mode.

    Assigns each parameter in `eligible` to one owner rank, balancing
    total NS gram cost across ranks.  NS gram cost per parameter:
        2 * n^2 * m + 20 * n^3
    where n = min(d_out, d_in), m = max(d_out, d_in).
    This captures the dominant flops: initial/final n x m matmuls and
    5 iterations of 4 n x n gram matmuls.

    The LPT (Longest Processing Time first) heuristic sorts parameters
    by decreasing cost and greedily assigns each to the least-loaded rank.
    Approximation ratio ≤ 4/3 of optimal makespan; equals round-robin for
    uniform-cost models (standard transformers without GQA).

    Returns: dict mapping id(p) -> owner_rank.
    Computed once and cached; deterministic on all ranks since the eligible
    list is in the same order everywhere.
    """
    if world_size <= 1:
        return {id(p): 0 for p in eligible}

    costs = []
    for p in eligible:
        d_out, d_in = p.data.shape[0], p.data.shape[1]
        n = min(d_out, d_in)
        m = max(d_out, d_in)
        costs.append(2 * n * n * m + 20 * n * n * n)

    rank_loads = [0] * world_size
    assignment: dict[int, int] = {}
    for idx in sorted(range(len(eligible)), key=lambda i: -costs[i]):
        owner = min(range(world_size), key=lambda r: rank_loads[r])
        assignment[id(eligible[idx])] = owner
        rank_loads[owner] += costs[idx]
    return assignment


def _pad_rows(t: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Pad tensor with zero rows to reach target_rows (for dist.gather/scatter alignment).

    No-op (returns contiguous view) if t already has target_rows rows.
    Used to handle uneven FSDP sharding where ceil(d_out/K) may exceed local_size
    on the last rank.
    """
    if t.shape[0] == target_rows:
        return t.contiguous()
    pad = t.new_zeros(target_rows - t.shape[0], t.shape[1])
    return torch.cat([t, pad], dim=0)


# ======================================================================
#  Newton-Schulz for momentum orthogonalization
# ======================================================================

def _gram_newton_schulz_momentum(
    M: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gram Newton-Schulz orthogonalization (Dao & Gu, 2026).

    Operates on the n x n Gram matrix R = XX^T instead of the n x m rectangular
    X, shifting per-iteration cost from O(mn^2) to O(n^3).  Mathematically
    equivalent to standard Polar-Express NS but significantly faster for
    rectangular matrices (25-50% end-to-end optimizer step speedup).

    Uses float16 (not bfloat16) for internal computation — better precision in
    the critical range prevents spurious negative eigenvalues in R = XX^T.

    Includes a restart at iteration 3 (validated for PE 5-step coefficients):
    reapply accumulated Q to X, recompute R, and reset Q to I.  This bounds
    eigenvector drift and spurious eigenvalue growth at the cost of one extra
    n x m matmul.

    Args:
        M: Momentum matrix (d_out, d_in) — plain tensor (not DTensor).
        steps: Number of NS iterations (default 5).
        eps: Normalization epsilon.
    """
    orig_dtype = M.dtype
    transposed = False
    if M.shape[0] > M.shape[1]:
        M = M.T
        transposed = True

    n = M.shape[0]  # n <= m after transpose

    # Normalize and cast to float16 (better precision than bf16 for Gram NS)
    X = M.float()
    X = (X / (X.norm() + eps)).to(torch.float16)

    coeffs = _GRAM_POLAR_EXPRESS_COEFFS_5[:steps]
    restart_at = 3  # 1-indexed; validated for PE 5-step coefficients

    R = X @ X.T  # n x n Gram matrix
    Q = torch.eye(n, dtype=X.dtype, device=X.device)

    for i, (a, b, c) in enumerate(coeffs):
        t = i + 1  # 1-indexed

        # Restart: reapply Q to X, recompute Gram, reset Q
        if t == restart_at:
            X = Q @ X
            R = X @ X.T
            Q = torch.eye(n, dtype=X.dtype, device=X.device)

        # Z = b*R + c*R^2  (avoid explicit I addition for fp16 stability)
        Z = b * R + c * (R @ R)

        # Q <- Q(Z + aI) = Q @ Z + a * Q
        Q = Q @ Z + a * Q

        # R <- (Z + aI) R (Z + aI), expanded without forming (Z + aI):
        #   RZ = R(Z + aI) = R @ Z + a * R
        #   R  = (Z + aI) RZ = Z @ RZ + a * RZ
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ

    # Final application: X_out = Q @ X
    X = Q @ X

    if transposed:
        X = X.T
    return X.to(orig_dtype)


def _newton_schulz_momentum(
    M: torch.Tensor,
    steps: int = 5,
    mode: str = "muon",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Newton-Schulz orthogonalization for a momentum matrix.

    Supports standard fixed-coefficient modes, Polar Express per-iteration
    adaptive coefficients, and Gram Newton-Schulz.

    Args:
        M: Momentum matrix (d_out, d_in) — plain tensor (not DTensor).
        steps: Number of NS iterations (default 5).
        mode: Coefficient mode:
            "muon"               — fixed (3.4445, -4.7750, 2.0315), Muon reference
            "polar_express"      — per-iteration optimal coefficients (arxiv:2505.16932)
            "gram_polar_express" — Gram NS with PE coefficients (Dao & Gu, 2026)
            "convergent"         — fixed (3.0, -16/5, 6/5), converges to sigma=1
            "cubic"              — fixed (1.5, -0.5, 0.0), cubic-only
        eps: Normalization epsilon.
    """
    if mode == "gram_polar_express":
        return _gram_newton_schulz_momentum(M, steps=steps, eps=eps)

    orig_dtype = M.dtype
    transposed = False
    if M.shape[0] > M.shape[1]:
        M = M.T
        transposed = True

    X = M.float()
    X = (X / (X.norm() + eps)).to(torch.bfloat16)

    if mode == "polar_express":
        coeffs = _POLAR_EXPRESS_COEFFS_5[:steps]
        for a, b, c in coeffs:
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))
    else:
        if mode == "muon":
            a, b, c = 3.4445, -4.7750, 2.0315
        elif mode == "convergent":
            a, b, c = 3.0, -16.0 / 5.0, 6.0 / 5.0
        elif mode == "cubic":
            a, b, c = 1.5, -0.5, 0.0
        else:
            raise ValueError(f"Unknown NS mode: {mode!r}")
        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))

    if transposed:
        X = X.T
    return X.to(orig_dtype)


# ======================================================================
#  torchtitan integration
# ======================================================================

_bounded_muon_config: dict = {
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "max_norm": 1.0,
    "project_gradients": True,
    "soft_blend": False,
    "out_norm_dim_0": False,
    "rotation_lr": None,
    "fallback_lr": None,
    "embedding_lr": None,
    "muon_beta1": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_ns_mode": "muon",
    "muon_scale": 0.2,
    "muon_norm_preserve": False,
    "muon_bias_correction": False,
    "muon_geodesic": False,
    "muon_adam_scale": False,
    "muon_flat_scale": False,
}


def configure_bounded_muon(
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    max_norm: float = 1.0,
    project_gradients: bool = True,
    soft_blend: bool = False,
    out_norm_dim_0: bool = False,
    rotation_lr: Optional[float] = None,
    fallback_lr: Optional[float] = None,
    embedding_lr: Optional[float] = None,
    muon_beta1: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
    muon_ns_mode: str = "muon",
    muon_scale: float = 0.2,
    muon_norm_preserve: bool = False,
    muon_bias_correction: bool = False,
    muon_geodesic: bool = False,
    muon_adam_scale: bool = False,
    muon_flat_scale: bool = False,
) -> None:
    """Configure BoundedMuon settings used by make_build_bounded_muon_optimizers().

    Args:
        betas: AdamW beta coefficients for scalars/embeddings.
        eps: AdamW epsilon.
        weight_decay: AdamW weight decay (non-constrained params only).
        max_norm: Maximum row norm (used as target norm for normalization).
        project_gradients: Project gradients/momentum to tangent space before update.
        soft_blend: Smooth blending based on predicted overshoot.
        out_norm_dim_0: Normalize output projections (wo, w2) along dim=0.
        rotation_lr: Separate LR for constrained matrices. None = optimizer.lr.
        fallback_lr: LR for scalars (1D). None = optimizer.lr.
        embedding_lr: LR for embeddings. None = fallback_lr.
        muon_beta1: Momentum coefficient β1 for the Muon momentum buffer.
        muon_nesterov: Use Nesterov momentum (lookahead m_hat = β1*m + (1-β1)*g).
        muon_ns_steps: Newton-Schulz iterations for momentum orthogonalization.
        muon_ns_mode: NS coefficient mode — "muon", "polar_express", "gram_polar_express",
            "convergent", "cubic", or a "dist_*" mode.
            "gram_polar_express" uses Gram Newton-Schulz (Dao & Gu, 2026): operates on the
            n x n Gram matrix instead of the full n x m rectangle, shifting per-iteration
            cost from O(mn^2) to O(n^3). Mathematically equivalent to standard PE but
            25-50% faster for rectangular matrices.
            "dist_*" modes use ZeRO-style parameter ownership with gather/scatter for FSDP2
            scaling: parameters assigned to owner ranks via LPT-greedy scheduling, owner
            gathers full grad, runs NS (inner mode = suffix after "dist_"), scatters result
            shards back. Non-owners never hold the full matrix — ~K× less comm and memory.
            Examples: "dist_muon" (recommended), "dist_gram_polar_express", "dist_polar_express".
            muon_surprise / adam_scale not supported in dist_* modes.
        muon_scale: Moonlight RMS scaling factor (default 0.2). Used when muon_norm_preserve=False.
        muon_norm_preserve: If True, rescale NS(m_hat) to preserve ‖m_hat‖_F then apply
            shape-only normalization (scale = 1/sqrt(max(R,C))). Step magnitude then tracks
            gradient/momentum magnitude rather than being fixed. Default False.
        muon_bias_correction: Scale effective lr by (1 - muon_beta1^t) at step t.
            Momentum-coupled warmup — no extra hyperparameters needed. Duration scales
            automatically with muon_beta1 (β1=0.95 → ~100 steps; β1=0.99 → ~500 steps).
            Default False (disabled, constant Moonlight step from t=1).
        muon_geodesic: Use Riemannian exponential map instead of linear SGD + retraction.
            Per-row rotation by angle θ=lr_eff*‖u_tan_i‖ gives exact on-sphere steps
            and implicit per-row adaptivity. Default False.
        muon_adam_scale: Adaptive lr scaling via EMA of normalized gradient novelty
            ‖g − m_{t-1}‖² / ‖m_{t-1}‖² (computed before momentum update). Reuses
            betas[1] and eps — no new hyperparameters. Dampens lr when surprise > 1
            (direction change); no effect when gradient direction is stable. Default False.

            Motivation: BM's Moonlight step is constant regardless of gradient direction
            history. At training transitions (~500 steps), the dominant gradient singular
            vectors pivot — NS produces a new frame and BM immediately takes the full lr
            step in it, causing activation norm jumps. Adam avoids this via v_t = EMA(g²),
            but NS uniformizes singular values → v_t → const in BSM (the v_t collapse
            problem). This option tracks direction changes directly: surprise is the
            normalized Kalman innovation — large when gradient deviates from its prediction,
            small when direction is stable. The EMA with bias correction gives Bayes-optimal
            dampening for non-stationary gradient processes, without the NS collapse issue
            (signal is on raw g vs m, before NS uniformization).
        muon_flat_scale: Use dimension-independent scaling: scale = muon_scale (no sqrt
            factor). Gives angular step θ ≈ lr * muon_scale per row, independent of matrix
            dimensions. More principled for normalized architectures where the Moonlight
            sqrt(max(R,C)) factor makes angular steps depend on layer width. Typical values
            are larger than Moonlight (e.g. 3–8 instead of 0.2) since the sqrt factor is
            removed. Default False (use Moonlight scaling).
    """
    _bounded_muon_config["betas"] = betas
    _bounded_muon_config["eps"] = eps
    _bounded_muon_config["weight_decay"] = weight_decay
    _bounded_muon_config["max_norm"] = max_norm
    _bounded_muon_config["project_gradients"] = project_gradients
    _bounded_muon_config["soft_blend"] = soft_blend
    _bounded_muon_config["out_norm_dim_0"] = out_norm_dim_0
    _bounded_muon_config["rotation_lr"] = rotation_lr
    _bounded_muon_config["fallback_lr"] = fallback_lr
    _bounded_muon_config["embedding_lr"] = embedding_lr
    _bounded_muon_config["muon_beta1"] = muon_beta1
    _bounded_muon_config["muon_nesterov"] = muon_nesterov
    _bounded_muon_config["muon_ns_steps"] = muon_ns_steps
    _bounded_muon_config["muon_ns_mode"] = muon_ns_mode
    _bounded_muon_config["muon_scale"] = muon_scale
    _bounded_muon_config["muon_norm_preserve"] = muon_norm_preserve
    _bounded_muon_config["muon_bias_correction"] = muon_bias_correction
    _bounded_muon_config["muon_geodesic"] = muon_geodesic
    _bounded_muon_config["muon_adam_scale"] = muon_adam_scale
    _bounded_muon_config["muon_flat_scale"] = muon_flat_scale


def make_build_bounded_muon_optimizers():
    """Factory: returns build_optimizers_fn for BoundedMuon."""

    def build_bounded_muon_optimizers_fn(
        model_parts: list[torch.nn.Module],
        optimizer_config: OptimizerConfig,
        parallel_dims: ParallelDims,
        ft_manager: FTManager,
    ):
        if optimizer_config.early_step_in_backward:
            raise NotImplementedError(
                "BoundedMuon does not support early_step_in_backward."
            )

        fallback_lr = _bounded_muon_config.get("fallback_lr")
        if fallback_lr is None or fallback_lr <= 0:
            fallback_lr = optimizer_config.lr

        embedding_lr = _bounded_muon_config.get("embedding_lr")
        if embedding_lr is None or embedding_lr <= 0:
            embedding_lr = fallback_lr

        return BoundedMuonOptimizerContainer(
            model_parts=model_parts,
            constrained_lr=optimizer_config.lr,
            embedding_lr=embedding_lr,
            scalar_lr=fallback_lr,
            bounded_muon_config=_bounded_muon_config,
        )

    return build_bounded_muon_optimizers_fn


# ======================================================================
#  Main container
# ======================================================================

class BoundedMuonOptimizerContainer(SharedBoundMuonOptimizerContainer):
    """
    True Muon + BSA constraints.

    Extends SharedBoundMuonOptimizerContainer by replacing the Adam step for
    constrained 2D matrix params with a pure Muon step:
      1. Accumulate projected gradient into a full momentum buffer (muon_m)
      2. All-gather momentum → NS on global gram → scale by 0.2*sqrt(max(R,C))
      3. Direct SGD update to local shard; zero p.grad so AdamW skips the param
    Scalars and embeddings continue using fused AdamW.

    Momentum is stored as a plain (replicated) full tensor in optimizer state,
    using the same all-gather infrastructure as BSM (one all-gather per param).

    Step order:
      1. Pre-step: project p.grad to tangent space (inherited _pre_step_projections)
      2. Muon step: EMA(g) → all-gather → NS → SGD update (zeros p.grad after)
      3. AdamW step: scalars + embeddings only (matrix p.grad is None)
      4. Post-step: enforce row-norm constraints (inherited _enforce_constraints)
    """

    def __init__(
        self,
        model_parts: list[torch.nn.Module],
        constrained_lr: float,
        embedding_lr: float,
        scalar_lr: float,
        bounded_muon_config: dict,
    ):
        # Build the BSM config for the parent: pass through the shared fields,
        # but always disable muon_on_gradient (we handle NS on momentum ourselves).
        bsm_config = {
            "betas": bounded_muon_config["betas"],
            "eps": bounded_muon_config["eps"],
            "weight_decay": bounded_muon_config["weight_decay"],
            "mode": "normalized",
            "max_norm": bounded_muon_config["max_norm"],
            "project_gradients": bounded_muon_config["project_gradients"],
            "soft_blend": bounded_muon_config["soft_blend"],
            "out_norm_dim_0": bounded_muon_config.get("out_norm_dim_0", False),
            "rotation_lr": bounded_muon_config.get("rotation_lr"),
            "fallback_lr": bounded_muon_config.get("fallback_lr"),
            "embedding_lr": bounded_muon_config.get("embedding_lr"),
            # partial orthogonal fields (unused, but required by parent __init__)
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
            # Muon-on-gradient: DISABLED — we do NS on momentum instead
            "muon_on_gradient": False,
            "muon_ns_steps": 5,
            "muon_ns_mode": "muon",
        }

        super().__init__(
            model_parts=model_parts,
            constrained_lr=constrained_lr,
            embedding_lr=embedding_lr,
            scalar_lr=scalar_lr,
            bounded_spherical_config=bsm_config,
        )

        # Muon-on-momentum config
        self._muon_beta1 = bounded_muon_config["muon_beta1"]
        self._muon_nesterov = bounded_muon_config["muon_nesterov"]
        self._muon_ns_steps = bounded_muon_config["muon_ns_steps"]
        self._muon_ns_mode = bounded_muon_config["muon_ns_mode"]
        self._muon_scale = bounded_muon_config["muon_scale"]
        self._muon_norm_preserve = bounded_muon_config.get("muon_norm_preserve", False)
        self._muon_bias_correction = bounded_muon_config.get("muon_bias_correction", False)
        self._muon_geodesic = bounded_muon_config.get("muon_geodesic", False)
        self._muon_adam_scale = bounded_muon_config.get("muon_adam_scale", False)
        self._muon_flat_scale = bounded_muon_config.get("muon_flat_scale", False)

        # Cached LPT ownership assignment for dist_gram mode.
        # Keyed by optimizer index; populated lazily on first step.
        self._dist_gram_ownership: dict[int, dict[int, int]] = {}

        # Build eligible-param lists for the Muon step.
        # For mode="none" (pure Muon, no constraints) all constrained lists are
        # empty, so we fall back to scanning the model's named_parameters directly.
        self._muon_eligible_per_optimizer: list[list[torch.nn.Parameter]] = []
        for model_part, opt, norm_ps, bound_ps, po_ps, pob_ps in zip(
            self.model_parts if hasattr(self, "model_parts") else [None] * len(self.optimizers),
            self.optimizers,
            self._normalized_params_per_optimizer,
            self._bounded_params_per_optimizer,
            self._partial_ortho_params_per_optimizer,
            self._partial_ortho_bounded_params_per_optimizer,
        ):
            constrained_eligible = [
                p for p in itertools.chain(norm_ps, bound_ps, po_ps, pob_ps)
                if p.dim() == 2 and p.shape[0] > 1 and p.shape[1] > 1
            ]
            if constrained_eligible or self._mode not in ("none",):
                eligible = constrained_eligible
            else:
                # mode=none (pure Muon): apply Muon to all 2D weight matrices
                # that are not embeddings or the output head (tok_embeddings / output).
                # Use named_parameters to filter by name, same as BSM's param splitting.
                all_opt_param_ids = {
                    id(p) for pg in opt.param_groups for p in pg["params"]
                }
                eligible = []
                if model_part is not None:
                    for name, p in model_part.named_parameters():
                        if (id(p) in all_opt_param_ids
                                and p.dim() == 2
                                and p.shape[0] > 1
                                and p.shape[1] > 1
                                and "tok_embeddings" not in name
                                and "output" not in name):
                            eligible.append(p)
                else:
                    # Fallback: scan all param groups, skip 1D params
                    for pg in opt.param_groups:
                        for p in pg["params"]:
                            if p.dim() == 2 and p.shape[0] > 1 and p.shape[1] > 1:
                                eligible.append(p)
            self._muon_eligible_per_optimizer.append(eligible)

        from torchtitan.tools.logging import logger
        logger.info(
            f"BoundedMuonOptimizerContainer created:\n"
            f"  Mode: {self._mode}, max_norm={self._max_norm}\n"
            f"  Muon: beta1={self._muon_beta1}, nesterov={self._muon_nesterov}, "
            f"ns_steps={self._muon_ns_steps}, ns_mode={self._muon_ns_mode}, "
            f"scale={self._muon_scale}, norm_preserve={self._muon_norm_preserve}, "
            f"flat_scale={self._muon_flat_scale}\n"
            f"  Eligible matrix params: "
            f"{sum(len(ps) for ps in self._muon_eligible_per_optimizer)}"
        )

    def step(self, *args, **kwargs):
        """Tangent projection → Muon momentum step → AdamW (scalars) → constraint."""
        # 1. Project p.grad to tangent space (inherited)
        if self._project_gradients:
            self._pre_step_projections()

        # 2. Muon step: accumulate momentum, NS, SGD update, zero p.grad
        self._apply_bounded_muon_step()

        # 3. AdamW for scalars/embeddings only
        #    Matrix params have p.grad=None after step 2 — fused AdamW skips them.
        #    Call OptimizersContainer.step() directly to bypass BSM's step() logic.
        OptimizersContainer.step(self, *args, **kwargs)

        # 4. Post-step constraint enforcement (inherited)
        self._step_count += 1
        self._enforce_constraints()

    @torch.no_grad()
    def _apply_bounded_muon_step(self):
        """Muon step for all constrained 2D matrix params.

        For each eligible param:
          1. All-gather p.grad to get the full projected gradient
          2. Update (or init) the full momentum buffer muon_m
          3. Compute Nesterov lookahead m_hat (or use m directly)
          4. NS on m_hat using the global gram (correct, no per-iter all-reduces)
          5. Moonlight scaling: scale = muon_scale * sqrt(max(R, C))
          6. Apply SGD update to the local shard of p.data
          7. Zero p.grad so the subsequent AdamW.step() skips this param
        """
        beta1 = self._muon_beta1
        ns_steps = self._muon_ns_steps
        ns_mode = self._muon_ns_mode
        muon_scale = self._muon_scale
        norm_preserve = self._muon_norm_preserve
        nesterov = self._muon_nesterov
        bias_correction = self._muon_bias_correction
        geodesic = self._muon_geodesic
        adam_scale = self._muon_adam_scale
        flat_scale = self._muon_flat_scale
        _gl = self._get_local

        # Momentum-coupled warmup: scale lr by (1 - β1^t).
        # Naturally tied to momentum timescale — β1=0.95 → ~100 steps, β1=0.99 → ~500.
        t = self._step_count + 1  # 1-indexed
        warmup_factor = (1.0 - beta1 ** t) if bias_correction else 1.0

        for opt_idx, (opt, eligible) in enumerate(
            zip(self.optimizers, self._muon_eligible_per_optimizer)
        ):
            if not eligible:
                continue

            # Build param→lr mapping (LR may change with scheduler)
            param_lr: dict[int, float] = {}
            for pg in opt.param_groups:
                lr_val = pg["lr"]
                for pp in pg["params"]:
                    param_lr[id(pp)] = lr_val

            # dist_*: build LPT ownership map lazily (once per eligible list).
            # Activated for any mode starting with "dist_" (e.g. "dist_muon",
            # "dist_gram_polar_express", "dist_polar_express").
            if ns_mode.startswith("dist_") and opt_idx not in self._dist_gram_ownership:
                _pg0 = self._get_fsdp_process_group(eligible[0]) if eligible else None
                _ws = dist.get_world_size(_pg0) if _pg0 is not None else 1
                self._dist_gram_ownership[opt_idx] = _compute_dist_gram_ownership(
                    eligible, _ws
                )

            for p in eligible:
                if p.grad is None:
                    continue

                # ── dist_* branch ─────────────────────────────────────────────────
                # ZeRO-style parameter ownership with gather/scatter.
                # Owner rank: gather full grad, update momentum, run NS, scatter shards.
                # Non-owner ranks: send grad shard, receive result shard — never hold
                # the full matrix. inner NS mode = ns_mode.removeprefix("dist_").
                if ns_mode.startswith("dist_"):
                    fsdp_pg = self._get_fsdp_process_group(p)
                    # "dist_gram" is a legacy alias for "dist_gram_polar_express".
                    _ns_mode = "dist_gram_polar_express" if ns_mode == "dist_gram" else ns_mode
                    inner_mode = _ns_mode.removeprefix("dist_")

                    # Local grad shard — no all_gather.
                    local_grad = _gl(p.grad).contiguous()  # (local_size, d_in)
                    local_size = local_grad.shape[0]
                    d_out, d_in = p.data.shape  # global shape from DTensor / plain tensor

                    if fsdp_pg is not None:
                        local_rank = dist.get_rank(fsdp_pg)
                        world_size = dist.get_world_size(fsdp_pg)
                    else:
                        local_rank = 0
                        world_size = 1

                    owner_rank = self._dist_gram_ownership[opt_idx].get(id(p), 0)
                    is_owner = (local_rank == owner_rank)
                    chunk_size = math.ceil(d_out / world_size)

                    # ── Phase 1: gather grad to owner ──────────────────────────────
                    if fsdp_pg is not None:
                        padded_grad = _pad_rows(local_grad, chunk_size)
                        if is_owner:
                            gather_list = [
                                torch.empty(
                                    chunk_size, d_in,
                                    dtype=local_grad.dtype, device=local_grad.device
                                )
                                for _ in range(world_size)
                            ]
                            dist.gather(
                                padded_grad,
                                gather_list=gather_list,
                                dst=owner_rank,
                                group=fsdp_pg,
                            )
                            full_grad = torch.cat(gather_list, dim=0)[:d_out]
                        else:
                            dist.gather(
                                padded_grad,
                                gather_list=None,
                                dst=owner_rank,
                                group=fsdp_pg,
                            )
                    else:
                        # Plain tensor (single rank): no gather needed.
                        full_grad = local_grad

                    # ── NS on owner only ────────────────────────────────────────────
                    state = opt.state.setdefault(p, {})
                    if is_owner:
                        if "muon_m" not in state:
                            state["muon_m"] = torch.zeros_like(full_grad)
                        m = state["muon_m"]
                        m.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
                        if nesterov:
                            m_hat = beta1 * m + (1.0 - beta1) * full_grad
                        else:
                            m_hat = m
                        m_ortho = _newton_schulz_momentum(
                            m_hat, steps=ns_steps, mode=inner_mode
                        )
                        # norm_preserve rescaling done on owner so scatter carries
                        # the correctly-scaled shards.
                        if norm_preserve:
                            m_hat_norm = m_hat.float().norm().clamp(min=1e-8)
                            m_o_f32 = m_ortho.float()
                            m_ortho = (
                                m_o_f32 * (m_hat_norm / m_o_f32.norm().clamp(min=1e-8))
                            ).to(m_hat.dtype)

                    # ── Phase 2: scatter m_ortho shards ────────────────────────────
                    if fsdp_pg is not None:
                        if is_owner:
                            m_ortho_c = m_ortho.contiguous()
                            scatter_list = [
                                _pad_rows(
                                    m_ortho_c[r * chunk_size : r * chunk_size + min(chunk_size, d_out - r * chunk_size)],
                                    chunk_size,
                                )
                                for r in range(world_size)
                            ]
                        else:
                            scatter_list = None
                        recv_buf = torch.empty(
                            chunk_size, d_in,
                            dtype=local_grad.dtype, device=local_grad.device
                        )
                        dist.scatter(
                            recv_buf,
                            scatter_list=scatter_list,
                            src=owner_rank,
                            group=fsdp_pg,
                        )
                        m_ortho_local = recv_buf[:local_size]
                    else:
                        # Plain tensor: m_ortho is already the full (= local) result.
                        m_ortho_local = m_ortho

                    # ── Scale + apply ───────────────────────────────────────────────
                    if norm_preserve:
                        scale = 1.0 / math.sqrt(max(d_out, d_in))
                    elif flat_scale:
                        scale = muon_scale
                    else:
                        scale = muon_scale * math.sqrt(max(d_out, d_in))

                    lr = param_lr.get(id(p), opt.defaults["lr"])
                    lr_eff = lr * warmup_factor

                    _gl(p.data).add_(m_ortho_local * scale, alpha=-lr_eff)
                    p.grad = None
                    continue
                # ── end dist_* ────────────────────────────────────────────────────

                # All-gather full gradient once (1 comm op, same as BSM)
                full_grad, shard_info = self._all_gather_grad(p)
                orig_shape = full_grad.shape   # (d_out, d_in) — full matrix

                # Init or fetch full momentum buffer
                state = opt.state.setdefault(p, {})
                if "muon_m" not in state:
                    state["muon_m"] = torch.zeros_like(full_grad)
                m = state["muon_m"]

                # Gradient novelty: ‖g − m_{t-1}‖² / ‖m_{t-1}‖²
                # Computed BEFORE momentum update (m is still m_{t-1}).
                # Logged always (cheap scalar). Used for adaptive lr when adam_scale=True.
                beta2 = opt.defaults["betas"][1]
                g_f = full_grad.float()
                m_f = m.float()
                m_norm_sq = (m_f.norm() ** 2).add(self._eps)
                surprise_val = ((g_f - m_f).norm() ** 2).div(m_norm_sq).clamp(min=0.0).item()
                state["muon_surprise"] = surprise_val

                # Momentum update (on full matrix, replicated on all ranks)
                m.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)

                # Nesterov lookahead: m_hat = β1*m + (1-β1)*g
                if nesterov:
                    m_hat = beta1 * m + (1.0 - beta1) * full_grad
                else:
                    m_hat = m

                # NS on full global momentum (global gram — correct, no sharding artifacts)
                m_ortho = _newton_schulz_momentum(m_hat, steps=ns_steps, mode=ns_mode)

                R, C = orig_shape
                if norm_preserve:
                    # Option 2: rescale NS output to preserve ‖m_hat‖_F, then apply
                    # shape-only normalization.  Step magnitude tracks momentum magnitude
                    # (decays as training converges), giving adaptive scaling like BSM.
                    m_hat_norm = m_hat.float().norm().clamp(min=1e-8)
                    m_ortho_f32 = m_ortho.float()
                    m_ortho_norm = m_ortho_f32.norm().clamp(min=1e-8)
                    m_ortho = (m_ortho_f32 * (m_hat_norm / m_ortho_norm)).to(m_hat.dtype)
                    scale = 1.0 / math.sqrt(max(R, C))
                elif flat_scale:
                    # Option 3: dimension-independent angular scaling.
                    # Per-row angular step θ ≈ lr * muon_scale, independent of matrix
                    # dimensions.  More principled for normalized architectures where
                    # rows live on the unit sphere and optimal angular step shouldn't
                    # depend on layer width.  Typical muon_scale values: 3–8.
                    scale = muon_scale
                else:
                    # Option 1 (default): Moonlight fixed RMS scaling.
                    # Per-element update ≈ lr * muon_scale regardless of gradient magnitude.
                    scale = muon_scale * math.sqrt(max(R, C))

                # Effective LR: apply momentum-coupled warmup factor
                lr = param_lr.get(id(p), opt.defaults["lr"])
                lr_eff = lr * warmup_factor

                # Adaptive lr: dampen by EMA of gradient novelty when adam_scale=True.
                # lr_eff shrinks when surprise > 1 (direction change); no amplification.
                if adam_scale:
                    if "muon_v" not in state:
                        state["muon_v"] = torch.tensor(
                            surprise_val, dtype=torch.float32, device=p.device
                        )
                    else:
                        state["muon_v"].mul_(beta2).add_(surprise_val * (1.0 - beta2))
                    v_hat = state["muon_v"].item() / max(1.0 - beta2 ** t, 1e-8)
                    lr_eff = lr_eff / max(math.sqrt(max(v_hat, 0.0)), 1.0)

                if geodesic and self._mode == "normalized":
                    # Riemannian exponential map on the unit sphere.
                    # For each row w_i (unit norm) and update u_i = m_ortho_i * scale:
                    #   u_tan_i = u_i - (u_i·w_i)*w_i   (tangential component)
                    #   θ_i     = lr_eff * ‖u_tan_i‖    (rotation angle)
                    #   w_new_i = cos(θ_i)*w_i + sin(θ_i)*(u_tan_i/‖u_tan_i‖)
                    # Gives exact on-sphere steps; per-row θ provides implicit adaptivity.
                    full_w, w_shard_info = self._get_full_weight(p)
                    u = -m_ortho.float() * (scale * lr_eff)  # negative: gradient descent
                    w_f = full_w.float()
                    dot = (u * w_f).sum(dim=-1, keepdim=True)
                    u_tan = u - dot * w_f
                    theta = u_tan.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    w_new = torch.cos(theta) * w_f + torch.sin(theta) * (u_tan / theta)
                    if w_shard_info is None:
                        _gl(p.data).copy_(w_new.to(p.data.dtype))
                    else:
                        _, offset, local_size = w_shard_info
                        _gl(p.data).copy_(
                            w_new[offset : offset + local_size].to(p.data.dtype)
                        )
                else:
                    # Standard linear SGD update to local shard
                    if shard_info is None:
                        _gl(p.data).add_(m_ortho * scale, alpha=-lr_eff)
                    else:
                        _, offset, local_size = shard_info
                        _gl(p.data).add_(
                            m_ortho[offset : offset + local_size] * scale, alpha=-lr_eff
                        )

                # Zero gradient: fused AdamW.step() will skip this param
                p.grad = None
