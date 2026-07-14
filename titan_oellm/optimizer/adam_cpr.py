"""AdamCPR — Constrained Parameter Regularization (Franke et al., arXiv:2311.09058)
in the torchtitan OptimizersContainer pattern.

This wraps the *reference* AdamCPR implementation vendored from
https://github.com/automl/CPR (``_adamcpr_upstream.py`` +
``_cpr_group_parameter.py``) so it is selectable via ``[optimizer].name = "adam_cpr"``.

CPR replaces weight decay with a hard constraint on a per-parameter regularization
statistic (e.g. squared L2 norm) enforced by an augmented-Lagrangian update:
each regularized weight tensor carries a Lagrange multiplier λ that accumulates
the constraint violation ``stat(w) - κ`` (clipped ≥ 0), and the weight is pulled
back by ``-2·λ·w``. The upper bound κ is set automatically (``inflection_point``,
the default), after a warmup (``warm_start``), as a multiple of the initial
statistic (``dependent``), or to a fixed value (``uniform``).

Contract (mirrors the other titan_oellm optimizers):
  _adam_cpr_config / configure_adam_cpr / make_build_adam_cpr_optimizers /
  AdamCPROptimizerContainer.
"""

import torch

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.ft import FTManager
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims

from titan_oellm.optimizer._adamcpr_upstream import AdamCPR

# Module-level config, populated by configure_adam_cpr() during model init and
# read by the build closure at optimizer-construction time.
_adam_cpr_config: dict = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "kappa_init_method": "inflection_point",
    "kappa_init_param": 1000.0,
    "reg_function": "l2",
    "kappa_update": 1.0,
    "reg_step_size": 200,
    "reg_ema_decay": 0.99,
    "reg_embedding": False,
    "reg_by_lr": False,
    "amsgrad": False,
}


def configure_adam_cpr(
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    kappa_init_method: str = "inflection_point",
    kappa_init_param: float = 1000.0,
    reg_function: str = "l2",
    kappa_update: float = 1.0,
    reg_step_size: int = 200,
    reg_ema_decay: float = 0.99,
    reg_embedding: bool = False,
    reg_by_lr: bool = False,
    amsgrad: bool = False,
) -> None:
    """Set AdamCPR hyperparameters used by make_build_adam_cpr_optimizers().

    Args:
        betas, eps, amsgrad: standard Adam parameters.
        kappa_init_method: how the constraint upper bound κ is initialized —
            "inflection_point" (default, automatic: watch an EMA of the
            statistic and lock κ when its rate of change peaks),
            "warm_start" (κ = statistic measured at step ``kappa_init_param``),
            "dependent" (κ = ``kappa_init_param`` × initial statistic),
            "uniform" (κ = ``kappa_init_param``, fixed).
        kappa_init_param: meaning depends on kappa_init_method (warmup steps /
            scale / fixed bound).
        reg_function: regularization statistic — "l2" (default), "l1", "std", "huber".
        kappa_update: Lagrange-multiplier step size μ (λ ← max(0, λ + μ·(stat-κ))).
        reg_step_size: sampling cadence for inflection-point detection.
        reg_ema_decay: EMA decay for the statistic in inflection-point mode.
        reg_embedding: also regularize embedding weights (default False).
        reg_by_lr: scale the constraint pullback by the current lr.
    """
    _adam_cpr_config.update(
        betas=betas,
        eps=eps,
        kappa_init_method=kappa_init_method,
        kappa_init_param=kappa_init_param,
        reg_function=reg_function,
        kappa_update=kappa_update,
        reg_step_size=reg_step_size,
        reg_ema_decay=reg_ema_decay,
        reg_embedding=reg_embedding,
        reg_by_lr=reg_by_lr,
        amsgrad=amsgrad,
    )


def make_build_adam_cpr_optimizers():
    """Return a build_optimizers_fn closure that constructs AdamCPR.

    Uses the module-level config set via configure_adam_cpr().
    """

    def build_adam_cpr_optimizers_fn(
        model_parts: list[torch.nn.Module],
        optimizer_config: OptimizerConfig,
        parallel_dims: ParallelDims,
        ft_manager: FTManager,
    ):
        if getattr(optimizer_config, "early_step_in_backward", False):
            raise NotImplementedError("AdamCPR does not support early_step_in_backward.")
        return AdamCPROptimizerContainer(
            model_parts=model_parts,
            lr=optimizer_config.lr,
            cpr_config=_adam_cpr_config,
        )

    return build_adam_cpr_optimizers_fn


class AdamCPROptimizerContainer(OptimizersContainer):
    """One reference-AdamCPR optimizer per model part.

    AdamCPR is a full ``torch.optim.Optimizer`` that groups the model's params
    itself (regularizing Linear weights; excluding biases / norms / embeddings
    unless ``reg_embedding``) and tracks its own step counter for κ scheduling —
    so no external step feeding is needed. Compatible with torchtitan's LR
    scheduler / checkpoint / pipeline patterns via ``_post_init``.
    """

    def __init__(
        self,
        model_parts: list[torch.nn.Module],
        lr: float,
        cpr_config: dict,
    ):
        from torchtitan.tools.logging import logger

        self.model_parts = model_parts
        self.optimizers: list[AdamCPR] = []
        all_params: list[torch.nn.Parameter] = []

        for model in model_parts:
            optimizer = AdamCPR(
                model,  # AdamCPR groups params internally (regularize vs. not)
                lr=lr,
                betas=cpr_config["betas"],
                eps=cpr_config["eps"],
                kappa_init_method=cpr_config["kappa_init_method"],
                kappa_init_param=cpr_config["kappa_init_param"],
                reg_function=cpr_config["reg_function"],
                kappa_update=cpr_config["kappa_update"],
                reg_step_size=cpr_config["reg_step_size"],
                reg_ema_decay=cpr_config["reg_ema_decay"],
                reg_embedding=cpr_config["reg_embedding"],
                reg_by_lr=cpr_config["reg_by_lr"],
                amsgrad=cpr_config["amsgrad"],
            )
            self.optimizers.append(optimizer)
            for group in optimizer.param_groups:
                all_params.extend(group["params"])

        self._validate_length(len(self.model_parts))
        self._post_init(all_params, {"lr": lr})

        n_reg = sum(
            len(g["params"]) for opt in self.optimizers for g in opt.param_groups
            if g.get("regularize", False)
        )
        n_total = len(all_params)
        logger.info(
            "AdamCPROptimizerContainer created:\n"
            f"  Model parts: {len(model_parts)}\n"
            f"  kappa_init_method: {cpr_config['kappa_init_method']} "
            f"(param={cpr_config['kappa_init_param']})\n"
            f"  reg_function: {cpr_config['reg_function']}, "
            f"kappa_update={cpr_config['kappa_update']}, reg_by_lr={cpr_config['reg_by_lr']}\n"
            f"  Regularized params: {n_reg}/{n_total} "
            f"(embeddings {'regularized' if cpr_config['reg_embedding'] else 'excluded'})"
        )
