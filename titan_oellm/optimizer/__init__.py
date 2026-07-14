"""Custom optimizers for titan-oellm.

Ported from titan-sci. Provides constrained / Muon optimizers that follow the
torchtitan ``OptimizersContainer`` pattern. Each optimizer exposes:

- ``_<name>_config``            module-level defaults dict
- ``configure_<name>(cfg)``     mutate the defaults from a job-config section
- ``make_build_<name>_optimizers()``  → ``build_optimizers_fn`` closure
- ``<Name>OptimizerContainer``  the OptimizersContainer subclass

Selection is done by ``[optimizer].name`` in the job config; the qwen3 train
spec dispatches ``name.lower()`` to the matching ``make_build_*`` factory.
"""

from titan_oellm.optimizer.constrained_adam import (
    ConstrainedAdam,
    ConstrainedAdamWithFallback,
    ConstrainedAdamOptimizerContainer,
    configure_constrained_adam,
    make_build_constrained_adam_optimizers,
    _constrained_adam_config,
)

from titan_oellm.optimizer.adam_cpr import (
    AdamCPROptimizerContainer,
    configure_adam_cpr,
    make_build_adam_cpr_optimizers,
    _adam_cpr_config,
)

from titan_oellm.optimizer.shared_bound_adam import (
    BoundedSphericalAdamOptimizerContainer,
    configure_bounded_spherical_adam,
    make_build_bounded_spherical_adam_optimizers,
    _bounded_spherical_adam_config,
)

from titan_oellm.optimizer.shared_bound_muon import (
    SharedBoundMuonOptimizerContainer,
    configure_shared_bound_muon,
    make_build_shared_bound_muon_optimizers,
    _shared_bound_muon_config,
)

from titan_oellm.optimizer.bounded_muon import (
    BoundedMuonOptimizerContainer,
    configure_bounded_muon,
    make_build_bounded_muon_optimizers,
    _bounded_muon_config,
)

__all__ = [
    # AdamCPR — reference CPR (github.com/automl/CPR, arXiv:2311.09058)
    "AdamCPROptimizerContainer",
    "configure_adam_cpr",
    "make_build_adam_cpr_optimizers",
    "_adam_cpr_config",
    # ConstrainedAdam (CPR-style, arXiv:2311.09058)
    "ConstrainedAdam",
    "ConstrainedAdamWithFallback",
    "ConstrainedAdamOptimizerContainer",
    "configure_constrained_adam",
    "make_build_constrained_adam_optimizers",
    "_constrained_adam_config",
    # BoundedSphericalAdam (BSA — bounded/normalized/partial-orthogonal Adam)
    "BoundedSphericalAdamOptimizerContainer",
    "configure_bounded_spherical_adam",
    "make_build_bounded_spherical_adam_optimizers",
    "_bounded_spherical_adam_config",
    # SharedBoundMuon (BSA + Muon gradient-direction orthogonalization)
    "SharedBoundMuonOptimizerContainer",
    "configure_shared_bound_muon",
    "make_build_shared_bound_muon_optimizers",
    "_shared_bound_muon_config",
    # BoundedMuon (BSA projection + true Muon: NS on momentum, pure SGD)
    "BoundedMuonOptimizerContainer",
    "configure_bounded_muon",
    "make_build_bounded_muon_optimizers",
    "_bounded_muon_config",
]
