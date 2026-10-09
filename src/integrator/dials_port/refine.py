"""The outer refinement loop: scale, error model, outlier rejection.

Mirrors `scaler.py`'s cycle structure. One cycle is

1.  minimise the scaling target over the model parameters, with the
    error model and the outlier set held fixed;
2.  reject outliers against the newly refined scales;
3.  refine the error model against the same scales, on the outlier-free
    subset, and feed the corrected variances back into the target.

Each step conditions on the output of the others, so the loop is a
block-coordinate descent on a moving objective rather than a descent on
one fixed functional. Nothing guarantees monotonicity across cycles;
`history` records every cycle so the trajectory is inspectable rather
than merely asserted.

Choice of minimiser
-------------------
`torch.optim.LBFGS` in float64, not `scipy.optimize.least_squares`.
The point of `target.py` is that the likelihood is swappable, and a
general likelihood is not a sum of squares -- `least_squares` would need
a residual vector and a Jacobian that only exist for the Gaussian case,
which is exactly the assumption being relaxed. LBFGS needs only a scalar
and its gradient, both of which autograd supplies for any likelihood.
The cost is that DIALS' Gauss-Newton and Levenberg-Marquardt engines
converge in fewer steps on the Gaussian target.

Convergence uses DIALS' criterion (`scaling_refiner.py:172-186`): stop
when the *relative* change in RMSD between successive steps falls below
`rmsd_tolerance`, default `1e-4`.

The gauge degeneracy
--------------------
The objective is invariant under `g -> c g`, `Ih -> Ih / c` for any
constant `c`: the residual `I - g Ih` and the profiled-out weighted mean
both transform so that nothing changes. The overall scale is therefore
not determined by the data, and a model carrying a redundant global
factor has an exactly flat direction. DIALS breaks it by fixing the
first scale parameter (`ScalingModelBase.fix_initial_parameter`, invoked
from `algorithm.py:616`); a model used here should do the same, either
by holding a parameter fixed or by normalising its output.

Left unbroken, a quasi-Newton minimiser will drift along the flat
direction until the scales overflow. `refine_scale` guards against that
-- a step producing a non-finite loss or parameter is rolled back and
the minimisation stops -- but the guard only contains the symptom.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from .error_model import (
    BasicErrorModel,
    ErrorModelFilter,
    ErrorModelResult,
    Prefactor,
    refine_error_model,
)
from .outliers import reject_outliers
from .target import ScalingTarget
from .types import AsuMap, Reflections


class ScaleModel(Protocol):
    """A parameterised inverse scale factor.

    Any `torch.nn.Module` whose `forward` maps reflections to a `(N,)`
    inverse scale factor satisfies this.
    """

    def __call__(self, refl: Reflections) -> torch.Tensor:
        """Return `(N,)` inverse scale factors."""
        ...

    def parameters(self) -> Iterable[torch.Tensor]:
        """Return the tensors to refine."""
        ...


@dataclass
class ScaleStep:
    """One LBFGS step of the inner scale minimisation.

    Attributes:
        step: step index within the cycle.
        rmsd: `sqrt(L/N)` after the step.
        loss: the target value after the step.
    """

    step: int
    rmsd: float
    loss: float


@dataclass
class RefinementCycle:
    """One outer cycle.

    Attributes:
        cycle: cycle index.
        n_refl: reflections entering the scale minimisation.
        rmsd: RMSD at the end of the scale minimisation.
        steps: the inner minimisation trajectory.
        scale_converged: whether the inner loop met `rmsd_tolerance`.
        n_outliers: cumulative outliers after this cycle.
        error_model: `(a, b)` after this cycle, if refined.
        error_model_failed: whether the error model hit its fallback.
    """

    cycle: int
    n_refl: int
    rmsd: float
    steps: list[ScaleStep] = field(default_factory=list)
    scale_converged: bool = False
    n_outliers: int = 0
    error_model: tuple[float, float] | None = None
    error_model_failed: bool = False


@dataclass
class RefinementResult:
    """Outcome of `refine`.

    Attributes:
        scale: `(N,)` refined inverse scale factors for all input rows.
        outliers: `(N,)` boolean mask of rejected reflections.
        error_model: the refined error model, or `None`.
        history: one entry per outer cycle.
        converged: whether the outer RMSD sequence met its tolerance.
    """

    scale: torch.Tensor
    outliers: torch.Tensor
    error_model: BasicErrorModel | None
    history: list[RefinementCycle]
    converged: bool

    @property
    def rmsd(self) -> float:
        """RMSD of the final cycle."""
        return self.history[-1].rmsd if self.history else float("nan")


def _relative_change(previous: float, current: float) -> float:
    """Relative change used by DIALS' `test_rmsd_convergence`."""
    if previous > 0:
        return abs((previous - current) / previous)
    return 0.0


def refine_scale(
    refl: Reflections,
    amap: AsuMap,
    model: ScaleModel,
    target: ScalingTarget,
    max_steps: int = 50,
    rmsd_tolerance: float = 1e-4,
    lr: float = 1.0,
    history_size: int = 10,
) -> tuple[list[ScaleStep], bool]:
    """Minimise the target over the model parameters.

    LBFGS is driven one iteration at a time so that DIALS' relative-RMSD
    criterion can be tested between steps; the optimiser's curvature
    history persists across calls, so this is a single LBFGS run and not
    a sequence of restarts.

    Args:
        refl: observations to minimise against.
        amap: symmetry map for the same rows.
        model: the scale parameterisation.
        target: the objective.
        max_steps: cap on LBFGS iterations.
        rmsd_tolerance: relative RMSD change defining convergence.
        lr: LBFGS step size.
        history_size: LBFGS curvature history length.

    Returns:
        `(steps, converged)`.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        rmsd = float(target.rmsd(refl, amap, model(refl)))
        return [ScaleStep(0, rmsd, rmsd**2 * len(refl))], True

    opt = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=1,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = target(refl, amap, model(refl))
        loss.backward()
        return loss

    steps: list[ScaleStep] = []
    converged = False
    for step in range(max_steps):
        snapshot = [p.detach().clone() for p in params]
        loss = float(opt.step(closure).detach())
        with torch.no_grad():
            rmsd = float(target.rmsd(refl, amap, model(refl)))

        finite = np.isfinite(loss) and np.isfinite(rmsd)
        finite &= all(bool(torch.isfinite(p).all()) for p in params)
        if not finite:
            # The unbroken gauge lets LBFGS drift until the scales
            # overflow; roll back rather than propagate a NaN scale.
            with torch.no_grad():
                for p, saved in zip(params, snapshot, strict=True):
                    p.copy_(saved)
            break

        steps.append(ScaleStep(step, rmsd, loss))
        if len(steps) > 1:
            if _relative_change(steps[-2].rmsd, rmsd) < rmsd_tolerance:
                converged = True
                break
    if not steps:
        with torch.no_grad():
            rmsd = float(target.rmsd(refl, amap, model(refl)))
        steps.append(ScaleStep(0, rmsd, rmsd**2 * len(refl)))
    return steps, converged


def refine(
    refl: Reflections,
    amap: AsuMap,
    model: ScaleModel,
    target: ScalingTarget,
    n_cycles: int = 3,
    rmsd_tolerance: float = 1e-4,
    max_steps: int = 50,
    outlier_rejection: bool = True,
    zmax: float = 6.0,
    zmax_lower: float | None = None,
    error_model: bool | BasicErrorModel = True,
    error_model_filters: ErrorModelFilter | None = None,
    error_model_anomalous: bool = True,
    error_model_prefactor: Prefactor = "dials",
    lr: float = 1.0,
) -> RefinementResult:
    """Alternate scale refinement, outlier rejection and error modelling.

    Outliers are excluded from the scale minimisation and from the error
    model fit but are still assigned a scale on output, as DIALS does
    when it expands scales to all reflections.

    Args:
        refl: unmerged observations. `variance` must be the
            *uncorrected* reported variance; the error model is applied
            on top of it each cycle.
        amap: symmetry map for the same rows.
        model: the scale parameterisation.
        target: the objective, including the choice of likelihood.
        n_cycles: number of outer cycles.
        rmsd_tolerance: relative RMSD change defining convergence, both
            within a cycle and between cycles.
        max_steps: cap on LBFGS iterations per cycle.
        outlier_rejection: run outlier rejection each cycle.
        zmax: outlier threshold.
        zmax_lower: lower-tail outlier threshold; see
            `outliers.reject_outliers`.
        error_model: `True` to refine one from scratch, an existing
            `BasicErrorModel` to start from, or `False` to skip.
        error_model_filters: filter thresholds for the error model.
        error_model_anomalous: group Friedel mates separately when
            fitting the error model, as DIALS does.
        error_model_prefactor: multiplicity convention for the
            normalised deviations; see
            `error_model.deviation_prefactor`.
        lr: LBFGS step size.

    Returns:
        The refinement outcome, with per-cycle history.
    """
    n = len(refl)
    outliers = torch.zeros(n, dtype=torch.bool)
    em: BasicErrorModel | None = (
        error_model if isinstance(error_model, BasicErrorModel) else None
    )
    fit_error_model = error_model is not False

    history: list[RefinementCycle] = []
    converged = False
    raw_variance = refl.variance

    for cycle in range(n_cycles):
        working = refl if not bool(outliers.any()) else refl.select(~outliers)
        working_amap = _select_amap(amap, ~outliers)
        if em is not None:
            working = _with_variance(
                working, em.apply(working.variance, working.intensity)
            )

        steps, step_converged = refine_scale(
            working,
            working_amap,
            model,
            target,
            max_steps=max_steps,
            rmsd_tolerance=rmsd_tolerance,
            lr=lr,
        )
        entry = RefinementCycle(
            cycle=cycle,
            n_refl=len(working),
            rmsd=steps[-1].rmsd if steps else float("nan"),
            steps=steps,
            scale_converged=step_converged,
        )

        with torch.no_grad():
            scale = model(refl).detach()

        if outlier_rejection:
            corrected = (
                refl
                if em is None
                else _with_variance(
                    refl, em.apply(raw_variance, refl.intensity)
                )
            )
            outliers = reject_outliers(
                corrected,
                amap,
                scale=scale,
                zmax=zmax,
                zmax_lower=zmax_lower,
            )
            entry.n_outliers = int(outliers.sum())

        if fit_error_model:
            clean = refl.select(~outliers)
            try:
                result: ErrorModelResult = refine_error_model(
                    clean,
                    _select_amap(amap, ~outliers),
                    scale[~outliers],
                    initial=em,
                    filters=error_model_filters,
                    anomalous=error_model_anomalous,
                    prefactor=error_model_prefactor,
                )
            except ValueError:
                # Too few suitable reflections. DIALS logs and keeps the
                # previous model (`scaler.py:391-395`) rather than
                # aborting the whole refinement.
                entry.error_model_failed = True
            else:
                em = result.model
                entry.error_model = (em.a, em.b)
                entry.error_model_failed = result.failed

        history.append(entry)
        if len(history) > 1:
            change = _relative_change(history[-2].rmsd, entry.rmsd)
            if change < rmsd_tolerance:
                converged = True
                break

    with torch.no_grad():
        scale = model(refl).detach()
    return RefinementResult(
        scale=scale,
        outliers=outliers,
        error_model=em,
        history=history,
        converged=converged,
    )


def _with_variance(
    refl: Reflections, variance: torch.Tensor
) -> Reflections:
    """Copy of `refl` carrying a different variance column."""
    fields = {
        name: getattr(refl, name) for name in refl.__dataclass_fields__
    }
    fields["variance"] = variance
    return Reflections(**fields)


def _select_amap(amap: AsuMap, mask: torch.Tensor) -> AsuMap:
    """Subset an `AsuMap`, renumbering both groupings.

    Args:
        amap: the full map.
        mask: `(N,)` boolean row mask.

    Returns:
        A new `AsuMap` covering only the selected rows.
    """
    if bool(mask.all()):
        return amap
    anom, n_anom = _compact(amap.group_id[mask])
    laue, n_laue = _compact(amap.group_id_laue[mask])
    return AsuMap(
        asu_hkl=amap.asu_hkl[mask],
        isym=amap.isym[mask],
        plus=amap.plus[mask],
        centric=amap.centric[mask],
        epsilon=amap.epsilon[mask],
        sys_absent=amap.sys_absent[mask],
        group_id=anom,
        group_id_laue=laue,
        n_groups=n_anom,
        n_groups_laue=n_laue,
    )


def _compact(group_id: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Renumber group ids to `[0, n)`."""
    uniq, inverse = torch.unique(group_id, return_inverse=True)
    return inverse, int(uniq.numel())
