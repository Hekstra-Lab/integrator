"""The scaling objective, with a swappable observation likelihood.

DIALS minimises a Gaussian weighted least-squares functional

    L = sum_hl w_hl (I_hl - g_hl * Ih_h)^2,    w_hl = 1 / sigma_hl^2

(`target_function.py:60-129`). Two features of that expression carry
assumptions that do not survive a change of integrator:

1.  The model is scaled *up* to meet the observation, not the other way
    round: the residual is `I - g*Ih`, so the weight belongs to the
    observation and is independent of `g`.

2.  `Ih` is not a refined parameter. At fixed `g` the minimiser of `L`
    over `Ih` is the inverse-variance weighted mean

        Ih = sum(w g I) / sum(w g^2)

    (`Ih_table.py:642-648`), so DIALS substitutes it back in and
    profiles it out. It still depends on the scale parameters, which is
    why `calculate_gradients` carries an explicit `dIh/dp` term. Here
    `merged_intensity` is written in torch and autograd supplies that
    term for free, provided `Ih` is not detached.

    That term is, however, arithmetically zero for DIALS' own target.
    The chain-rule contribution is `dL/dIh * dIh/dp`, and `dL/dIh`
    vanishes precisely because `Ih` was chosen as the minimiser -- the
    envelope theorem. `calc_dIh_by_dpi` and `term_2` of
    `calculate_gradients` compute a quantity that sums to zero. Setting
    `detach_ih=True` therefore leaves the Gaussian gradient unchanged
    to machine precision (verified at 2e-15 relative in
    `scripts/verify_dials_port_target.py`).

    The cancellation is not a property of least squares as such but of
    the merge agreeing with the likelihood. With `weight_mode="expected"`
    the inverse-variance weighted mean coincides with the
    maximum-likelihood merge for the Gamma and Poisson likelihoods too,
    and the term vanishes there as well. Under `"observed"` weights it
    does not: `detach_ih` then changes the Gamma gradient by ~14% and
    the Poisson gradient by ~7%.

The weighting bias
------------------
DIALS holds `w = 1/sigma^2` FIXED during minimisation, and `sigma` is a
property of the observation. For counting data that is close to
harmless. For a variational integrator whose posterior is strictly
positive with `sigma` a deterministic function of `I` -- a Gamma
posterior has `var = I^2/k` -- it is not: an observation that fluctuates
low reports a smaller sigma, receives a larger weight, and pulls the
weighted mean down with it. The merged intensity is then biased low,
systematically, and the bias grows as the posterior gets more
dispersed.

`weight_mode` exposes the three options:

* `"observed"` -- `w = 1/var`, evaluated at the observed intensity.
  This is DIALS' behaviour and the default.
* `"expected"` -- `w` evaluated at the current model mean `g*Ih`
  instead, which breaks the correlation between a fluctuation and its
  own weight and removes the bias. The variance is transported from the
  observed to the expected point by holding `var / I^p` fixed, with `p`
  = `variance_power`: `p=2` keeps the coefficient of variation constant
  (Gamma), `p=1` keeps the Fano factor constant (Poisson), `p=0` is a
  no-op.
* `"unity"` -- unweighted, for diagnostics.

Weights are always detached from the graph, matching DIALS' treatment of
them as constants within a minimisation step; with `"expected"` this
makes the objective an IRLS/Gauss-Newton surrogate rather than a
strictly differentiated one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

import torch

from .types import DTYPE, AsuMap, Reflections

WeightMode = Literal["observed", "expected", "unity"]

_EPS = 1e-10


def merged_intensity(
    intensity: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Profile out `Ih`, the weighted mean over symmetry equivalents.

    Implements `IhTableBlock.calc_Ih`: `Ih = sum(w g I) / sum(w g^2)`,
    the closed-form minimiser of the weighted least-squares functional
    at fixed `g`. Groups whose denominator vanishes are returned as
    zero rather than raising.

    Args:
        intensity: `(N,)` observed intensities.
        weights: `(N,)` observation weights, typically `1/variance`.
        scale: `(N,)` inverse scale factors.
        group_id: `(N,)` symmetry group index in `[0, n_groups)`.
        n_groups: number of distinct groups.

    Returns:
        `(Ih_per_group, Ih_per_refl)` of shapes `(n_groups,)` and
        `(N,)`.
    """
    wg = weights * scale
    num = torch.zeros(n_groups, dtype=DTYPE, device=intensity.device)
    den = torch.zeros(n_groups, dtype=DTYPE, device=intensity.device)
    num = num.index_add(0, group_id, wg * intensity)
    den = den.index_add(0, group_id, wg * scale)
    ih = num / den.clamp_min(_EPS)
    ih = torch.where(den > 0, ih, torch.zeros_like(ih))
    return ih, ih[group_id]


def resolve_groups(
    amap: AsuMap, anomalous: bool, pool_centrics: bool = True
) -> tuple[torch.Tensor, int]:
    """Symmetry grouping, with cctbx's treatment of centric reflections.

    An anomalous grouping must not split centric reflections by Friedel
    parity: a centric reflection satisfies `F(h) = F(-h)` exactly, so
    its two hemispheres are the same measurement.
    `cctbx.miller.set.map_to_asu(anomalous_flag=True)`, which DIALS'
    `IhTable` uses, pools them. Splitting them invents spurious groups
    -- on the HEWL reference, 25099 instead of the correct 23096, and
    the error model then fits 307677 reflections instead of DIALS'
    306944.

    `AsuMap` builds its anomalous key the same way, so this is normally
    a no-op. It is applied anyway because pooling is idempotent and the
    cost is one `torch.unique`, which keeps the grouping correct here
    regardless of how the key upstream is defined.

    Args:
        amap: the symmetry map.
        anomalous: keep Friedel mates in separate groups.
        pool_centrics: pool the two hemispheres of centric reflections
            when `anomalous` is true. `False` takes `AsuMap`'s key
            verbatim.

    Returns:
        `(group_id, n_groups)`.
    """
    if not anomalous or not pool_centrics:
        return amap.groups(anomalous)
    key = amap.group_id_laue * 2 + (amap.plus & ~amap.centric).to(
        amap.group_id_laue.dtype
    )
    uniq, inverse = torch.unique(key, return_inverse=True)
    return inverse, int(uniq.numel())


def group_multiplicity(
    group_id: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Return `(N,)` the number of observations in each row's group."""
    counts = torch.zeros(n_groups, dtype=DTYPE, device=group_id.device)
    counts = counts.index_add(
        0, group_id, torch.ones_like(group_id, dtype=DTYPE)
    )
    return counts[group_id]


@runtime_checkable
class Likelihood(Protocol):
    """An observation likelihood for the scaling target.

    `__call__` returns the per-reflection contribution to the
    functional; the target sums it. `weights` supplies the weights used
    to profile out `Ih`, which is a separate concern only because the
    weighted mean is the exact minimiser for the Gaussian case and an
    approximation elsewhere.
    """

    weights_depend_on_ih: bool

    def __call__(
        self, obs: Reflections, scale: torch.Tensor, ih: torch.Tensor
    ) -> torch.Tensor:
        """Per-reflection functional contribution, shape `(N,)`."""
        ...

    def weights(
        self,
        obs: Reflections,
        scale: torch.Tensor,
        ih: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-reflection weights for the `Ih` merge, shape `(N,)`."""
        ...


class _WeightedLikelihood:
    """Weight bookkeeping shared by the concrete likelihoods.

    Attributes:
        weight_mode: one of `"observed"`, `"expected"`, `"unity"`.
        variance_power: exponent `p` in the transport rule
            `var(mu) = var(I) * (mu/I)^p` used by `"expected"`.
    """

    variance_power: float = 2.0

    def __init__(
        self,
        weight_mode: WeightMode = "observed",
        variance_power: float | None = None,
    ) -> None:
        if weight_mode not in ("observed", "expected", "unity"):
            raise ValueError(f"Unknown weight_mode: {weight_mode!r}")
        self.weight_mode: WeightMode = weight_mode
        if variance_power is not None:
            self.variance_power = float(variance_power)

    @property
    def weights_depend_on_ih(self) -> bool:
        """Whether the weights need `Ih`, forcing a second merge pass."""
        return self.weight_mode == "expected"

    def expected_variance(
        self, obs: Reflections, mu: torch.Tensor
    ) -> torch.Tensor:
        """Transport the observed variance to the model mean `mu`."""
        if self.variance_power == 0.0:
            return obs.variance
        ratio = mu.clamp_min(_EPS) / obs.intensity.abs().clamp_min(_EPS)
        return obs.variance * ratio**self.variance_power

    def weights(
        self,
        obs: Reflections,
        scale: torch.Tensor,
        ih: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-reflection weights, always detached from the graph."""
        if self.weight_mode == "unity":
            return torch.ones_like(obs.intensity)
        if self.weight_mode == "observed" or ih is None:
            return 1.0 / obs.variance.clamp_min(_EPS)
        mu = (scale * ih).detach()
        return 1.0 / self.expected_variance(obs, mu).clamp_min(_EPS)


class GaussianLeastSquares(_WeightedLikelihood):
    """DIALS' target: `w (I - g Ih)^2`.

    The baseline. With `weight_mode="observed"` and
    `ScalingTarget(detach_ih=False)` this reproduces
    `ScalingTarget.compute_functional_gradients` including the implicit
    `dIh/dp` term.

    Args:
        weight_mode: see the module docstring.
        variance_power: exponent used to transport the variance when
            `weight_mode="expected"`. Defaults to `2` (constant
            coefficient of variation).
    """

    def __call__(
        self, obs: Reflections, scale: torch.Tensor, ih: torch.Tensor
    ) -> torch.Tensor:
        """Weighted squared residuals, shape `(N,)`."""
        w = self.weights(obs, scale, ih)
        residual = obs.intensity - scale * ih
        return w * residual**2


class GammaLikelihood(_WeightedLikelihood):
    """Negative log-density of a Gamma observation about `g Ih`.

    Parameterisation: the observation is treated as a draw from
    `Gamma(shape=k, mean=mu)` with `mu = g Ih` and the shape taken from
    the reported posterior, `k = I^2 / var`. Holding `k` fixed and
    moving the mean is the statement that the integrator's *relative*
    precision is a property of the measurement while its location is
    predicted by the scale model. The negative log-density is

        -log p = log Gamma(k) - k log k + k log mu
                 - (k - 1) log I + k I / mu

    Expanding about `I = mu` gives `-log p ~ (k / 2 mu^2) (I - mu)^2`,
    i.e. an effective weight `k/mu^2` evaluated at the *model* mean.
    The Gamma likelihood therefore carries `weight_mode="expected"`
    weighting intrinsically and is not subject to the downward `Ih` bias
    described in the module docstring, whatever `weight_mode` is set to
    for the merge itself.

    Observations must be positive; non-positive intensities are clamped
    and counted in `n_clamped`.

    Args:
        weight_mode: weighting used for the `Ih` merge only.
        min_shape: lower clamp on `k`, guarding against `var >> I^2`.
        max_shape: upper clamp on `k`.
    """

    variance_power = 2.0

    def __init__(
        self,
        weight_mode: WeightMode = "expected",
        min_shape: float = 1e-3,
        max_shape: float = 1e8,
    ) -> None:
        super().__init__(weight_mode)
        self.min_shape = float(min_shape)
        self.max_shape = float(max_shape)
        self.n_clamped = 0

    def shape(self, obs: Reflections) -> torch.Tensor:
        """Gamma shape `k = I^2 / var`, clamped."""
        i = obs.intensity.clamp_min(_EPS)
        k = i**2 / obs.variance.clamp_min(_EPS)
        return k.clamp(self.min_shape, self.max_shape)

    def __call__(
        self, obs: Reflections, scale: torch.Tensor, ih: torch.Tensor
    ) -> torch.Tensor:
        """Negative log-density per reflection, shape `(N,)`."""
        k = self.shape(obs).detach()
        i = obs.intensity
        self.n_clamped = int((i <= 0).sum())
        i = i.clamp_min(_EPS)
        mu = (scale * ih).clamp_min(_EPS)
        return (
            torch.lgamma(k)
            - k * torch.log(k)
            + k * torch.log(mu)
            - (k - 1.0) * torch.log(i)
            + k * i / mu
        )


class PoissonLikelihood(_WeightedLikelihood):
    """Counting limit: negative log-density of `I/gain ~ Poisson(mu/gain)`.

    Intensities need not be integers -- the terms in `n` alone are
    constant in the parameters and are dropped -- so this is the
    quasi-Poisson functional `mu/gain - (I/gain) log(mu/gain)`, which
    has the same minimiser and the same gradients as the exact
    log-density. Non-positive intensities contribute the `mu` term only.

    Args:
        gain: detector gain, `counts = intensity / gain`.
        weight_mode: weighting used for the `Ih` merge. Poisson
            transports variance with `variance_power=1`.
    """

    variance_power = 1.0

    def __init__(
        self, gain: float = 1.0, weight_mode: WeightMode = "observed"
    ) -> None:
        super().__init__(weight_mode)
        self.gain = float(gain)

    def __call__(
        self, obs: Reflections, scale: torch.Tensor, ih: torch.Tensor
    ) -> torch.Tensor:
        """Negative log-density per reflection, shape `(N,)`."""
        n = (obs.intensity / self.gain).clamp_min(0.0)
        rate = (scale * ih / self.gain).clamp_min(_EPS)
        return rate - n * torch.log(rate)


ScaleLike = torch.Tensor | Callable[[], torch.Tensor]


def _resolve(scale: ScaleLike) -> torch.Tensor:
    """Accept either a tensor or a nullary callable producing one."""
    return scale() if callable(scale) else scale


class ScalingTarget:
    """The scaling objective: profile out `Ih`, then score the residuals.

    Args:
        likelihood: the observation likelihood.
        anomalous: keep Friedel mates in separate symmetry groups.
        detach_ih: cut `Ih` out of the autograd graph. DIALS keeps the
            `dIh/dp` term (`calc_dIh_by_dpi`), so `detach_ih=False` --
            the default -- is what matches DIALS. `True` gives the
            alternating/EM-style gradient in which `Ih` is treated as a
            constant. The two coincide whenever the merge is the
            likelihood's own MLE; see the module docstring.
        n_merge_passes: how many times to alternate weights and `Ih`
            when the weights depend on `Ih`. Ignored otherwise.
        pool_centrics: see `resolve_groups`. Only used when
            `anomalous` is true.
    """

    def __init__(
        self,
        likelihood: Likelihood,
        anomalous: bool = False,
        detach_ih: bool = False,
        n_merge_passes: int = 2,
        pool_centrics: bool = True,
    ) -> None:
        self.likelihood = likelihood
        self.anomalous = anomalous
        self.detach_ih = detach_ih
        self.n_merge_passes = int(n_merge_passes)
        self.pool_centrics = pool_centrics

    def merge(
        self, refl: Reflections, amap: AsuMap, scale: ScaleLike
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(Ih_per_refl, weights)` at the given scale."""
        g = _resolve(scale)
        group_id, n_groups = resolve_groups(
            amap, self.anomalous, self.pool_centrics
        )
        n_passes = (
            self.n_merge_passes
            if getattr(self.likelihood, "weights_depend_on_ih", False)
            else 1
        )
        ih_refl: torch.Tensor | None = None
        w = self.likelihood.weights(refl, g, None)
        for _ in range(max(1, n_passes)):
            _, ih_refl = merged_intensity(
                refl.intensity, w, g, group_id, n_groups
            )
            w = self.likelihood.weights(refl, g, ih_refl)
        assert ih_refl is not None
        if self.detach_ih:
            ih_refl = ih_refl.detach()
        return ih_refl, w

    def residuals(
        self, refl: Reflections, amap: AsuMap, scale: ScaleLike
    ) -> torch.Tensor:
        """Per-reflection functional contributions, shape `(N,)`."""
        g = _resolve(scale)
        ih_refl, _ = self.merge(refl, amap, g)
        return self.likelihood(refl, g, ih_refl)

    def __call__(
        self, refl: Reflections, amap: AsuMap, scale: ScaleLike
    ) -> torch.Tensor:
        """Scalar loss, summed over reflections."""
        return self.residuals(refl, amap, scale).sum()

    def rmsd(
        self, refl: Reflections, amap: AsuMap, scale: ScaleLike
    ) -> torch.Tensor:
        """`sqrt(L / N)`, DIALS' `RMSD_I` when the likelihood is Gaussian.

        Used as the convergence statistic by `refine`, mirroring
        `ScalingTarget.rmsds`. For non-Gaussian likelihoods the
        functional is not a sum of squares, so this is a monotone
        rescaling of the loss rather than a residual RMS; it still
        serves as a scale-free convergence signal.
        """
        resid = self.residuals(refl, amap, scale)
        return torch.sqrt(resid.sum().abs() / max(len(refl), 1))
