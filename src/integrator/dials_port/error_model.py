"""The DIALS two-parameter basic error model.

The model inflates the reported variance,

    sigma'^2 = a^2 (sigma^2 + (b I)^2),

with `I` the *unscaled observed* intensity, not `Ih` and not `I/g`
(`error_model.py:115-120`). `a` corrects a uniform under- or
over-estimate of the errors, `b` adds a term proportional to the
intensity, which sets the asymptotic `I/sigma` limit `1/(ab)`.

Refinement is not a joint fit. With `minimisation="individual"` DIALS
alternates two unrelated one-dimensional targets until `a` stops moving
(`engine.py:204-231`):

* `a` from the *shape* of the deviation distribution. The normalised
  deviations

      delta_hl = sqrt((n_h - 1)/n_h) (I_hl/g_hl - Ih_h) / sigma'_hl

  are sorted and regressed against standard-normal order statistics
  over the central +/- 1.5 sigma window. Both slope and intercept are
  fitted; only the slope is used, multiplicatively: `a <- a * slope`.
* `b` from the *scale* of the deviation distribution. Reflections are
  binned into 10 bins equally spaced in `log(Ih g p)`, and `b` is moved
  until the per-bin variance of `delta` is 1, under the penalty
  `phi(v) = (0.5 - v)^2 + 1/v - 1.25` (minimised at `v = 1`), weighted
  by the bin mean intensity.

Assumptions that do not transfer
--------------------------------
The `a` target is a Q-Q fit against a *standard normal*. It is
therefore a statement that the deviations of an observation from its
symmetry mean are Gaussian and homoscedastic once divided by `sigma'`.
For a strictly-positive posterior with `sigma` tied to `I` the
deviations are right-skewed, and a Q-Q fit against the normal absorbs
that skew into `a`. `order_statistics` is exposed so a different
reference distribution can be substituted.

The multiplicity prefactor in `delta_hl` is the reciprocal of the one
the variance algebra calls for, which biases `a` low by a factor
`(n_h - 1)/n_h`. See `deviation_prefactor`; the shipped convention is
reproduced by default and `prefactor="unbiased"` corrects it.

Two of DIALS' filters are sign-based proxies for "weak":
`intensities > 0.001` and `avg_I_over_var > 0.85` both exist to drop
reflections whose intensity is near or below zero. On strictly-positive
intensities the first is a no-op and the second nearly so, which means
the filtered set is larger and weaker than DIALS intends. Both
thresholds are parameters.

Guards ported verbatim: `min_reflections_required = 250`, and the
refinement is declared failed with a fallback to `a = 1.0` if the
converged `a <= 0.4` (`engine.py:224-234`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri

from .target import group_multiplicity, merged_intensity, resolve_groups
from .types import DTYPE, AsuMap, Reflections

_EPS = 1e-10

MIN_REFLECTIONS_REQUIRED = 250


@dataclass
class BasicErrorModel:
    """The two-parameter model `sigma'^2 = a^2 (sigma^2 + (b I)^2)`.

    Attributes:
        a: multiplicative error correction.
        b: intensity-proportional error term.
    """

    a: float = 1.0
    b: float = 0.02

    def apply(
        self, variance: torch.Tensor, intensity: torch.Tensor
    ) -> torch.Tensor:
        """Corrected variance, mirroring `BasicErrorModel.update_variances`.

        Args:
            variance: `(N,)` reported variance.
            intensity: `(N,)` observed, unscaled intensity.

        Returns:
            `(N,)` corrected variance.
        """
        return (self.a**2) * (variance + (self.b * intensity) ** 2)

    def sigma_prime(
        self,
        variance: torch.Tensor,
        intensity: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """`sigma'` on the *scaled* intensity scale, i.e. divided by `g`.

        This is `calc_sigmaprime`: the deviations it normalises are
        computed as `I/g - Ih`, so the error has to be divided by `g`
        too.
        """
        return torch.sqrt(self.apply(variance, intensity)) / scale

    @property
    def i_sigma_limit(self) -> float:
        """Asymptotic `I/sigma` limit, `1/(a b)`."""
        d = abs(self.a * self.b)
        return float("inf") if d == 0.0 else 1.0 / d


Prefactor = Literal["dials", "unbiased"]


def deviation_prefactor(
    n_h: torch.Tensor, mode: Prefactor = "dials"
) -> torch.Tensor:
    """Multiplicity correction applied to `(I/g - Ih)/sigma'`.

    An observation contributes to the mean it is compared against, so
    `Var(I_l - Ih) = sigma_l^2 (1 - w_l/W)`, which for equal weights is
    `sigma^2 (n-1)/n`. Normalising that to unit variance requires
    multiplying by `sqrt(n/(n-1))`.

    `calc_deltahl` multiplies by `sqrt((n-1)/n)` instead -- the
    reciprocal. The resulting deviations have standard deviation
    `(n-1)/n` rather than `1`, verified by simulation at multiplicities
    2, 3, 4, 8, 20, 40 (`0.502, 0.667, 0.749, 0.875, 0.950, 0.975`
    against a target of `1`). Since `a` is fitted to make that standard
    deviation `1`, DIALS' `a` -- and therefore every output sigma -- is
    low by a factor `(n-1)/n`. At multiplicity 2 that is a factor of
    two; on the HEWL reference, whose error-model subset has a mean
    multiplicity of 24, refitting with `"unbiased"` raises `a` from
    0.8775 to 0.9662.

    `"dials"` reproduces the shipped behaviour and is the default,
    because fidelity is the point of this port. `"unbiased"` uses
    `sqrt(n/(n-1))` and recovers a planted `a` correctly.

    Args:
        n_h: `(N,)` multiplicity of each reflection's group.
        mode: which convention to use.

    Returns:
        `(N,)` prefactor.
    """
    n = n_h.clamp_min(1.0)
    if mode == "dials":
        return torch.sqrt((n - 1.0) / n)
    if mode == "unbiased":
        return torch.sqrt(n / (n - 1.0).clamp_min(_EPS))
    raise ValueError(f"Unknown prefactor mode: {mode!r}")


def normalised_deviations(
    intensity: torch.Tensor,
    variance: torch.Tensor,
    scale: torch.Tensor,
    ih: torch.Tensor,
    n_h: torch.Tensor,
    model: BasicErrorModel,
    prefactor: Prefactor = "dials",
) -> torch.Tensor:
    """Normalised deviations `delta_hl`.

    Ports `calc_deltahl`.

    Args:
        intensity: `(N,)` observed intensity.
        variance: `(N,)` reported variance.
        scale: `(N,)` inverse scale factor.
        ih: `(N,)` merged intensity broadcast to reflections.
        n_h: `(N,)` multiplicity of each reflection's group.
        model: the error model supplying `a` and `b`.
        prefactor: multiplicity convention, see `deviation_prefactor`.

    Returns:
        `(N,)` normalised deviations.
    """
    sigma_p = model.sigma_prime(variance, intensity, scale)
    scale_factor = deviation_prefactor(n_h, prefactor)
    return (
        scale_factor * ((intensity / scale) - ih) / sigma_p.clamp_min(_EPS)
    )


def order_statistics(n: int) -> np.ndarray:
    """Expected standard-normal order statistics for `n` sorted samples.

    `Phi^-1((i + 0.5)/n)`, the plotting positions implied by DIALS'
    `linspace` in `calculate_sorted_deviations`.

    Args:
        n: number of samples.

    Returns:
        `(n,)` array of expected order statistics.
    """
    v = (np.arange(n, dtype=np.float64) + 0.5) / n
    return np.asarray(ndtri(v), dtype=np.float64)


def central_window(
    delta: np.ndarray, cutoff: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted deviations and their normal quantiles, central window only.

    Reproduces `calculate_sorted_deviations` including its index
    arithmetic: the window is chosen by rank, not by value, so it always
    contains the same *fraction* of the data whatever the deviations
    look like.

    Args:
        delta: `(N,)` normalised deviations, any order.
        cutoff: half-width of the retained window in sigma.

    Returns:
        `(x, y)` with `x` the normal quantiles and `y` the sorted
        deviations, both restricted to the central window.
    """
    n = delta.size
    ordered = np.sort(delta)
    i_min = math.ceil(float(ndtr(-cutoff)) * n - 0.5)
    i_max = math.ceil(float(ndtr(cutoff)) * n - 0.5)
    i_min = max(0, i_min)
    i_max = min(n, i_max)
    central_n = i_max - i_min
    if central_n <= 1:
        return np.zeros(0), np.zeros(0)
    v = (np.arange(central_n, dtype=np.float64) + i_min + 0.5) / n
    return np.asarray(ndtri(v), dtype=np.float64), ordered[i_min:i_max]


@dataclass
class ErrorModelFilter:
    """Thresholds for `filter_unsuitable_reflections`.

    Attributes:
        min_partiality: drop partials below this.
        min_i_over_var: per-group mean of `I/var` must exceed this. A
            sign-based proxy for "weak" -- near a no-op for
            strictly-positive intensities.
        min_ih: per-reflection `Ih * g` must exceed this. The
            normal-deviation assumption is only defended above it.
        min_intensity: per-reflection `I` must exceed this. The second
            sign-based proxy; a no-op for positive posteriors.
        max_variance_ratio_floor: floor of the internal-variance cut,
            `ratio < max(floor, max(Ih)/divisor)`.
        variance_ratio_divisor: divisor in the same expression.
        min_reflections: refuse to refine below this many reflections.
    """

    min_partiality: float = 0.4
    min_i_over_var: float = 0.85
    min_ih: float = 25.0
    min_intensity: float = 0.001
    max_variance_ratio_floor: float = 50.0
    variance_ratio_divisor: float = 40.0
    min_reflections: int = MIN_REFLECTIONS_REQUIRED


@dataclass
class ErrorModelData:
    """The filtered inputs the error model is refined on.

    `ih` is *not* recomputed as reflections are filtered away: DIALS'
    `IhTableBlock.select` carries the `Ih_values` column across
    unchanged, so every filter step compares against the merge of the
    full outlier-free table. `n_h`, by contrast, *is* recomputed on the
    final filtered set.

    Attributes:
        intensity: `(M,)` observed intensity.
        variance: `(M,)` reported variance.
        scale: `(M,)` inverse scale factor.
        ih: `(M,)` merged intensity of the unfiltered table.
        n_h: `(M,)` multiplicity within the filtered set.
        partiality: `(M,)` partiality.
        index: `(M,)` row index into the input reflections.
    """

    intensity: torch.Tensor
    variance: torch.Tensor
    scale: torch.Tensor
    ih: torch.Tensor
    n_h: torch.Tensor
    partiality: torch.Tensor
    index: torch.Tensor

    def __len__(self) -> int:
        return int(self.intensity.shape[0])


def _select_groups(
    keep_group: torch.Tensor, group_id: torch.Tensor
) -> torch.Tensor:
    """Row mask from a per-group mask, i.e. `select_on_groups`."""
    return keep_group[group_id]


def _renumber(group_id: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Compact group ids to `[0, n)` after a selection."""
    uniq, inv = torch.unique(group_id, return_inverse=True)
    return inv, int(uniq.numel())


def prepare_error_model_data(
    refl: Reflections,
    amap: AsuMap,
    scale: torch.Tensor,
    filters: ErrorModelFilter | None = None,
    anomalous: bool = True,
    pool_centrics: bool = True,
) -> ErrorModelData:
    """Filter reflections down to those suitable for error modelling.

    Ports `filter_unsuitable_reflections` step for step, in its original
    order. DIALS builds this table with anomalous grouping and with
    outliers already removed (`scaler.py:383-384`), so `anomalous`
    defaults to `True` and outlier removal is the caller's job.

    Args:
        refl: unmerged observations, outliers already removed.
        amap: symmetry map for the same rows.
        scale: `(N,)` inverse scale factors, held fixed.
        filters: threshold set; defaults to DIALS' values.
        anomalous: group Friedel mates separately.
        pool_centrics: see `target.resolve_groups`.

    Returns:
        The filtered data.

    Raises:
        ValueError: if fewer than `filters.min_reflections` survive.
    """
    f = filters or ErrorModelFilter()
    group_id, n_groups = resolve_groups(amap, anomalous, pool_centrics)
    weights = 1.0 / refl.variance.clamp_min(_EPS)
    _, ih_all = merged_intensity(
        refl.intensity, weights, scale, group_id, n_groups
    )

    index = torch.arange(len(refl), dtype=torch.int64)
    keep = refl.partiality > f.min_partiality

    def take(mask: torch.Tensor) -> None:
        nonlocal index, group_id
        index = index[mask]
        group_id = group_id[mask]

    take(keep)
    group_id, n_groups = _renumber(group_id)

    i_over_var = refl.intensity[index] / refl.variance[index].clamp_min(_EPS)
    sums = torch.zeros(n_groups, dtype=DTYPE).index_add(
        0, group_id, i_over_var
    )
    counts = torch.zeros(n_groups, dtype=DTYPE).index_add(
        0, group_id, torch.ones_like(i_over_var)
    )
    take(_select_groups(sums / counts > f.min_i_over_var, group_id))
    group_id, n_groups = _renumber(group_id)

    n_h = group_multiplicity(group_id, n_groups)
    keep = (
        (ih_all[index] * scale[index] > f.min_ih)
        & (n_h > 1.0)
        & (refl.intensity[index] > f.min_intensity)
    )
    take(keep)
    group_id, n_groups = _renumber(group_id)
    if index.numel() < f.min_reflections:
        raise ValueError(
            f"Insufficient reflections ({index.numel()} < "
            f"{f.min_reflections}) to perform error modelling."
        )

    n_h = group_multiplicity(group_id, n_groups)
    take(n_h > 1.0)
    group_id, n_groups = _renumber(group_id)

    i = refl.intensity[index]
    g = scale[index]
    mu = ih_all[index]
    n_h = group_multiplicity(group_id, n_groups)
    resid_sq = torch.zeros(n_groups, dtype=DTYPE).index_add(
        0, group_id, ((i / g) - mu) ** 2
    )
    n_per_group = torch.zeros(n_groups, dtype=DTYPE).index_add(
        0, group_id, torch.ones_like(i)
    )
    var_sum = torch.zeros(n_groups, dtype=DTYPE).index_add(
        0, group_id, refl.variance[index] / g**2
    )
    group_var = resid_sq / (n_per_group - 1.0).clamp_min(_EPS)
    avg_var = var_sum / n_per_group
    ratio = group_var / avg_var.clamp_min(_EPS)
    limit = max(
        f.max_variance_ratio_floor,
        float(mu.max()) / f.variance_ratio_divisor,
    )
    take(_select_groups(ratio < limit, group_id))
    group_id, n_groups = _renumber(group_id)

    if index.numel() < f.min_reflections:
        raise ValueError(
            f"Insufficient reflections ({index.numel()} < "
            f"{f.min_reflections}) to perform error modelling."
        )

    return ErrorModelData(
        intensity=refl.intensity[index],
        variance=refl.variance[index],
        scale=scale[index],
        ih=ih_all[index],
        n_h=group_multiplicity(group_id, n_groups),
        partiality=refl.partiality[index],
        index=index,
    )


@dataclass
class BinningInfo:
    """Diagnostics from the intensity binning.

    Attributes:
        boundaries: `(n_bins + 1,)` bin edges, descending.
        mean_intensities: `(n_bins,)` mean `Ih g p` per bin, the weights
            of the `b` target.
        refl_per_bin: `(n_bins,)` occupancy.
        initial_variances: `(n_bins,)` bin variance of `delta` at
            `a = 1, b = 0`, i.e. before any correction.
        bin_variances: `(n_bins,)` current bin variance of `delta`.
    """

    boundaries: np.ndarray
    mean_intensities: np.ndarray
    refl_per_bin: np.ndarray
    initial_variances: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    bin_variances: np.ndarray = field(default_factory=lambda: np.zeros(0))


class ErrorModelBinner:
    """Fixed bins in `log(Ih g p)`, and the bin variances of `delta`.

    The binning is computed once, from the data alone, and never
    updated: only the deviations that fall in each bin change as `a` and
    `b` move. Ports `ErrorModelBinner._create_summation_matrix`,
    including its two corrections -- boundaries are pushed down to fill
    under-occupied bins, and bins still short by more than 5 are dropped
    entirely.

    Args:
        data: the filtered error-model data.
        n_bins: target number of bins.
        min_reflections_required: minimum bin occupancy, itself raised
            to `n/100` when the dataset is large.
        prefactor: multiplicity convention, see `deviation_prefactor`.
    """

    def __init__(
        self,
        data: ErrorModelData,
        n_bins: int = 10,
        min_reflections_required: int = MIN_REFLECTIONS_REQUIRED,
        prefactor: Prefactor = "dials",
    ) -> None:
        self.data = data
        self.n_bins = n_bins
        self.min_reflections_required = min_reflections_required
        self.prefactor = prefactor
        self._build()
        self.info.initial_variances = self.bin_variances_for(
            BasicErrorModel(1.0, 0.0)
        )
        self.info.bin_variances = self.info.initial_variances.copy()

    def _build(self) -> None:
        ih = (
            self.data.ih * self.data.scale * self.data.partiality
        ).numpy()
        n = ih.size
        order = np.argsort(-ih, kind="stable")
        i_max = float(ih.max())
        i_min_data = float(ih.min())
        i_min = max(1.0, i_min_data)
        spacing = (math.log(i_max) - math.log(i_min)) / float(self.n_bins)
        bounds = [i_max] + [
            math.exp(math.log(i_max) - i * spacing)
            for i in range(1, self.n_bins + 1)
        ]
        bounds[-1] = i_min_data - 0.01

        min_required = self.min_reflections_required
        if n > 100 * min_required:
            min_required = int(n / 100.0)
        min_per_bin = min(min_required, int(n / (3.0 * self.n_bins)))

        assign = np.full(n, -1, dtype=np.int64)
        per_bin: list[int] = []
        n_cumul = 0
        for i in range(len(bounds) - 1):
            hi = bounds[i]
            sel = (ih <= hi) & (ih > bounds[i + 1])
            n_in_bin = int(np.count_nonzero(sel))
            if n_in_bin < min_per_bin:
                m = n_cumul + min_per_bin
                if m < n:
                    bounds[i + 1] = float(ih[order[m]])
                    sel = (ih <= hi) & (ih > bounds[i + 1])
                    n_in_bin = int(np.count_nonzero(sel))
            per_bin.append(n_in_bin)
            assign[sel] = i
            n_cumul += n_in_bin

        counts = np.asarray(per_bin, dtype=np.float64)
        keep = counts >= (min_per_bin - 5)
        if not keep.all():
            bounds_arr = np.asarray(bounds)
            new_bounds = np.append(bounds_arr[:-1][keep], bounds_arr[-1])
            remap = np.full(len(per_bin), -1, dtype=np.int64)
            remap[np.nonzero(keep)[0]] = np.arange(int(keep.sum()))
            valid = assign >= 0
            assign[valid] = remap[assign[valid]]
            bounds = list(new_bounds)
            counts = counts[keep]

        n_kept = len(bounds) - 1
        means = np.array(
            [
                float(ih[(ih <= bounds[i]) & (ih > bounds[i + 1])].mean())
                for i in range(n_kept)
            ]
        )
        self.assign = torch.from_numpy(assign)
        self.n_kept = n_kept
        self.info = BinningInfo(
            boundaries=np.asarray(bounds),
            mean_intensities=means,
            refl_per_bin=counts,
        )

    @property
    def weights(self) -> np.ndarray:
        """Bin weights of the `b` target: the bin mean intensities."""
        return self.info.mean_intensities

    def deviations(self, model: BasicErrorModel) -> torch.Tensor:
        """Normalised deviations at the given `(a, b)`."""
        return normalised_deviations(
            self.data.intensity,
            self.data.variance,
            self.data.scale,
            self.data.ih,
            self.data.n_h,
            model,
            self.prefactor,
        )

    def bin_variances_for(self, model: BasicErrorModel) -> np.ndarray:
        """Per-bin variance of `delta`, DIALS' biased `E[d^2] - E[d]^2`."""
        d = self.deviations(model).numpy()
        assign = self.assign.numpy()
        valid = assign >= 0
        idx = assign[valid]
        dv = d[valid]
        counts = np.bincount(idx, minlength=self.n_kept).astype(np.float64)
        s1 = np.bincount(idx, weights=dv, minlength=self.n_kept)
        s2 = np.bincount(idx, weights=dv**2, minlength=self.n_kept)
        counts = np.maximum(counts, 1.0)
        return (s2 / counts) - (s1 / counts) ** 2

    def update(self, model: BasicErrorModel) -> np.ndarray:
        """Recompute and store the bin variances for a new `(a, b)`."""
        self.info.bin_variances = self.bin_variances_for(model)
        return self.info.bin_variances


def variance_penalty(bin_variances: np.ndarray) -> np.ndarray:
    """`phi(v) = (0.5 - v)^2 + 1/v - 1.25`, minimised at `v = 1`.

    Args:
        bin_variances: `(n_bins,)` per-bin variance of `delta`.

    Returns:
        `(n_bins,)` penalty values.
    """
    v = np.clip(bin_variances, 1e-8, None)
    return (0.5 - v) ** 2 + (1.0 / v) - 1.25


@dataclass
class ErrorModelResult:
    """Outcome of an error model refinement.

    Attributes:
        model: the refined model.
        converged: whether the `a` sequence met its tolerance.
        failed: whether the `a <= 0.4` fallback was taken.
        n_refl: number of reflections used.
        a_history: `a` after each cycle.
        b_history: `b` after each cycle.
        binning: binning diagnostics at the refined parameters.
    """

    model: BasicErrorModel
    converged: bool
    failed: bool
    n_refl: int
    a_history: list[float]
    b_history: list[float]
    binning: BinningInfo


def _fit_a(
    binner: ErrorModelBinner, model: BasicErrorModel, cutoff: float
) -> float:
    """One `a` step: the slope of the central-window normal Q-Q fit.

    DIALS drives a quadratic through LBFGS here. The quadratic is an
    ordinary two-parameter least-squares fit, so its minimiser is the
    closed-form slope; the intercept is fitted and then discarded, as
    in `ErrorModelA_APM.resolve_model_parameters`.
    """
    x, y = central_window(binner.deviations(model).numpy(), cutoff)
    if x.size < 2:
        return model.a
    xm = x.mean()
    ym = y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom <= 0.0:
        return model.a
    slope = float(((x - xm) * (y - ym)).sum() / denom)
    return model.a * slope


def _fit_b(
    binner: ErrorModelBinner, model: BasicErrorModel, b_max: float
) -> float:
    """One `b` step: bounded 1-D minimisation of the weighted penalty.

    DIALS uses LBFGS with an analytic gradient. The objective is a
    smooth scalar function of a single bounded parameter, so a bounded
    Brent search finds the same minimum without the gradient.
    """
    w = binner.weights
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return model.b

    def objective(b: float) -> float:
        v = binner.bin_variances_for(BasicErrorModel(model.a, abs(b)))
        return float((variance_penalty(v) * w).sum() / w_sum)

    res = minimize_scalar(
        objective, bounds=(0.0, b_max), method="bounded",
        options={"xatol": 1e-6},
    )
    return float(abs(res.x))


def refine_error_model(
    refl: Reflections,
    amap: AsuMap,
    scale: torch.Tensor,
    initial: BasicErrorModel | None = None,
    filters: ErrorModelFilter | None = None,
    n_bins: int = 10,
    max_cycles: int = 50,
    a_tolerance: float = 0.01,
    central_cutoff: float = 1.5,
    b_max: float = 1.0,
    anomalous: bool = True,
    pool_centrics: bool = True,
    prefactor: Prefactor = "dials",
    min_a: float = 0.4,
) -> ErrorModelResult:
    """Refine `(a, b)` at fixed scale factors.

    Alternates the two one-dimensional targets until `a` changes by less
    than `a_tolerance` relatively, then takes one final `a` step -- the
    order in `ErrorModelRefinery.run`. If the converged `a` is at or
    below `min_a` the refinement is declared failed and DIALS' fallback
    is applied: `a` fixed at 1.0 and `b` refined once against it.

    Args:
        refl: outlier-free unmerged observations. `variance` must be the
            *uncorrected* reported variance -- the model is absolute,
            not incremental, so feeding it an already-corrected variance
            will return `a` near 1 and `b` near 0.
        amap: symmetry map for the same rows.
        scale: `(N,)` inverse scale factors, held fixed throughout.
        initial: starting parameters; defaults to `a=1, b=0.02`.
        filters: filter thresholds.
        n_bins: number of intensity bins for the `b` target.
        max_cycles: cap on alternations.
        a_tolerance: relative change in `a` defining convergence.
        central_cutoff: half-width in sigma of the `a` fitting window.
        b_max: upper bound of the `b` search.
        anomalous: group Friedel mates separately, as DIALS does here.
        pool_centrics: see `target.resolve_groups`.
        prefactor: multiplicity convention, see `deviation_prefactor`.
        min_a: converged `a` at or below this triggers the fallback.

    Returns:
        The refinement outcome.
    """
    data = prepare_error_model_data(
        refl, amap, scale, filters, anomalous, pool_centrics
    )
    binner = ErrorModelBinner(
        data, n_bins=n_bins,
        min_reflections_required=(
            filters.min_reflections if filters else MIN_REFLECTIONS_REQUIRED
        ),
        prefactor=prefactor,
    )
    model = BasicErrorModel(
        a=initial.a if initial else 1.0,
        b=initial.b if initial else 0.02,
    )

    a_hist: list[float] = []
    b_hist: list[float] = []
    converged = False
    for _ in range(max_cycles):
        model.a = _fit_a(binner, model, central_cutoff)
        a_hist.append(model.a)
        model.b = _fit_b(binner, model, b_max)
        b_hist.append(model.b)
        if len(a_hist) > 1 and a_hist[-2] > 0:
            rel = abs((a_hist[-2] - a_hist[-1]) / a_hist[-2])
            if rel < a_tolerance:
                converged = True
                break

    failed = not (converged and a_hist and a_hist[-1] > min_a)
    if failed:
        model = BasicErrorModel(a=1.0, b=0.02)
        model.b = _fit_b(binner, model, b_max)
        a_hist.append(model.a)
        b_hist.append(model.b)
    else:
        model.a = _fit_a(binner, model, central_cutoff)
        a_hist.append(model.a)

    binner.update(model)
    return ErrorModelResult(
        model=model,
        converged=converged,
        failed=failed,
        n_refl=len(data),
        a_history=a_hist,
        b_history=b_hist,
        binning=binner.info,
    )
