"""Merging statistics: CC1/2, CC*, R-factors, completeness, I/sigma.

A torch port of `iotbx.merging_statistics.dataset_statistics` together
with the extensions DIALS layers on top of it in
`dials.algorithms.scaling.scaling_library.ExtendedDatasetStatistics`
(`R_split` and the inverse-variance weighted CC1/2).

Everything is expressed as `index_add_` reductions over a group index,
so a full 400k-reflection dataset costs a handful of passes rather than
a Python loop over 12k symmetry groups.

Two conventions are worth stating up front, because they are the usual
source of confusion when comparing numbers against a `dials.scale` log:

* CC1/2 exists in two flavours. The one DIALS prints by default is
  `cc_half_random`, which assigns each group's observations to two
  halves at random. cctbx draws those assignments from a scitbx Mersenne
  twister, so an exact reproduction is impossible from torch; use
  `n_repeats` to measure the resulting Monte-Carlo spread. The
  `sigma-tau` variant (Assmann et al., J. Appl. Cryst. 49, 1021) is
  analytic and reproduces cctbx to machine precision, which makes it the
  better primary metric here.
* `cc_anom` is a *half-dataset* correlation of anomalous differences,
  not a correlation between two datasets. It is computed on the
  anomalous grouping with unweighted half means, matching
  `cctbx.miller.array.half_dataset_anomalous_correlation`.

The DIALS merge itself (`merge.merge`) is not imported here, so this
module stays usable on its own; the small grouped reduction it needs is
reimplemented in `_merge_equivalents`. R-factors come out of that same
reduction. `merge.r_factors` is the canonical DIALS-side implementation
and the two are required to agree; `merging_statistics` accepts an
`r_factor_fn` hook so the merge module's version can be substituted
without this module importing it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import gemmi
import numpy as np
import torch

from .symmetry import as_spacegroup, complete_set
from .types import DTYPE, IDTYPE, AsuMap, MergedData, Reflections


class CrystalMetadata(Protocol):
    """The part of `io.Experiment` the statistics actually need."""

    @property
    def cell(self) -> tuple[float, float, float, float, float, float]:
        """Unit cell `(a, b, c, alpha, beta, gamma)`."""

    @property
    def space_group(self) -> str:
        """Space group symbol, Hermann-Mauguin or Hall."""


# cctbx substitutes this for a zero sigma inside merge_equivalents_obs.
_SIGMA_FLOOR = 1e-3

# `binning::init_limits` default, and `complete_set` d_min slack.
_RELATIVE_TOLERANCE = 1e-6


def _group_sum(
    values: torch.Tensor, group_id: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Sum `values` into `n_groups` bins indexed by `group_id`.

    Args:
        values: `(N,)` values to accumulate.
        group_id: `(N,)` destination index in `[0, n_groups)`.
        n_groups: number of output bins.

    Returns:
        `(n_groups,)` sums, zero where a group is empty.
    """
    out = torch.zeros(n_groups, dtype=values.dtype)
    out.index_add_(0, group_id, values)
    return out


def _group_count(group_id: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Count observations per group.

    Args:
        group_id: `(N,)` group index.
        n_groups: number of output bins.

    Returns:
        `(n_groups,)` int64 counts.
    """
    return torch.bincount(group_id, minlength=n_groups).to(IDTYPE)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson product-moment correlation of two 1-D tensors.

    Args:
        x: `(n,)` first variable.
        y: `(n,)` second variable.

    Returns:
        The correlation coefficient, or `nan` if either variance is
        zero or fewer than two points are supplied.
    """
    if x.numel() < 2:
        return float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    den = torch.sqrt((dx * dx).sum() * (dy * dy).sum())
    if den == 0:
        return float("nan")
    return float((dx * dy).sum() / den)


def d_star_sq(
    hkl: torch.Tensor,
    cell: tuple[float, float, float, float, float, float],
) -> torch.Tensor:
    """Vectorised `1/d^2` from Miller indices and a unit cell.

    `symmetry.resolution` loops in Python, which is too slow for the
    unmerged arrays this module reduces over.

    Args:
        hkl: `(N, 3)` integer Miller indices.
        cell: unit cell `(a, b, c, alpha, beta, gamma)`.

    Returns:
        `(N,)` `d*^2` in inverse square Angstrom.
    """
    g = gemmi.UnitCell(*cell).reciprocal_metric_tensor()
    h = hkl.to(DTYPE)
    return (
        h[:, 0] ** 2 * g.u11
        + h[:, 1] ** 2 * g.u22
        + h[:, 2] ** 2 * g.u33
        + 2.0 * h[:, 0] * h[:, 1] * g.u12
        + 2.0 * h[:, 0] * h[:, 2] * g.u13
        + 2.0 * h[:, 1] * h[:, 2] * g.u23
    )


@dataclass
class ResolutionBins:
    """Equal-volume resolution shells.

    Attributes:
        bin_id: `(N,)` shell index in `[0, n_bins)`.
        limits: `(n_bins + 1,)` ascending `d*^2` shell edges.
    """

    bin_id: torch.Tensor
    limits: torch.Tensor

    @property
    def n_bins(self) -> int:
        """Number of shells."""
        return int(self.limits.numel()) - 1

    @property
    def d_edges(self) -> torch.Tensor:
        """`(n_bins + 1,)` shell edges in Angstrom, descending."""
        return 1.0 / torch.sqrt(self.limits)

    def d_range(self, i: int) -> tuple[float, float]:
        """Return `(d_max, d_min)` in Angstrom for shell `i`."""
        edges = self.d_edges
        return float(edges[i]), float(edges[i + 1])

    @property
    def labels(self) -> list[str]:
        """Human-readable `"d_max - d_min"` label per shell."""
        return [
            "{:.2f} - {:.2f}".format(*self.d_range(i))
            for i in range(self.n_bins)
        ]


def resolution_bins(
    d: torch.Tensor,
    n_bins: int = 20,
    d_max: float | None = None,
    d_min: float | None = None,
    relative_tolerance: float = _RELATIVE_TOLERANCE,
) -> ResolutionBins:
    """Assign reflections to equal-volume resolution shells.

    Reproduces `cctbx.miller.binning::init_limits`, which is what
    `setup_binner(n_bins=...)` builds: shell edges are equally spaced in
    reciprocal-space *volume*, i.e. in `d*^3`, and are stored as `d*^2`.
    The outer edges are pushed out by `relative_tolerance` times the
    `d*^2` span so that no observation falls outside the first or last
    shell.

    Args:
        d: `(N,)` resolution in Angstrom.
        n_bins: number of shells.
        d_max: low-resolution limit. Defaults to `max(d)`.
        d_min: high-resolution limit. Defaults to `min(d)`.
        relative_tolerance: outward expansion of the outer edges.

    Returns:
        A `ResolutionBins` carrying the per-reflection shell index and
        the `d*^2` edges.

    Raises:
        ValueError: if `n_bins` is not positive or `d` is empty.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if d.numel() == 0:
        raise ValueError("cannot bin an empty resolution array")

    dss = 1.0 / (d.to(DTYPE) ** 2)
    lo = float(dss.min()) if d_max is None else 1.0 / (d_max * d_max)
    hi = float(dss.max()) if d_min is None else 1.0 / (d_min * d_min)

    # cctbx leaves d_star_sq_min at zero when the data span is degenerate.
    if lo == hi:
        lo = 0.0

    span = hi - lo
    hi += span * relative_tolerance
    lo -= span * relative_tolerance
    lo = max(lo, 0.0)

    r_lo = math.sqrt(lo)
    r_hi = math.sqrt(hi)
    four_pi = 4.0 * math.pi
    volume_low = four_pi / 3.0 * r_lo**3
    volume_per_bin = (four_pi / 3.0 * r_hi**3 - volume_low) / n_bins

    limits = [lo]
    for i_bin in range(1, n_bins):
        limits.append(
            ((volume_low + i_bin * volume_per_bin) * 3.0 / four_pi) ** (2 / 3)
        )
    limits.append(hi)
    limit_t = torch.tensor(limits, dtype=DTYPE)

    # cctbx assigns to the first shell whose upper edge exceeds d*^2,
    # i.e. shells are closed below and open above.
    bin_id = torch.bucketize(dss, limit_t[1:-1], right=True)
    return ResolutionBins(bin_id=bin_id.to(IDTYPE), limits=limit_t)


@dataclass
class _MergeReduction:
    """Per-group output of the DIALS weighted merge.

    Attributes:
        mean: `(G,)` merged intensity.
        sigma: `(G,)` merged standard deviation.
        mult: `(G,)` observation count.
        sum_abs_dev: `(G,)` `sum |I_i - <I>|`.
        sum_intensity: `(G,)` `sum I_i`, signed.
    """

    mean: torch.Tensor
    sigma: torch.Tensor
    mult: torch.Tensor
    sum_abs_dev: torch.Tensor
    sum_intensity: torch.Tensor


def _merge_equivalents(
    intensity: torch.Tensor,
    sigma: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
    use_internal_variance: bool = False,
) -> _MergeReduction:
    """Vectorised `cctbx::miller::merge_equivalents_obs`.

    The merged intensity is the inverse-variance weighted mean and the
    merged sigma is the propagated `1 / sqrt(sum 1 / sigma_i^2)`. A zero
    input sigma is replaced by `1e-3`, as cctbx does, rather than
    producing an infinite weight.

    With `use_internal_variance` the propagated sigma is replaced by the
    standard error implied by the observed scatter wherever that is the
    larger of the two. That estimate is the weighted sample variance
    under the unbiased reliability-weight denominator
    `V1 - V2 / V1`, divided by `n`:

        `sigma_int^2 = sum w_i (I_i - <I>)^2 / (n (V1 - V2 / V1))`

    with `V1 = sum w_i` and `V2 = sum w_i^2`. For equal weights this
    collapses to the familiar `sum (I_i - <I>)^2 / (n (n - 1))`. DIALS
    turns this off for merging statistics but `cc_half_sigma_tau`
    depends on it.

    Args:
        intensity: `(N,)` observed intensity.
        sigma: `(N,)` observed standard deviation.
        group_id: `(N,)` symmetry group index.
        n_groups: number of groups.
        use_internal_variance: replace the propagated sigma by the
            scatter-derived one where the latter is larger.

    Returns:
        The per-group reduction.
    """
    s = torch.where(sigma == 0, torch.full_like(sigma, _SIGMA_FLOOR), sigma)
    w = 1.0 / (s * s)

    sum_w = _group_sum(w, group_id, n_groups)
    sum_wi = _group_sum(w * intensity, group_id, n_groups)
    mult = _group_count(group_id, n_groups)

    filled = mult > 0
    safe_w = torch.where(filled, sum_w, torch.ones_like(sum_w))
    mean = torch.where(filled, sum_wi / safe_w, torch.zeros_like(sum_w))
    sigma_out = torch.where(
        filled,
        torch.sqrt(1.0 / safe_w),
        torch.zeros_like(sum_w),
    )

    dev = intensity - mean[group_id]
    sum_abs_dev = _group_sum(torch.abs(dev), group_id, n_groups)
    sum_intensity = _group_sum(intensity, group_id, n_groups)

    if use_internal_variance:
        many = mult > 1
        sum_ww = _group_sum(w * w, group_id, n_groups)
        sum_wdd = _group_sum(w * dev * dev, group_id, n_groups)
        eff = safe_w - sum_ww / safe_w
        denom = mult.to(DTYPE) * torch.where(
            many & (eff > 0), eff, torch.ones_like(eff)
        )
        sigma_int = torch.sqrt(sum_wdd / denom)
        sigma_out = torch.where(
            many & (eff > 0) & (sigma_int > sigma_out), sigma_int, sigma_out
        )

    return _MergeReduction(
        mean=mean,
        sigma=sigma_out,
        mult=mult,
        sum_abs_dev=sum_abs_dev,
        sum_intensity=sum_intensity,
    )


def _r_factors(red: _MergeReduction) -> tuple[float, float, float]:
    """R-merge, R-meas and R-pim from a merge reduction.

    Only groups with multiplicity greater than one contribute, to either
    numerator or denominator, and the denominator is the signed sum of
    intensities.

    Args:
        red: output of `_merge_equivalents`.

    Returns:
        `(r_merge, r_meas, r_pim)`.
    """
    many = red.mult > 1
    if not bool(many.any()):
        return 0.0, 0.0, 0.0
    n = red.mult[many].to(DTYPE)
    dev = red.sum_abs_dev[many]
    den = float(red.sum_intensity[many].sum())
    if den == 0:
        return 0.0, 0.0, 0.0
    return (
        float(dev.sum()) / den,
        float((torch.sqrt(n / (n - 1.0)) * dev).sum()) / den,
        float((torch.sqrt(1.0 / (n - 1.0)) * dev).sum()) / den,
    )


@dataclass
class _HalfSplit:
    """Weighted means of a random half-dataset split.

    Attributes:
        mean_1: `(G,)` weighted mean of the first half.
        mean_2: `(G,)` weighted mean of the second half.
        sum_w_1: `(G,)` weight sum of the first half.
        sum_w_2: `(G,)` weight sum of the second half.
        valid: `(G,)` bool, true where the group had at least two
            observations and so contributed to both halves.
    """

    mean_1: torch.Tensor
    mean_2: torch.Tensor
    sum_w_1: torch.Tensor
    sum_w_2: torch.Tensor
    valid: torch.Tensor


def _split_halves(
    intensity: torch.Tensor,
    weight: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
    generator: torch.Generator,
) -> _HalfSplit:
    """Split each group's observations at random into two halves.

    Mirrors `cctbx::miller::split_unmerged::process_group`: a group of
    `n` observations contributes `floor(n / 2)` to the first half, plus
    one more with probability one half when `n` is odd; groups with
    `n < 2` are dropped. The subset is drawn uniformly, here by ranking
    a uniform variate within each group rather than by cctbx's partial
    Fisher-Yates shuffle. The two are distributionally identical but
    consume randomness differently, so individual splits do not match.

    Args:
        intensity: `(N,)` observed intensity.
        weight: `(N,)` weight for the within-half mean.
        group_id: `(N,)` group index.
        n_groups: number of groups.
        generator: torch RNG.

    Returns:
        The half means and their weight sums.
    """
    counts = _group_count(group_id, n_groups)

    # Stable sort by group of a random permutation leaves the members of
    # each group contiguous and in uniformly random order.
    u = torch.rand(intensity.shape[0], generator=generator, dtype=DTYPE)
    shuffled = torch.argsort(u)
    perm = shuffled[torch.argsort(group_id[shuffled], stable=True)]

    g = group_id[perm]
    starts = torch.zeros_like(counts)
    starts[1:] = torch.cumsum(counts, 0)[:-1]
    rank = torch.arange(g.shape[0], dtype=IDTYPE) - starts[g]

    n_first = torch.div(counts, 2, rounding_mode="floor")
    odd = (counts % 2) == 1
    coin = torch.rand(n_groups, generator=generator, dtype=DTYPE) < 0.5
    n_first = n_first + (odd & coin).to(IDTYPE)

    in_first = rank < n_first[g]
    i = intensity[perm]
    w = weight[perm]

    sum_w_1 = _group_sum(w * in_first, g, n_groups)
    sum_w_2 = _group_sum(w * ~in_first, g, n_groups)
    sum_wi_1 = _group_sum(w * i * in_first, g, n_groups)
    sum_wi_2 = _group_sum(w * i * ~in_first, g, n_groups)

    valid = counts > 1
    safe_1 = torch.where(sum_w_1 > 0, sum_w_1, torch.ones_like(sum_w_1))
    safe_2 = torch.where(sum_w_2 > 0, sum_w_2, torch.ones_like(sum_w_2))
    return _HalfSplit(
        mean_1=sum_wi_1 / safe_1,
        mean_2=sum_wi_2 / safe_2,
        sum_w_1=sum_w_1,
        sum_w_2=sum_w_2,
        valid=valid,
    )


@dataclass
class HalfDatasetCC:
    """A half-dataset correlation and its Monte-Carlo spread.

    Attributes:
        value: mean correlation over `n_repeats` random splits.
        std: sample standard deviation over repeats, `nan` for a single
            repeat. This is the spread of one split, not the standard
            error of `value`.
        n_refl: number of groups (or Bijvoet pairs) entering the
            correlation.
        n_repeats: number of splits averaged.
    """

    value: float
    std: float
    n_refl: int
    n_repeats: int

    def __float__(self) -> float:
        return self.value


def _scaled(
    refl: Reflections, apply_scale: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(intensity, sigma)`, optionally divided by the scale.

    DIALS forms its merging-statistics array as
    `intensity.scale.value / inverse_scale_factor`, with the sigma
    scaled to match.

    Args:
        refl: observations.
        apply_scale: divide by `refl.scale`.

    Returns:
        `(intensity, sigma)` tensors.
    """
    if not apply_scale:
        return refl.intensity, refl.sigma
    return refl.intensity / refl.scale, refl.sigma / refl.scale


def cc_half_random(
    refl: Reflections,
    amap: AsuMap,
    anomalous: bool = False,
    seed: int = 0,
    n_repeats: int = 1,
    apply_scale: bool = True,
) -> HalfDatasetCC:
    """Half-dataset CC1/2 from random split of each group.

    Reproduces `cctbx.miller.compute_cc_one_half`: observations with a
    non-positive sigma are dropped, each group with at least two
    survivors is split at random into two halves, each half is collapsed
    to an inverse-variance weighted mean, and the two resulting vectors
    are correlated.

    The split is random, and cctbx draws it from a scitbx Mersenne
    twister that torch cannot reproduce, so this value will *not* match
    a `dials.scale` log exactly. Raise `n_repeats` to average the split
    noise away and read `HalfDatasetCC.std` to see how large it is.
    `cc_half_sigma_tau` is the deterministic alternative.

    Args:
        refl: unmerged observations.
        amap: symmetry reduction of `refl.hkl`.
        anomalous: keep Friedel mates in separate groups.
        seed: RNG seed.
        n_repeats: number of independent splits to average.
        apply_scale: divide intensities by `refl.scale` first.

    Returns:
        The averaged correlation with its spread over repeats.
    """
    intensity, sigma = _scaled(refl, apply_scale)
    group_id, n_groups = amap.groups(anomalous)

    keep = sigma > 0
    intensity = intensity[keep]
    sigma = sigma[keep]
    group_id = group_id[keep]
    weight = 1.0 / (sigma * sigma)

    return _repeat_cc(
        lambda gen: _cc_from_split(
            _split_halves(intensity, weight, group_id, n_groups, gen)
        ),
        seed,
        n_repeats,
    )


def _cc_from_split(split: _HalfSplit) -> tuple[float, int]:
    """Correlate the two halves over the groups that have both."""
    v = split.valid
    return _pearson(split.mean_1[v], split.mean_2[v]), int(v.sum())


def _repeat_cc(
    trial: Callable[[torch.Generator], tuple[float, int]],
    seed: int,
    n_repeats: int,
) -> HalfDatasetCC:
    """Average a random-split statistic over repeats.

    Args:
        trial: callable taking a generator and returning
            `(correlation, n_refl)`.
        seed: RNG seed.
        n_repeats: number of trials.

    Returns:
        Mean, sample standard deviation and reflection count.

    Raises:
        ValueError: if `n_repeats` is not positive.
    """
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    generator = torch.Generator().manual_seed(seed)
    values: list[float] = []
    n_refl = 0
    for _ in range(n_repeats):
        cc, n_refl = trial(generator)
        values.append(cc)
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return HalfDatasetCC(float("nan"), float("nan"), n_refl, n_repeats)
    std = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
    return HalfDatasetCC(float(finite.mean()), std, n_refl, n_repeats)


def sigma_tau_merge(
    refl: Reflections,
    amap: AsuMap,
    anomalous: bool = False,
    apply_scale: bool = True,
) -> MergedData:
    """Merge with unit sigmas, as `cc_one_half_sigma_tau` requires.

    cctbx computes the sigma-tau CC1/2 from a merge in which every
    observation is given `sigma = 1` and the merged sigma is taken from
    the internal scatter. The unit sigmas make the merged sigma a pure
    function of the spread within a group, which is what the estimator
    needs; they also change the merge weight, so this is not the same
    merged array as `merge.merge`.

    Args:
        refl: unmerged observations.
        amap: symmetry reduction of `refl.hkl`.
        anomalous: keep Friedel mates in separate groups.
        apply_scale: divide intensities by `refl.scale` first.

    Returns:
        Merged data whose `sigma` holds the internal-variance estimate.
    """
    intensity, _ = _scaled(refl, apply_scale)
    group_id, n_groups = amap.groups(anomalous)

    red = _merge_equivalents(
        intensity,
        torch.ones_like(intensity),
        group_id,
        n_groups,
        use_internal_variance=True,
    )
    return _as_merged(red, refl, amap, group_id, n_groups)


def cc_half_sigma_tau(merged: MergedData) -> tuple[float, int]:
    """Deterministic CC1/2 by the sigma-tau method.

    From Assmann, Brehm and Diederichs, J. Appl. Cryst. 49, 1021 (2016):
    with `var_y` the sample variance of the merged intensities and
    `var_e` twice the mean internal variance,

        `CC1/2 = (var_y - var_e / 2) / (var_y + var_e / 2)`

    evaluated over reflections of multiplicity greater than one. No
    random numbers are involved, so unlike `cc_half_random` this
    reproduces cctbx exactly.

    Args:
        merged: output of `sigma_tau_merge`, i.e. a merge performed with
            unit sigmas and internal variance.

    Returns:
        `(cc_one_half, n_refl)` where `n_refl` counts the reflections of
        multiplicity greater than one.
    """
    sel = merged.multiplicity > 1
    n_refl = int(sel.sum())
    if n_refl <= 1:
        return 0.0, n_refl
    i = merged.intensity[sel]
    var_y = float(((i - i.mean()) ** 2).sum() / (n_refl - 1))
    var_e = 2.0 * float((merged.sigma[sel] ** 2).mean())
    return (var_y - 0.5 * var_e) / (var_y + 0.5 * var_e), n_refl


def cc_star(cc_half: float) -> float:
    """Convert CC1/2 to CC*.

    `CC* = sign(CC1/2) sqrt(2 |CC1/2| / (1 + CC1/2))`, the correlation
    of the merged data with the (unobservable) true intensities.

    Args:
        cc_half: half-dataset correlation.

    Returns:
        CC*; zero when `cc_half` is zero, `-inf` at the `-1` pole, and
        `nan` if `cc_half` is `nan`.
    """
    if math.isnan(cc_half):
        return float("nan")
    if cc_half == 0:
        return 0.0
    if cc_half < -0.999:
        return float("-inf")
    sign = -1.0 if cc_half < 0 else 1.0
    return sign * math.sqrt(2.0 * abs(cc_half) / (1.0 + cc_half))


def r_split(
    refl: Reflections,
    amap: AsuMap,
    anomalous: bool = False,
    seed: int = 0,
    n_repeats: int = 1,
    apply_scale: bool = True,
) -> HalfDatasetCC:
    """R_split over a random half-dataset split.

    White et al., J. Appl. Cryst. 45, 335 (2012), in the form DIALS
    uses:

        `R_split = 2^-1/2 sum|I_1 - I_2| / (0.5 sum(I_1 + I_2))`

    DIALS splits observations rather than whole images, reusing the
    inverse-variance weighted split of CC1/2, and reports this only when
    `additional_stats=True`. There is no cctbx equivalent to compare
    against, and like CC1/2 the value carries split noise.

    Args:
        refl: unmerged observations.
        amap: symmetry reduction of `refl.hkl`.
        anomalous: keep Friedel mates in separate groups.
        seed: RNG seed.
        n_repeats: number of independent splits to average.
        apply_scale: divide intensities by `refl.scale` first.

    Returns:
        The averaged R_split with its spread over repeats. `value` is
        `-1` when the denominator vanishes, matching DIALS.
    """
    intensity, sigma = _scaled(refl, apply_scale)
    group_id, n_groups = amap.groups(anomalous)

    keep = sigma > 0
    intensity = intensity[keep]
    sigma = sigma[keep]
    group_id = group_id[keep]
    weight = 1.0 / (sigma * sigma)

    def trial(gen: torch.Generator) -> tuple[float, int]:
        split = _split_halves(intensity, weight, group_id, n_groups, gen)
        v = split.valid
        i1 = split.mean_1[v]
        i2 = split.mean_2[v]
        den = 0.5 * float((i1 + i2).sum())
        if den == 0:
            return -1.0, int(v.sum())
        num = float(torch.abs(i1 - i2).sum())
        return num / (math.sqrt(2.0) * den), int(v.sum())

    return _repeat_cc(trial, seed, n_repeats)


def weighted_cc_half(
    refl: Reflections,
    amap: AsuMap,
    anomalous: bool = False,
    seed: int = 0,
    n_repeats: int = 1,
    apply_scale: bool = True,
) -> tuple[HalfDatasetCC, float]:
    """CC1/2 weighted by the inverse joint variance of the two halves.

    `ExtendedDatasetStatistics.weighted_cchalf`. Each half mean carries
    the propagated variance `1 / sum(w)` of its inverse-variance
    weighted mean; a reflection then enters the Pearson correlation with
    weight `1 / (v_1 + v_2)`, normalised to sum to one. The
    accompanying sample size is Kish's effective sample size
    `n_eff = 1 / sum(w_norm^2)` (Kish, Survey Sampling, 1965), which is
    the count a significance test should use in place of the raw number
    of reflections.

    Args:
        refl: unmerged observations.
        amap: symmetry reduction of `refl.hkl`.
        anomalous: keep Friedel mates in separate groups.
        seed: RNG seed.
        n_repeats: number of independent splits to average.
        apply_scale: divide intensities by `refl.scale` first.

    Returns:
        `(cc, n_eff)` with `cc` averaged over repeats and `n_eff` from
        the last split.
    """
    intensity, sigma = _scaled(refl, apply_scale)
    group_id, n_groups = amap.groups(anomalous)

    keep = sigma > 0
    intensity = intensity[keep]
    sigma = sigma[keep]
    group_id = group_id[keep]
    weight = 1.0 / (sigma * sigma)

    n_eff = float("nan")

    def trial(gen: torch.Generator) -> tuple[float, int]:
        nonlocal n_eff
        split = _split_halves(intensity, weight, group_id, n_groups, gen)
        v = split.valid
        if int(v.sum()) < 2:
            return float("nan"), int(v.sum())
        x = split.mean_1[v]
        y = split.mean_2[v]
        joint_w = 1.0 / (1.0 / split.sum_w_1[v] + 1.0 / split.sum_w_2[v])
        norm_w = joint_w / joint_w.sum()
        dx = x - (x * norm_w).sum()
        dy = y - (y * norm_w).sum()
        sxy = (dx * dy * norm_w).sum()
        sx = (dx * dx * norm_w).sum()
        sy = (dy * dy * norm_w).sum()
        n_eff = float(1.0 / (norm_w * norm_w).sum())
        if sx == 0 or sy == 0:
            return float("nan"), int(v.sum())
        return float(sxy / torch.sqrt(sx * sy)), int(v.sum())

    return _repeat_cc(trial, seed, n_repeats), n_eff


def cc_anom(
    refl: Reflections,
    amap: AsuMap,
    seed: int = 0,
    n_repeats: int = 1,
    apply_scale: bool = True,
) -> HalfDatasetCC:
    """Half-dataset correlation of anomalous differences, CC(anom).

    This is DIALS' `cc_ano` column and the `Anomalous correlation` line
    of its summary: `cctbx.miller.array.half_dataset_anomalous_
    correlation`. Observations are grouped keeping Friedel mates apart
    and split at random into two halves with *unweighted* means; each
    half then yields `dI = I(+) - I(-)` for every acentric reflection
    whose two mates both survived the split, and the two `dI` vectors
    are correlated.

    Centric reflections are excluded, since `I(+)` and `I(-)` of a
    centric are symmetry equivalent and carry no anomalous difference.
    `build_asu_map` already places every observation of a centric in one
    hemisphere, so they have no mate to pair against and the explicit
    filter here is a guard rather than a correction. The anomalous
    grouping it produces was checked to be in one-to-one correspondence
    with `cctbx.miller.set.map_to_asu(anomalous_flag=True)`.

    Args:
        refl: unmerged observations.
        amap: symmetry reduction of `refl.hkl`.
        seed: RNG seed.
        n_repeats: number of independent splits to average.
        apply_scale: divide intensities by `refl.scale` first.

    Returns:
        The averaged CC(anom) with its spread over repeats; `n_refl`
        counts contributing Bijvoet pairs.
    """
    intensity, sigma = _scaled(refl, apply_scale)

    keep = sigma > 0
    intensity = intensity[keep]
    group_anom = amap.group_id[keep]
    group_laue = amap.group_id_laue[keep]
    plus = amap.plus[keep]
    centric = amap.centric[keep]
    n_anom = amap.n_groups
    n_laue = amap.n_groups_laue

    # Per-group attributes, gathered by scattering the observation-level
    # ones; every group has at least one member by construction.
    laue_of = torch.zeros(n_anom, dtype=IDTYPE)
    laue_of[group_anom] = group_laue
    plus_of = torch.zeros(n_anom, dtype=torch.bool)
    plus_of[group_anom] = plus
    centric_laue = torch.zeros(n_laue, dtype=torch.bool)
    centric_laue[group_laue] = centric

    anom_ids = torch.arange(n_anom, dtype=IDTYPE)
    present = torch.zeros(n_anom, dtype=torch.bool)
    present[group_anom] = True

    plus_group = torch.full((n_laue,), -1, dtype=IDTYPE)
    minus_group = torch.full((n_laue,), -1, dtype=IDTYPE)
    sel_p = present & plus_of
    sel_m = present & ~plus_of
    plus_group[laue_of[sel_p]] = anom_ids[sel_p]
    minus_group[laue_of[sel_m]] = anom_ids[sel_m]

    paired = (plus_group >= 0) & (minus_group >= 0) & ~centric_laue
    p_idx = plus_group[paired]
    m_idx = minus_group[paired]
    weight = torch.ones_like(intensity)

    def trial(gen: torch.Generator) -> tuple[float, int]:
        split = _split_halves(intensity, weight, group_anom, n_anom, gen)
        both = split.valid[p_idx] & split.valid[m_idx]
        p = p_idx[both]
        m = m_idx[both]
        dano_1 = split.mean_1[p] - split.mean_1[m]
        dano_2 = split.mean_2[p] - split.mean_2[m]
        return _pearson(dano_1, dano_2), int(both.sum())

    return _repeat_cc(trial, seed, n_repeats)


@dataclass
class CompleteSet:
    """Enumerated symmetry-unique reflections used as a denominator.

    Attributes:
        hkl: `(M, 3)` Miller indices.
        d: `(M,)` resolution in Angstrom.
        centric: `(M,)` bool.
    """

    hkl: torch.Tensor
    d: torch.Tensor
    centric: torch.Tensor

    def in_range(self, d_max: float, d_min: float) -> torch.Tensor:
        """Boolean mask for `d_min <= d <= d_max`, as cctbx filters."""
        return (self.d >= d_min) & (self.d <= d_max)


def build_complete_set(
    cell: tuple[float, float, float, float, float, float],
    spacegroup: str | int | gemmi.SpaceGroup,
    d_min: float,
    d_max: float | None = None,
    anomalous: bool = False,
) -> CompleteSet:
    """Enumerate the complete set and tag it with `d` and centricity.

    Wraps `symmetry.complete_set`, which is the completeness denominator
    DIALS uses via `miller.set.complete_set`. cctbx builds it down to
    `d_min * (1 - 1e-6)`; that slack is applied here so a reflection
    sitting exactly on the high-resolution limit is not lost to rounding.

    Args:
        cell: unit cell `(a, b, c, alpha, beta, gamma)`.
        spacegroup: space group of the crystal.
        d_min: high resolution limit in Angstrom.
        d_max: low resolution limit, or `None` for no limit.
        anomalous: keep Friedel mates as distinct reflections.

    Returns:
        The enumerated set.
    """
    sg = as_spacegroup(spacegroup)
    # `symmetry.complete_set` forwards d_max straight to gemmi, which
    # spells "no low-resolution limit" as 0.0 rather than None.
    hkl = complete_set(
        cell,
        sg,
        d_min * (1.0 - _RELATIVE_TOLERANCE),
        0.0 if d_max is None else d_max,
        anomalous=anomalous,
    )
    ops = sg.operations()
    h_np = hkl.numpy()
    centric = torch.from_numpy(
        np.array(
            [
                ops.is_reflection_centric((int(h), int(k), int(ll)))
                for h, k, ll in h_np
            ],
            dtype=bool,
        )
    )
    return CompleteSet(
        hkl=hkl,
        d=1.0 / torch.sqrt(d_star_sq(hkl, cell)),
        centric=centric,
    )


def completeness(
    merged: MergedData,
    cell: tuple[float, float, float, float, float, float],
    spacegroup: str | int | gemmi.SpaceGroup,
    d_min: float,
    d_max: float,
    anomalous: bool = False,
    complete: CompleteSet | None = None,
) -> float:
    """Fraction of the complete set that was observed.

    Args:
        merged: merged reflections, already restricted to the shell.
        cell: unit cell `(a, b, c, alpha, beta, gamma)`.
        spacegroup: space group of the crystal.
        d_min: high resolution limit of the shell.
        d_max: low resolution limit of the shell.
        anomalous: count Friedel mates separately.
        complete: precomputed complete set covering at least
            `[d_min, d_max]`. Enumerated on demand when omitted, which
            is expensive to repeat per shell.

    Returns:
        Observed fraction, capped at one; zero if the denominator is
        empty.
    """
    if complete is None:
        complete = build_complete_set(
            cell, spacegroup, d_min, d_max, anomalous=anomalous
        )
    n_expected = int(complete.in_range(d_max, d_min).sum())
    if n_expected == 0:
        return 0.0
    return min(len(merged) / n_expected, 1.0)


def anomalous_completeness(
    merged: MergedData,
    cell: tuple[float, float, float, float, float, float],
    spacegroup: str | int | gemmi.SpaceGroup,
    d_min: float,
    d_max: float,
    complete: CompleteSet | None = None,
) -> float:
    """Fraction of acentric reflections observed as complete Bijvoet pairs.

    Matches `cctbx.miller.array.anomalous_completeness`: the numerator
    counts reflections for which both `I(+)` and `I(-)` are present, and
    the denominator is the number of *acentric* reflections in the
    non-anomalous complete set.

    Args:
        merged: anomalous merged reflections restricted to the shell.
        cell: unit cell `(a, b, c, alpha, beta, gamma)`.
        spacegroup: space group of the crystal.
        d_min: high resolution limit of the shell.
        d_max: low resolution limit of the shell.
        complete: precomputed *non-anomalous* complete set.

    Returns:
        Observed fraction, capped at one; zero if there are no acentric
        reflections in range.
    """
    if complete is None:
        complete = build_complete_set(
            cell, spacegroup, d_min, d_max, anomalous=False
        )
    in_range = complete.in_range(d_max, d_min)
    n_acentric = int((in_range & ~complete.centric).sum())
    if n_acentric == 0:
        return 0.0

    acentric = ~merged.centric
    key = torch.where(merged.plus.unsqueeze(1), merged.hkl, -merged.hkl)[
        acentric
    ]
    plus = merged.plus[acentric]
    _, inverse, counts = torch.unique(
        key, dim=0, return_inverse=True, return_counts=True
    )
    n_plus = torch.zeros(counts.shape[0], dtype=IDTYPE)
    n_plus.index_add_(0, inverse, plus.to(IDTYPE))
    n_pairs = int(((n_plus > 0) & (counts - n_plus > 0)).sum())
    return min(n_pairs / n_acentric, 1.0)


@dataclass
class IOverSigma:
    """The three signal-to-noise ratios DIALS distinguishes.

    Attributes:
        i_mean: mean merged intensity.
        sigi_mean: mean merged sigma.
        i_over_sigma_mean: mean of the *ratio* on the merged array. This
            is the `<I/sI>` column of a `dials.scale` table.
        i_mean_over_sigi_mean: *ratio of the means* on the merged array,
            which is systematically larger because the mean of a ratio
            is dominated by weak reflections.
        unmerged_i_over_sigma_mean: mean of the ratio before merging,
            smaller than the merged value by roughly the square root of
            the multiplicity.
    """

    i_mean: float
    sigi_mean: float
    i_over_sigma_mean: float
    i_mean_over_sigi_mean: float
    unmerged_i_over_sigma_mean: float


def i_over_sigma(
    merged: MergedData,
    refl: Reflections | None = None,
    apply_scale: bool = True,
) -> IOverSigma:
    """Signal-to-noise summaries of a merged array.

    Args:
        merged: merged reflections.
        refl: unmerged observations, needed only for
            `unmerged_i_over_sigma_mean`.
        apply_scale: divide unmerged intensities by `refl.scale`.

    Returns:
        The three ratios plus the two means they are built from.
    """
    if len(merged) == 0:
        nan = float("nan")
        return IOverSigma(nan, nan, nan, nan, nan)

    i_mean = float(merged.intensity.mean())
    sigi_mean = float(merged.sigma.mean())
    pos = merged.sigma > 0
    ratio = float("nan")
    if bool(pos.any()):
        ratio = float((merged.intensity[pos] / merged.sigma[pos]).mean())

    unmerged = float("nan")
    if refl is not None and len(refl) > 0:
        intensity, sigma = _scaled(refl, apply_scale)
        keep = sigma > 0
        if bool(keep.any()):
            unmerged = float((intensity[keep] / sigma[keep]).mean())

    return IOverSigma(
        i_mean=i_mean,
        sigi_mean=sigi_mean,
        i_over_sigma_mean=ratio,
        i_mean_over_sigi_mean=i_mean / sigi_mean
        if sigi_mean
        else float("nan"),
        unmerged_i_over_sigma_mean=unmerged,
    )


def n_obs(refl: Reflections) -> int:
    """Number of unmerged observations."""
    return len(refl)


def n_unique(merged: MergedData) -> int:
    """Number of merged reflections."""
    return len(merged)


def multiplicity(merged: MergedData) -> float:
    """Mean number of observations contributing to a merged reflection."""
    if len(merged) == 0:
        return 0.0
    return float(merged.multiplicity.to(DTYPE).mean())


@dataclass
class ShellStats:
    """Merging statistics for one resolution shell, or overall.

    Attributes:
        d_max: low resolution limit in Angstrom.
        d_min: high resolution limit in Angstrom.
        n_obs: number of unmerged observations.
        n_uniq: number of merged reflections.
        multiplicity: mean multiplicity.
        completeness: observed fraction of the complete set.
        anom_completeness: fraction of acentric reflections with both
            Friedel mates, or `None` for a non-anomalous run.
        i_mean: mean merged intensity.
        sigi_mean: mean merged sigma.
        i_over_sigma_mean: mean of merged `I / sigma`.
        i_mean_over_sigi_mean: ratio of the merged means.
        unmerged_i_over_sigma_mean: mean of unmerged `I / sigma`.
        r_merge: `sum|I - <I>| / sum I`.
        r_meas: multiplicity-corrected `r_merge`.
        r_pim: precision-indicating merging R-factor.
        cc_half: random half-dataset CC1/2.
        cc_half_std: Monte-Carlo spread of `cc_half` over repeats.
        cc_half_n_refl: reflections entering `cc_half`.
        cc_half_sigma_tau: deterministic sigma-tau CC1/2.
        cc_half_sigma_tau_n_refl: reflections entering the above.
        cc_star: CC* from `cc_half_sigma_tau`.
        cc_anom: half-dataset CC of anomalous differences.
        cc_anom_std: Monte-Carlo spread of `cc_anom`.
        cc_anom_n_pairs: Bijvoet pairs entering `cc_anom`.
        r_split: half-dataset R_split.
        weighted_cc_half: inverse-joint-variance weighted CC1/2.
        weighted_cc_half_neff: Kish effective sample size for the above.
    """

    d_max: float
    d_min: float
    n_obs: int
    n_uniq: int
    multiplicity: float
    completeness: float
    anom_completeness: float | None
    i_mean: float
    sigi_mean: float
    i_over_sigma_mean: float
    i_mean_over_sigi_mean: float
    unmerged_i_over_sigma_mean: float
    r_merge: float
    r_meas: float
    r_pim: float
    cc_half: float
    cc_half_std: float
    cc_half_n_refl: int
    cc_half_sigma_tau: float
    cc_half_sigma_tau_n_refl: int
    cc_star: float
    cc_anom: float
    cc_anom_std: float
    cc_anom_n_pairs: int
    r_split: float
    weighted_cc_half: float
    weighted_cc_half_neff: float


_HEADER = (
    " d_max  d_min   #obs  #uniq  mult.  %comp       <I>  <I/sI> "
    "  r_mrg  r_meas   r_pim   cc1/2  cc1/2t  cc_ano"
)
_ROW = (
    "{0:6.2f} {1:6.2f} {2:6d} {3:6d} {4:6.2f} {5:6.2f} {6:9.1f} {7:7.1f} "
    "{8:7.3f} {9:7.3f} {10:7.3f} {11:7.3f} {12:7.3f} {13:7.3f}"
)


@dataclass
class MergingStats:
    """Overall and per-shell merging statistics.

    Attributes:
        overall: statistics over the whole resolution range.
        shells: per-shell statistics, low resolution first.
        bins: the shell definition the table was built on.
        anomalous: whether Friedel mates were kept apart when merging.
        n_repeats: number of random splits averaged for the stochastic
            statistics.
    """

    overall: ShellStats
    shells: list[ShellStats]
    bins: ResolutionBins
    anomalous: bool
    n_repeats: int

    def to_table(self) -> str:
        """Render a `dials.scale`-like resolution-shell table.

        Returns:
            The formatted table, overall statistics on the last line.
        """
        lines = [_HEADER]
        for shell in [*self.shells, self.overall]:
            lines.append(
                _ROW.format(
                    shell.d_max,
                    shell.d_min,
                    shell.n_obs,
                    shell.n_uniq,
                    shell.multiplicity,
                    100.0 * shell.completeness,
                    shell.i_mean,
                    shell.i_over_sigma_mean,
                    shell.r_merge,
                    shell.r_meas,
                    shell.r_pim,
                    shell.cc_half,
                    shell.cc_half_sigma_tau,
                    shell.cc_anom,
                )
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_table()


def _as_merged(
    red: _MergeReduction,
    refl: Reflections,
    amap: AsuMap,
    group_id: torch.Tensor,
    n_groups: int,
    d: torch.Tensor | None = None,
) -> MergedData:
    """Assemble a `MergedData` from a reduction, dropping empty groups.

    Args:
        red: per-group reduction covering all `n_groups`.
        refl: the observations that were reduced.
        amap: symmetry reduction of `refl.hkl`.
        group_id: `(N,)` group index used for the reduction.
        n_groups: number of groups.
        d: `(N,)` resolution to carry over, defaulting to `refl.d`.
            Pass the cell-derived value to stay consistent with the
            binning, which does not use the `d` column.

    Returns:
        One row per non-empty group.
    """
    rep = torch.zeros(n_groups, dtype=IDTYPE)
    rep[group_id] = torch.arange(group_id.shape[0], dtype=IDTYPE)
    occupied = red.mult > 0
    idx = rep[occupied]
    return MergedData(
        hkl=amap.asu_hkl[idx],
        intensity=red.mean[occupied],
        sigma=red.sigma[occupied],
        multiplicity=red.mult[occupied],
        d=(refl.d if d is None else d)[idx],
        centric=amap.centric[idx],
        plus=amap.plus[idx],
    )


@dataclass
class _Context:
    """Everything `_shell_stats` needs that does not vary by shell."""

    cell: tuple[float, float, float, float, float, float]
    spacegroup: str | int | gemmi.SpaceGroup
    anomalous: bool
    complete: CompleteSet
    complete_anom: CompleteSet | None
    seed: int
    n_repeats: int
    apply_scale: bool
    r_factor_fn: (
        Callable[[Reflections, AsuMap, bool], Mapping[str, float]] | None
    )


def _select_asu_map(amap: AsuMap, sel: torch.Tensor) -> AsuMap:
    """Row-select an `AsuMap`, keeping the global group numbering.

    Args:
        amap: the full map.
        sel: boolean or integer row selection.

    Returns:
        A map over the selected rows. `n_groups` is unchanged, so group
        ids stay comparable across shells and empty groups simply
        contribute nothing to a reduction.
    """
    return AsuMap(
        asu_hkl=amap.asu_hkl[sel],
        isym=amap.isym[sel],
        plus=amap.plus[sel],
        centric=amap.centric[sel],
        epsilon=amap.epsilon[sel],
        sys_absent=amap.sys_absent[sel],
        group_id=amap.group_id[sel],
        group_id_laue=amap.group_id_laue[sel],
        n_groups=amap.n_groups,
        n_groups_laue=amap.n_groups_laue,
    )


def _shell_stats(
    refl: Reflections,
    amap: AsuMap,
    d: torch.Tensor,
    d_max: float,
    d_min: float,
    ctx: _Context,
) -> ShellStats:
    """Compute every statistic for one already-selected set of rows.

    Args:
        refl: observations of this shell.
        amap: symmetry reduction of `refl.hkl`.
        d: `(N,)` cell-derived resolution of `refl`.
        d_max: low resolution edge of the shell.
        d_min: high resolution edge of the shell.
        ctx: shell-invariant settings.

    Returns:
        The shell's statistics.
    """
    intensity, sigma = _scaled(refl, ctx.apply_scale)
    group_id, n_groups = amap.groups(ctx.anomalous)

    red = _merge_equivalents(intensity, sigma, group_id, n_groups)
    merged = _as_merged(red, refl, amap, group_id, n_groups, d=d)

    if ctx.r_factor_fn is None:
        r_merge, r_meas, r_pim = _r_factors(red)
    else:
        rf = ctx.r_factor_fn(refl, amap, ctx.anomalous)
        r_merge, r_meas, r_pim = rf["rmerge"], rf["rmeas"], rf["rpim"]

    ios = i_over_sigma(merged, refl, apply_scale=ctx.apply_scale)

    # cctbx enumerates the denominator only as deep as the data in the
    # shell reaches, so a partly filled outermost shell is not charged
    # for the part of the bin that holds no observations at all.
    shell_d_min = max(d_min, float(merged.d.min()) * (1.0 - 1e-6))
    comp = completeness(
        merged,
        ctx.cell,
        ctx.spacegroup,
        shell_d_min,
        d_max,
        anomalous=ctx.anomalous,
        complete=ctx.complete,
    )
    anom_comp = None
    if ctx.anomalous and ctx.complete_anom is not None:
        anom_comp = anomalous_completeness(
            merged,
            ctx.cell,
            ctx.spacegroup,
            shell_d_min,
            d_max,
            complete=ctx.complete_anom,
        )

    cc_r = cc_half_random(
        refl,
        amap,
        anomalous=ctx.anomalous,
        seed=ctx.seed,
        n_repeats=ctx.n_repeats,
        apply_scale=ctx.apply_scale,
    )
    cc_st, cc_st_n = cc_half_sigma_tau(
        sigma_tau_merge(
            refl,
            amap,
            anomalous=ctx.anomalous,
            apply_scale=ctx.apply_scale,
        )
    )
    cc_a = cc_anom(
        refl,
        amap,
        seed=ctx.seed,
        n_repeats=ctx.n_repeats,
        apply_scale=ctx.apply_scale,
    )
    rs = r_split(
        refl,
        amap,
        anomalous=ctx.anomalous,
        seed=ctx.seed,
        n_repeats=ctx.n_repeats,
        apply_scale=ctx.apply_scale,
    )
    wcc, neff = weighted_cc_half(
        refl,
        amap,
        anomalous=ctx.anomalous,
        seed=ctx.seed,
        n_repeats=ctx.n_repeats,
        apply_scale=ctx.apply_scale,
    )

    return ShellStats(
        d_max=d_max,
        d_min=d_min,
        n_obs=n_obs(refl),
        n_uniq=n_unique(merged),
        multiplicity=multiplicity(merged),
        completeness=comp,
        anom_completeness=anom_comp,
        i_mean=ios.i_mean,
        sigi_mean=ios.sigi_mean,
        i_over_sigma_mean=ios.i_over_sigma_mean,
        i_mean_over_sigi_mean=ios.i_mean_over_sigi_mean,
        unmerged_i_over_sigma_mean=ios.unmerged_i_over_sigma_mean,
        r_merge=r_merge,
        r_meas=r_meas,
        r_pim=r_pim,
        cc_half=cc_r.value,
        cc_half_std=cc_r.std,
        cc_half_n_refl=cc_r.n_refl,
        cc_half_sigma_tau=cc_st,
        cc_half_sigma_tau_n_refl=cc_st_n,
        cc_star=cc_star(cc_st),
        cc_anom=cc_a.value,
        cc_anom_std=cc_a.std,
        cc_anom_n_pairs=cc_a.n_refl,
        r_split=rs.value,
        weighted_cc_half=wcc.value,
        weighted_cc_half_neff=neff,
    )


def merging_statistics(
    refl: Reflections,
    amap: AsuMap,
    expt: CrystalMetadata,
    n_bins: int = 20,
    anomalous: bool = False,
    seed: int = 0,
    n_repeats: int = 1,
    apply_scale: bool = True,
    r_factor_fn: (
        Callable[[Reflections, AsuMap, bool], Mapping[str, float]] | None
    ) = None,
) -> MergingStats:
    """Full merging-statistics table, overall and by resolution shell.

    Reproduces the `dials.scale` "Merging statistics by resolution bin"
    table. DIALS reaches this through `merging_stats_from_scaled_array`,
    which runs `iotbx` with `sigma_filtering=None`,
    `eliminate_sys_absent=False` and `use_internal_variance=False`; the
    same choices are made here. Observations with a non-positive sigma
    are dropped first, also matching cctbx.

    Resolution comes from the unit cell rather than the `d` column, so
    the shells follow the cell the statistics are reported against.

    Args:
        refl: unmerged observations.
        amap: symmetry reduction of `refl.hkl`.
        expt: carrier of `cell` and `space_group`, i.e. an
            `io.Experiment`.
        n_bins: number of resolution shells.
        anomalous: keep Friedel mates in separate groups when merging.
            CC(anom) is computed either way.
        seed: RNG seed for the random-split statistics.
        n_repeats: number of random splits to average; raise this to
            suppress the Monte-Carlo noise on CC1/2, CC(anom) and
            R_split.
        apply_scale: divide intensities by `refl.scale`, as DIALS does
            when it forms the array it reports statistics on.
        r_factor_fn: optional replacement for the internal R-factor
            reduction, called as `fn(refl, amap, anomalous)` and
            expected to return a mapping with `rmerge`, `rmeas` and
            `rpim`. Pass `merge.r_factors` to keep a single
            implementation; the two are verified to agree.

    Returns:
        Overall and per-shell statistics.

    Raises:
        ValueError: if no observation survives the sigma filter.
    """
    cell = expt.cell
    spacegroup = expt.space_group

    keep = refl.sigma > 0
    if not bool(keep.any()):
        raise ValueError("no reflections with a positive sigma")
    if not bool(keep.all()):
        refl = refl.select(keep)
        amap = _select_asu_map(amap, keep)

    d = 1.0 / torch.sqrt(d_star_sq(refl.hkl, cell))
    bins = resolution_bins(d, n_bins=n_bins)
    d_max_all = float(d.max())
    d_min_all = float(d.min())

    ctx = _Context(
        cell=cell,
        spacegroup=spacegroup,
        anomalous=anomalous,
        complete=build_complete_set(
            cell, spacegroup, d_min_all, None, anomalous=anomalous
        ),
        complete_anom=(
            build_complete_set(cell, spacegroup, d_min_all, None)
            if anomalous
            else None
        ),
        seed=seed,
        n_repeats=n_repeats,
        apply_scale=apply_scale,
        r_factor_fn=r_factor_fn,
    )

    def stats_for(sel: torch.Tensor, d_max: float, d_min: float) -> ShellStats:
        return _shell_stats(
            refl.select(sel),
            _select_asu_map(amap, sel),
            d[sel],
            d_max,
            d_min,
            ctx,
        )

    all_rows = torch.ones(len(refl), dtype=torch.bool)
    overall = stats_for(all_rows, d_max_all, d_min_all)

    shells: list[ShellStats] = []
    for i in range(bins.n_bins):
        sel = bins.bin_id == i
        if not bool(sel.any()):
            continue
        shell_d_max, shell_d_min = bins.d_range(i)
        shells.append(stats_for(sel, shell_d_max, shell_d_min))

    return MergingStats(
        overall=overall,
        shells=shells,
        bins=bins,
        anomalous=anomalous,
        n_repeats=n_repeats,
    )
