"""French-Wilson posterior intensities and amplitudes.

Port of `dials.algorithms.merging.french_wilson`, which estimates
`E[J | I, sigma]` and `E[F | I, sigma]` under a Wilson prior whose scale
`Sigma` is estimated in resolution shells. The posterior moments
themselves are not computed: they are read off the tables in
`fw_tables` by linear interpolation in the reduced variable `h`, with
closed-form branches taking over past the end of each table.

The pipeline, in the order DIALS runs it:

1. Bin the merged reflections into equal-count shells in `d*^3`.
2. Estimate `Sigma` per shell as `median(I) / log(2)`, the consistent
   estimator of the mean of an exponential from its median.
3. Interpolate `Sigma` linearly on shell centres, clamping rather than
   extrapolating outside the first and last centre.
4. Form `h`, look up the tables, and reject weak or badly-conditioned
   observations.

Two latent DIALS bugs are reproduced by default; `fix_bugs=True` turns
both off. See `french_wilson` for the details.
"""

from __future__ import annotations

import math

import torch

from . import fw_tables
from .types import DTYPE, MergedData

H_MIN = -4.0
H_STEP = 0.1
MIN_BIN_SIZE = 20


def _as_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=DTYPE)


def _table_grid(n: int) -> torch.Tensor:
    """Reduced-variable grid of a table with `n` entries."""
    return torch.arange(n, dtype=DTYPE) * H_STEP + H_MIN


def interp(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
    """Piecewise-linear interpolation, clamped outside `xp`.

    Matches `numpy.interp` and the `scipy.interpolate.interp1d` call
    DIALS makes with `bounds_error=False` and constant end fills, down
    to the evaluation order `slope * (x - x0) + y0`.

    Args:
        x: query points, any shape.
        xp: `(K,)` sample abscissae, strictly increasing.
        fp: `(K,)` sample values.

    Returns:
        Interpolated values with the shape of `x`.
    """
    idx = torch.searchsorted(xp, x, right=True) - 1
    idx = idx.clamp(0, xp.numel() - 2)
    x0 = xp[idx]
    dx = xp[idx + 1] - x0
    slope = (fp[idx + 1] - fp[idx]) / torch.where(
        dx == 0, torch.ones_like(dx), dx
    )
    y = slope * (x - x0) + fp[idx]
    y = torch.where(x <= xp[0], fp[0], y)
    return torch.where(x >= xp[-1], fp[-1], y)


def standardized_median(a: torch.Tensor) -> torch.Tensor:
    """Consistent estimator of an exponential mean from its median.

    For `I ~ Exp(Sigma)` the median is `Sigma * log(2)`, so dividing the
    sample median by `log(2)` estimates `Sigma` while staying robust to
    the bright tail that would drag a plain mean.

    Args:
        a: sample values.

    Returns:
        Scalar estimate, `nan` for an empty sample.

    Note:
        `torch.median` returns the lower of the two central order
        statistics for an even-sized sample, where `numpy.median` --
        which is what DIALS calls -- averages them. The average is
        taken here explicitly.
    """
    n = a.numel()
    if n == 0:
        return torch.tensor(torch.nan, dtype=DTYPE)
    ordered = torch.sort(a.reshape(-1)).values
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2.0
    return median / math.log(2.0)


def determine_binning_counting_sorted(
    x: torch.Tensor,
    max_bins: int = 60,
    min_bin_size: int = MIN_BIN_SIZE,
) -> torch.Tensor:
    """Equal-count bin edges, taken as quantiles of the sorted values.

    Args:
        x: `(N,)` values to bin.
        max_bins: upper bound on the number of bins.
        min_bin_size: lower bound on the number of points per bin, which
            wins over `max_bins` for small inputs.

    Returns:
        `(n_bins + 1,)` non-decreasing edges. The first and last are the
        minimum and maximum of `x` exactly, so no point falls outside.
    """
    n_points = x.numel()
    points_per_bin = max(n_points // max_bins, min_bin_size)
    n_bins = max(n_points // points_per_bin, 1)
    positions = torch.linspace(0.0, float(n_points), n_bins + 1, dtype=DTYPE)
    ranks = torch.arange(n_points, dtype=DTYPE)
    return interp(positions, ranks, torch.sort(x).values)


def _bin_index(x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Zero-based bin of each point, following `scipy.binned_statistic`.

    scipy digitises with a right-closed search and then shifts points
    sitting on the rightmost edge back into the last bin. Because
    `edges` spans the full range of `x`, that is equivalent to clamping.
    """
    n_bins = edges.numel() - 1
    idx = torch.searchsorted(edges, x, right=True) - 1
    return idx.clamp(0, n_bins - 1)


def _binned_median(
    x: torch.Tensor, values: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """Per-bin `standardized_median`, vectorised.

    Reproduces the median that `scipy.stats.binned_statistic` computes
    for a callable statistic: for an even count, the mean of the two
    central order statistics.

    Args:
        x: `(N,)` binning coordinate.
        values: `(N,)` values to reduce.
        edges: `(n_bins + 1,)` bin edges.

    Returns:
        `(n_bins,)` estimates, `nan` for empty bins.
    """
    n_bins = edges.numel() - 1
    bin_idx = _bin_index(x, edges)

    by_value = torch.argsort(values)
    order = by_value[torch.argsort(bin_idx[by_value], stable=True)]
    sorted_values = values[order]

    counts = torch.bincount(bin_idx, minlength=n_bins)
    starts = torch.cumsum(counts, 0) - counts
    mid = starts.to(DTYPE) + (counts.to(DTYPE) - 1.0) / 2.0

    # Empty bins produce an out-of-range `mid`; clamping keeps the
    # gather in bounds and the result is masked out below.
    last = max(x.numel() - 1, 0)
    lo = torch.floor(mid).clamp(0, last).long()
    hi = torch.ceil(mid).clamp(0, last).long()
    medians = (sorted_values[lo] + sorted_values[hi]) / 2.0

    out = medians / math.log(2.0)
    return torch.where(counts > 0, out, torch.full_like(out, torch.nan))


def compute_expected_intensities(
    x: torch.Tensor,
    intensities: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Wilson scale `Sigma` at each reflection.

    `Sigma` is estimated once per bin and then interpolated linearly on
    the bin centres. Reflections outside the first and last centre take
    the end value rather than an extrapolation, which is what the
    `fill_value` argument in the DIALS source selects.

    Args:
        x: `(N,)` binning coordinate, `d*^3`.
        intensities: `(N,)` merged intensities.
        edges: `(n_bins + 1,)` bin edges.

    Returns:
        `(N,)` expected intensities.
    """
    bin_means = _binned_median(x, intensities, edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return interp(x, centers, bin_means)


def _posterior_moments_acentric(
    intensities: torch.Tensor,
    sigmas: torch.Tensor,
    expected: torch.Tensor,
    h_min: float,
) -> dict[str, torch.Tensor]:
    """Acentric branch: table lookup below `h = 3`, Gaussian above."""
    i_over_sig = intensities / sigmas
    h = i_over_sig - torch.abs(sigmas / expected)
    valid = (i_over_sig >= h_min + 0.3) & (h >= h_min)

    grid = _table_grid(len(fw_tables.ac_zj))
    root_sigma = torch.sqrt(sigmas)

    j = torch.zeros_like(h)
    sig_j = torch.zeros_like(h)
    f = torch.zeros_like(h)
    sig_f = torch.zeros_like(h)

    tabulated = valid & (h < 3.0)
    if bool(tabulated.any()):
        h_t = h[tabulated]
        j[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.ac_zj)) * sigmas[tabulated]
        )
        sig_j[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.ac_zj_sd))
            * sigmas[tabulated]
        )
        f[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.ac_zf))
            * root_sigma[tabulated]
        )
        sig_f[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.ac_zf_sd))
            * root_sigma[tabulated]
        )

    gaussian = valid & (h >= 3.0)
    if bool(gaussian.any()):
        j[gaussian] = h[gaussian] * sigmas[gaussian]
        sig_j[gaussian] = sigmas[gaussian]
        f[gaussian] = torch.sqrt(j[gaussian])
        sig_f[gaussian] = 0.5 * sigmas[gaussian] / f[gaussian]

    return {"J": j, "sigJ": sig_j, "F": f, "sigF": sig_f, "valid": valid}


def _posterior_moments_centric(
    intensities: torch.Tensor,
    sigmas: torch.Tensor,
    expected: torch.Tensor,
    h_min: float,
) -> dict[str, torch.Tensor]:
    """Centric branch: table lookup below `h = 4`, expansion above."""
    i_over_sig = intensities / sigmas
    h = i_over_sig - torch.abs(sigmas / (2.0 * expected))
    valid = (i_over_sig >= h_min + 0.3) & (h >= h_min)

    grid = _table_grid(len(fw_tables.c_zj))
    root_sigma = torch.sqrt(sigmas)

    j = torch.zeros_like(h)
    sig_j = torch.zeros_like(h)
    f = torch.zeros_like(h)
    sig_f = torch.zeros_like(h)

    tabulated = valid & (h < 4.0)
    if bool(tabulated.any()):
        h_t = h[tabulated]
        j[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.c_zj)) * sigmas[tabulated]
        )
        sig_j[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.c_zj_sd))
            * sigmas[tabulated]
        )
        f[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.c_zf))
            * root_sigma[tabulated]
        )
        sig_f[tabulated] = (
            interp(h_t, grid, _as_tensor(fw_tables.c_zf_sd))
            * root_sigma[tabulated]
        )

    # French-Wilson asymptotic expansion with the extra 1/h^6 term that
    # DIALS added (french_wilson.py:231-245).
    asymptotic = valid & (h >= 4.0)
    if bool(asymptotic.any()):
        h_a = h[asymptotic]
        s_a = sigmas[asymptotic]
        h_2 = 1.0 / (h_a * h_a)
        h_4 = h_2 * h_2
        h_6 = h_2 * h_4
        post_f = torch.sqrt(h_a) * (
            1.0
            - (3.0 / 8.0) * h_2
            - (87.0 / 128.0) * h_4
            - (2889.0 / 1024.0) * h_6
        )
        post_sig_f = torch.sqrt(
            h_a
            * ((1.0 / 4.0) * h_2 + (15.0 / 32.0) * h_4 + (273.0 / 128.0) * h_6)
        )
        j[asymptotic] = h_a * s_a * (1.0 - 0.5 * h_2 - 0.75 * h_4 - 3.0 * h_6)
        sig_j[asymptotic] = 2.0 * s_a * post_f * post_sig_f
        f[asymptotic] = post_f * torch.sqrt(s_a)
        sig_f[asymptotic] = post_sig_f * torch.sqrt(s_a)

    return {"J": j, "sigJ": sig_j, "F": f, "sigF": sig_f, "valid": valid}


def french_wilson(
    merged: MergedData,
    max_bins: int = 60,
    sigma_iobs_rejection_criterion: float = H_MIN,
    fix_bugs: bool = False,
    d_star_cubed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Posterior intensities and amplitudes for a merged dataset.

    Args:
        merged: unique merged reflections. `d` supplies the binning
            coordinate through `d*^3 = (1 / d)**3`.
        max_bins: upper bound on the number of resolution shells.
        sigma_iobs_rejection_criterion: lower limit on `h`, and, offset
            by `0.3`, on `I / sigma`. Only honoured when `fix_bugs` is
            true; see below.
        fix_bugs: repair the two defects described below instead of
            reproducing them.
        d_star_cubed: binning coordinate, overriding the one derived
            from `merged.d`. Because the bin edges are quantiles of the
            data itself, a large share of reflections sit exactly on an
            edge, and a one-ulp change in this coordinate moves them
            between shells. That perturbs the per-shell median `Sigma`
            by up to a percent and is the only reason this function
            disagrees with DIALS by more than roundoff. Supply cctbx's
            `d_star_cubed()` here to compare the two bit-for-bit.

    Returns:
        Keyed by `J`, `sigJ`, `F`, `sigF` and `valid`, each `(M,)` and
        aligned with the rows of `merged`. Entries where `valid` is
        false hold zeros and must not be used; DIALS drops them from its
        output array rather than returning a mask.

    Note:
        Two defects in the DIALS source are reproduced by default so
        that this function tracks DIALS numerically.

        1. `sigma_iobs_rejection_criterion` is accepted at the top level
           of `dials/algorithms/merging/french_wilson.py:29` and then
           never passed on; the limit is hard-coded to `-4` in the
           `h_min` defaults at `:157` and `:204`. The parameter is
           therefore inert, whatever the caller sets.
        2. `:40` reads `rejected = (sigmas <= 0) & ((intensities <= 0)
           | (sigmas < 0))`. The outer operator is almost certainly
           meant to be `|`, since the debug message immediately below
           it reports "reflections with non-positive sigmas". As
           written the clause is equivalent to `(sigmas < 0) | ((sigmas
           == 0) & (intensities <= 0))`, so the one case it fails to
           catch is `sigma == 0` with `I > 0`. That reflection reaches
           the `I / sigma` division, yields `h = inf`, passes both
           validity tests, and is emitted with `J` and `F` equal to
           `nan` -- a silent `nan` in the output rather than a rejected
           row.

        With `fix_bugs=True` the rejection becomes `sigmas <= 0`, its
        evident intent, and `sigma_iobs_rejection_criterion` is honoured
        as `h_min`.

        One deliberate divergence, unrelated to those: DIALS computes
        `n_bins = len(x) // points_per_bin` and crashes for fewer than
        `min_bin_size` reflections, because the resulting edge array is
        too short to bin with. `determine_binning_counting_sorted`
        floors the count at one bin instead.
    """
    intensities = merged.intensity.to(DTYPE)
    sigmas = merged.sigma.to(DTYPE)
    is_centric = merged.centric

    if fix_bugs:
        h_min = float(sigma_iobs_rejection_criterion)
        rejected = sigmas <= 0
    else:
        h_min = H_MIN
        rejected = (sigmas <= 0) & ((intensities <= 0) | (sigmas < 0))

    if d_star_cubed is None:
        d_star_cubed = (1.0 / merged.d.to(DTYPE)) ** 3
    edges = determine_binning_counting_sorted(d_star_cubed, max_bins=max_bins)
    expected = compute_expected_intensities(d_star_cubed, intensities, edges)
    rejected = rejected | (expected == 0)
    valid = ~rejected

    j = torch.zeros_like(intensities)
    sig_j = torch.zeros_like(intensities)
    f = torch.zeros_like(intensities)
    sig_f = torch.zeros_like(intensities)

    for mask, moments in (
        (~is_centric, _posterior_moments_acentric),
        (is_centric, _posterior_moments_centric),
    ):
        if not bool(mask.any()):
            continue
        part = moments(intensities[mask], sigmas[mask], expected[mask], h_min)
        j[mask] = part["J"]
        sig_j[mask] = part["sigJ"]
        f[mask] = part["F"]
        sig_f[mask] = part["sigF"]
        valid[mask] &= part["valid"]

    return {
        "J": j,
        "sigJ": sig_j,
        "F": f,
        "sigF": sig_f,
        "valid": valid,
    }
