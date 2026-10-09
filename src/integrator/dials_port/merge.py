"""Weighted merging of symmetry equivalents, ported from cctbx.

`dials.merge` does not merge anything itself: it builds a
`cctbx.miller.array` from the scaled reflection table and delegates to
`miller.array.merge_equivalents(algorithm="gaussian")`, which is the C++
`merge_equivalents_obs` at `cctbx/miller/merge_equivalents.h:266-420`.
This module reproduces that class exactly, vectorised over groups with
`index_add_` instead of the original loop over contiguous runs of equal
Miller index.

Two details of the original are easy to miss and are reproduced here:

- The `sigma_dynamic_range` prefilter (default `1e-6`). Observations
  whose sigma is not strictly greater than `1e-6 * max_sigma` within
  their own group are dropped from the weighted mean, yet the reported
  multiplicity still counts them, and the R-factor sums still run over
  them.
- Singleton groups keep their own sigma verbatim rather than
  `sqrt(1 / sum_w)`. The two agree analytically but not bitwise.

The anomalous flag is not an algorithm switch. DIALS runs the identical
code over a different grouping key; `grouping` builds it, and does not
simply forward `AsuMap.groups`, for the reason documented there.
"""

from __future__ import annotations

import torch

from .types import DTYPE, IDTYPE, AsuMap, MergedData, Reflections

# `merge_equivalents_obs` default, merge_equivalents.h:277.
SIGMA_DYNAMIC_RANGE = 1e-6


def _group_sum(
    values: torch.Tensor, group_id: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Sum `values` into `n_groups` bins indexed by `group_id`.

    Args:
        values: `(N,)` values to accumulate.
        group_id: `(N,)` destination bin of each value.
        n_groups: number of bins.

    Returns:
        `(n_groups,)` sums, zero for empty bins.
    """
    out = torch.zeros(n_groups, dtype=DTYPE)
    out.index_add_(0, group_id, values)
    return out


def _representatives(group_id: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Pick one source row per group, for carrying per-group metadata."""
    rep = torch.zeros(n_groups, dtype=IDTYPE)
    rep[group_id] = torch.arange(group_id.numel(), dtype=IDTYPE)
    return rep


def grouping(amap: AsuMap, anomalous: bool) -> tuple[torch.Tensor, int]:
    """Grouping key matching `cctbx.miller.set.map_to_asu`.

    A centric reflection satisfies `h ~ -h` under the rotation group
    alone, so `map_to_asu` sends both Friedel mates to the same
    anomalous ASU index and `merge_equivalents` pools them. Splitting
    centrics by `isym` parity would over-split them and inflate the
    merged reflection count -- on the HEWL test set, 25099 groups
    instead of cctbx's 23096.

    `AsuMap` now applies this fold itself, so this function agrees with
    `amap.groups(anomalous)` on the partition. It is retained because it
    fixes the group *ordering* the merged rows come back in, which
    `AsuMap` does not promise: centrics are emitted first here. Callers
    that join merged output by Miller index are unaffected either way.

    Args:
        amap: symmetry reduction.
        anomalous: keep Friedel mates in separate groups.

    Returns:
        `(group_id, n_groups)` with ids contiguous in `[0, n_groups)`.
    """
    if not anomalous:
        return amap.groups(False)

    # Negating the Laue id keeps the centric keys disjoint from the
    # anomalous ones without needing to know either id range.
    key = torch.where(amap.centric, -(amap.group_id_laue + 1), amap.group_id)
    uniq, inverse = torch.unique(key, return_inverse=True)
    return inverse.to(IDTYPE), int(uniq.numel())


def scale_observations(
    refl: Reflections,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the inverse scale factor to intensities and variances.

    Mirrors `scaled_data_as_miller_array`
    (`dials/algorithms/scaling/scaling_library.py:339-347`), which
    divides the intensity by `inverse_scale_factor` and the sigma by the
    same quantity.

    Args:
        refl: unmerged observations carrying `scale`.

    Returns:
        `(intensity, variance)`, both `(N,)` and already scaled.

    Note:
        DIALS forms `sqrt(variance) / scale`, which is negative when the
        scale is negative; this returns `variance / scale**2`, whose
        square root is always non-negative. The two differ only for
        non-positive scale factors, which a converged DIALS scaling run
        does not produce.
    """
    return refl.intensity / refl.scale, refl.variance / refl.scale**2


def merge(
    refl: Reflections,
    amap: AsuMap,
    anomalous: bool = False,
    use_internal_variance: bool = False,
) -> MergedData:
    """Merge symmetry equivalents with inverse-variance weights.

    For each group, with `w_i = 1 / sigma_i**2` over the observations
    that survive the `SIGMA_DYNAMIC_RANGE` prefilter:

        I = sum(w * I) / sum(w)

    and, when `use_internal_variance` is false, `sigma = sqrt(1 /
    sum(w))`. When it is true the external estimate is replaced by the
    larger of itself and the GSL weighted variance of the mean,

        V_w = sum(w) / (sum(w)**2 - sum(w**2)) * sum(w * (I - Ibar)**2)

    divided by the number of surviving observations.

    Args:
        refl: unmerged observations. Intensities and variances are
            scaled by `scale_observations` before merging.
        amap: symmetry reduction supplying the grouping key.
        anomalous: keep Friedel mates in separate groups.
        use_internal_variance: use the scatter of the observations when
            it exceeds the propagated error. `False` is the
            `dials.merge` default (`dials/algorithms/merging/merge.py`
            :166), and `True` is the cctbx default.

    Returns:
        One row per group. `multiplicity` counts every observation in
        the group, including any the prefilter dropped. Groups in which
        every observation was dropped report zero intensity and zero
        sigma, as cctbx does.
    """
    group_id, n_groups = grouping(amap, anomalous)
    intensity, variance = scale_observations(refl)
    sigma = torch.sqrt(variance)

    # cctbx seeds the per-group maximum with 0, so a group of entirely
    # non-positive sigmas gets a threshold of 0 and is emptied.
    max_sigma = torch.zeros(n_groups, dtype=DTYPE).scatter_reduce_(
        0, group_id, sigma, reduce="amax", include_self=True
    )
    keep = sigma > max_sigma[group_id] * SIGMA_DYNAMIC_RANGE

    kept = keep.nonzero(as_tuple=True)[0]
    gid = group_id[kept]
    i_k = intensity[kept]
    s_k = sigma[kept]
    w = 1.0 / (s_k * s_k)

    n_kept = torch.zeros(n_groups, dtype=IDTYPE)
    n_kept.index_add_(0, gid, torch.ones_like(gid))

    sum_w = _group_sum(w, gid, n_groups)
    sum_w_sq = _group_sum(w * w, gid, n_groups)
    sum_wi = _group_sum(w * i_k, gid, n_groups)

    # Exact for singletons: a one-term `index_add_` reproduces its input.
    sum_i = _group_sum(i_k, gid, n_groups)
    sum_s = _group_sum(s_k, gid, n_groups)

    ones = torch.ones(n_groups, dtype=DTYPE)
    mean = sum_wi / torch.where(sum_w > 0, sum_w, ones)

    delta = i_k - mean[gid]
    sum_w_delta_sq = _group_sum(w * delta * delta, gid, n_groups)

    single = n_kept == 1
    multi = n_kept >= 2

    merged_i = torch.zeros(n_groups, dtype=DTYPE)
    merged_i[single] = sum_i[single]
    merged_i[multi] = mean[multi]

    merged_s = torch.zeros(n_groups, dtype=DTYPE)
    merged_s[single] = sum_s[single]

    external = 1.0 / sum_w[multi]
    if use_internal_variance:
        sw = sum_w[multi]
        internal = (
            sw
            / (sw * sw - sum_w_sq[multi])
            * sum_w_delta_sq[multi]
            / n_kept[multi].to(DTYPE)
        )
        merged_s[multi] = torch.sqrt(torch.maximum(internal, external))
    else:
        merged_s[multi] = torch.sqrt(external)

    rep = _representatives(group_id, n_groups)
    return MergedData(
        hkl=amap.asu_hkl[rep],
        intensity=merged_i,
        sigma=merged_s,
        multiplicity=torch.bincount(group_id, minlength=n_groups).to(IDTYPE),
        d=refl.d[rep],
        centric=amap.centric[rep],
        plus=amap.plus[rep],
    )


def r_factors(
    refl: Reflections,
    amap: AsuMap,
    anomalous: bool = False,
) -> dict[str, float]:
    """Merging residual statistics, as accumulated by cctbx.

    Reproduces `merge_equivalents::compute_r_factors`
    (`merge_equivalents.h:65-97`). With `d_ij` the `j`-th observation of
    group `i`, `n_i` its full multiplicity and `<I_i>` the merged --
    that is, inverse-variance weighted -- intensity, the sums run over
    groups with `n_i > 1` only:

        num_i  = sum_j |d_ij - <I_i>|
        rmerge = sum_i num_i / sum_i sum_j d_ij
        rmeas  = sum_i sqrt(n_i / (n_i - 1)) * num_i / sum_i sum_j d_ij
        rpim   = sum_i sqrt(1 / (n_i - 1)) * num_i / sum_i sum_j d_ij
        rint   = sum_i num_i / sum_i sum_j |d_ij|

    Only `rint` takes the absolute value in the denominator; the other
    three divide by the signed sum. Using an unweighted group mean for
    `<I_i>` is the usual way to get these numbers slightly wrong.

    Args:
        refl: unmerged observations.
        amap: symmetry reduction supplying the grouping key.
        anomalous: keep Friedel mates in separate groups.

    Returns:
        `rmerge`, `rmeas`, `rpim` and `rint`. A statistic whose
        denominator vanishes is reported as `0.0`, matching cctbx.
    """
    merged = merge(refl, amap, anomalous=anomalous)
    group_id, n_groups = grouping(amap, anomalous)
    intensity, _ = scale_observations(refl)

    residual = intensity - merged.intensity[group_id]
    sum_num = _group_sum(residual.abs(), group_id, n_groups)
    sum_abs_den = _group_sum(intensity.abs(), group_id, n_groups)
    sum_signed_den = _group_sum(intensity, group_id, n_groups)

    multi = merged.multiplicity > 1
    n = merged.multiplicity[multi].to(DTYPE)
    num = sum_num[multi]

    r_int_num = float(num.sum())
    r_int_den = float(sum_abs_den[multi].sum())
    r_merge_den = float(sum_signed_den[multi].sum())
    r_meas_num = float((torch.sqrt(n / (n - 1.0)) * num).sum())
    r_pim_num = float((torch.sqrt(1.0 / (n - 1.0)) * num).sum())

    return {
        "rmerge": r_int_num / r_merge_den if r_merge_den else 0.0,
        "rmeas": r_meas_num / r_merge_den if r_merge_den else 0.0,
        "rpim": r_pim_num / r_merge_den if r_merge_den else 0.0,
        "rint": r_int_num / r_int_den if r_int_den else 0.0,
    }


def separate_bijvoet(merged: MergedData) -> dict[str, torch.Tensor]:
    """Split an anomalous merge into Friedel mates and their difference.

    Rows of `merged` are regrouped by ASU Miller index, so that the
    I(+) and I(-) members of a Bijvoet pair, which merged into separate
    groups, end up in the same output row.

    Args:
        merged: result of `merge(..., anomalous=True)`. Passing a
            non-anomalous merge is accepted but leaves one hemisphere
            entirely absent.

    Returns:
        Keyed by `hkl`, `centric`, `I_plus`, `sig_plus`, `I_minus`,
        `sig_minus`, `has_plus`, `has_minus`, `dano` and `sigdano`,
        where `dano = I(+) - I(-)` and `sigdano = sqrt(sig_plus**2 +
        sig_minus**2)`. A hemisphere that was not measured is `nan`,
        which propagates into `dano` and `sigdano`.

        A centric reflection is its own Friedel mate and merges into a
        single group, so both hemispheres are filled with that one
        value, as an MTZ would carry it. Its Bijvoet difference is zero
        by symmetry and carries no information, so `dano` and `sigdano`
        are `nan` for centrics -- the same reflections `cctbx`'s
        `anomalous_differences` drops.
    """
    uniq, inverse = torch.unique(merged.hkl, dim=0, return_inverse=True)
    n = int(uniq.shape[0])

    out: dict[str, torch.Tensor] = {"hkl": uniq}
    centric = torch.zeros(n, dtype=torch.bool)
    centric[inverse] = merged.centric
    out["centric"] = centric

    present: dict[str, torch.Tensor] = {}
    for side, hemisphere in (("plus", merged.plus), ("minus", ~merged.plus)):
        mask = hemisphere | merged.centric
        rows = inverse[mask]
        i_side = torch.full((n,), torch.nan, dtype=DTYPE)
        s_side = torch.full((n,), torch.nan, dtype=DTYPE)
        i_side[rows] = merged.intensity[mask]
        s_side[rows] = merged.sigma[mask]
        out[f"I_{side}"] = i_side
        out[f"sig_{side}"] = s_side
        flag = torch.zeros(n, dtype=torch.bool)
        flag[rows] = True
        present[f"has_{side}"] = flag

    acentric = ~centric
    out["dano"] = torch.where(
        acentric,
        out["I_plus"] - out["I_minus"],
        torch.nan,
    )
    out["sigdano"] = torch.where(
        acentric,
        torch.sqrt(out["sig_plus"] ** 2 + out["sig_minus"] ** 2),
        torch.nan,
    )
    out.update(present)
    return out
