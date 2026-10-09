"""Intensity outlier rejection on symmetry-equivalent groups.

Ports `NormDevOutlierRejection`, the `method="standard"` default
(`outlier_rejection.py:342-430`). Each observation is compared against
the weighted mean of *the other* members of its symmetry group,

    z = (I - g * sum_{j != l} w g I / sum_{j != l} w g^2)
        / sqrt(1/w + g^2 / sum_{j != l} w g^2)

and rejected when `|z| > zmax`. Leaving the test reflection out of its
own mean is what makes a single bright outlier detectable at all -- with
it included, a large enough outlier drags the mean far enough to hide
itself.

Two details of the iteration are easy to miss and are reproduced here:

* Only the single largest `z` in each group is flagged per round
  (`scaling_helper.h:317-341`). Groups are then re-examined with that
  observation removed, repeatedly, until a round flags nothing. One
  outlier can inflate the group mean enough to make its neighbours look
  like outliers too, so flagging them all at once over-rejects.
* After a round, the working set shrinks to the *non-flagged members of
  groups that produced a flag*. Groups that survived a round clean are
  dropped from consideration entirely and are never revisited.

Groups with `n <= 2` are never tested: with two observations the
leave-one-out mean is a single number and the deviation is symmetric by
construction.

Symmetry of the test
--------------------
`abs()` makes this a two-sided test with a single threshold. On
strictly-positive, right-skewed intensities the lower tail is bounded --
`I >= 0` caps how negative `z` can get, while the upper tail is
unbounded -- so for realistic `zmax` essentially every flagged outlier
is a *bright* reflection. The test is not symmetric in effect even
though it is written symmetrically. `zmax_lower` makes the two tails
independent: leave it `None` for DIALS' behaviour, set it to `inf` for a
purely one-sided upper test, or to a different finite value for an
asymmetric one.
"""

from __future__ import annotations

import torch

from .target import resolve_groups
from .types import DTYPE, AsuMap, Reflections

_EPS = 1e-10


def group_median(
    values: torch.Tensor, group_id: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Median of `values` within each group.

    Even-sized groups take the mean of the two central values, matching
    `scitbx::math::median_functor` and `numpy.median`.

    Args:
        values: `(N,)` values to reduce.
        group_id: `(N,)` group index in `[0, n_groups)`.
        n_groups: number of groups.

    Returns:
        `(n_groups,)` medians; empty groups are zero.
    """
    by_value = torch.argsort(values, stable=True)
    order = by_value[torch.argsort(group_id[by_value], stable=True)]
    ordered = values[order]

    counts = torch.zeros(n_groups, dtype=torch.int64).index_add(
        0, group_id, torch.ones_like(group_id)
    )
    starts = torch.zeros(n_groups, dtype=torch.int64)
    starts[1:] = torch.cumsum(counts, 0)[:-1]

    nonempty = counts > 0
    safe = counts.clamp_min(1)
    lo = starts + (safe - 1) // 2
    hi = starts + safe // 2
    medians = 0.5 * (ordered[lo] + ordered[hi])
    return torch.where(nonempty, medians, torch.zeros_like(medians))


def limit_outlier_weights(
    weights: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
    factor: float = 10.0,
) -> torch.Tensor:
    """Cap each weight at `factor` times its group median.

    Ports `limit_outlier_weights` (`scaling_helper.h:126-148`). Without
    it a single unusually precise observation dominates its group's
    weighted mean, and every other member of the group is then measured
    against essentially that one value.

    Args:
        weights: `(N,)` observation weights.
        group_id: `(N,)` group index.
        n_groups: number of groups.
        factor: multiple of the group median at which to cap.

    Returns:
        `(N,)` capped weights.
    """
    medians = group_median(weights, group_id, n_groups)
    return torch.minimum(weights, factor * medians[group_id])


def leave_one_out_z(
    intensity: torch.Tensor,
    weights: torch.Tensor,
    scale: torch.Tensor,
    group_id: torch.Tensor,
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised deviation from the leave-one-out weighted mean.

    Args:
        intensity: `(N,)` observed intensity.
        weights: `(N,)` weights, already capped.
        scale: `(N,)` inverse scale factors.
        group_id: `(N,)` group index.
        n_groups: number of groups.

    Returns:
        `(norm_dev, n_h)`. `norm_dev` is set to `1000` where the
        leave-one-out denominator vanishes, matching DIALS' trigger
        value; `n_h` is the group multiplicity per reflection.
    """
    wg = weights * scale
    wgi = torch.zeros(n_groups, dtype=DTYPE).index_add(
        0, group_id, wg * intensity
    )
    wg2 = torch.zeros(n_groups, dtype=DTYPE).index_add(0, group_id, wg * scale)
    counts = torch.zeros(n_groups, dtype=DTYPE).index_add(
        0, group_id, torch.ones_like(intensity)
    )

    wgi_others = wgi[group_id] - wg * intensity
    wg2_others = wg2[group_id] - wg * scale
    degenerate = wg2_others == 0.0
    safe = torch.where(degenerate, torch.ones_like(wg2_others), wg2_others)

    norm_dev = (intensity - scale * wgi_others / safe) / torch.sqrt(
        (1.0 / weights.clamp_min(_EPS)) + scale**2 / safe
    )
    norm_dev = torch.where(
        degenerate, torch.full_like(norm_dev, 1000.0), norm_dev
    )
    return norm_dev, counts[group_id]


def _reject_score(
    norm_dev: torch.Tensor, zmax: float, zmax_lower: float | None
) -> torch.Tensor:
    """Rescale so that a score above 1 means reject, in either tail."""
    if zmax_lower is None:
        return norm_dev.abs() / zmax
    upper = norm_dev.clamp_min(0.0) / zmax
    if zmax_lower == float("inf"):
        return upper
    return torch.maximum(upper, (-norm_dev).clamp_min(0.0) / zmax_lower)


def _compact(group_id: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Renumber group ids to `[0, n)` after a selection."""
    uniq, inverse = torch.unique(group_id, return_inverse=True)
    return inverse, int(uniq.numel())


def reject_outliers(
    refl: Reflections,
    amap: AsuMap,
    scale: torch.Tensor | None = None,
    zmax: float = 6.0,
    anomalous: bool = False,
    zmax_lower: float | None = None,
    weights: torch.Tensor | None = None,
    weight_cap: float = 10.0,
    pool_centrics: bool = True,
    max_rounds: int = 100,
) -> torch.Tensor:
    """Flag intensity outliers, one per group per round until stable.

    Args:
        refl: unmerged observations.
        amap: symmetry map for the same rows.
        scale: `(N,)` inverse scale factors; defaults to `refl.scale`.
        zmax: rejection threshold on the normalised deviation.
        anomalous: keep Friedel mates in separate groups.
        zmax_lower: threshold for the lower tail. `None` reproduces
            DIALS' symmetric `|z| > zmax`; `float("inf")` gives a
            one-sided upper test; any other value gives an asymmetric
            one.
        weights: `(N,)` observation weights; defaults to
            `1/variance`. DIALS passes error-model-corrected weights
            here once an error model exists.
        weight_cap: multiple of the group median at which weights are
            capped, or `0` to disable.
        pool_centrics: see `target.resolve_groups`.
        max_rounds: safety cap on the number of rounds.

    Returns:
        `(N,)` boolean mask, true for rejected reflections.
    """
    g = refl.scale if scale is None else scale
    group_id, n_groups = resolve_groups(amap, anomalous, pool_centrics)
    n = len(refl)

    w = 1.0 / refl.variance.clamp_min(_EPS) if weights is None else weights
    if weight_cap > 0:
        w = limit_outlier_weights(w, group_id, n_groups, weight_cap)

    outliers = torch.zeros(n, dtype=torch.bool)
    active = torch.arange(n, dtype=torch.int64)

    for _ in range(max_rounds):
        if active.numel() == 0:
            break
        gid, n_active = _compact(group_id[active])
        norm_dev, n_h = leave_one_out_z(
            refl.intensity[active],
            w[active],
            g[active],
            gid,
            n_active,
        )
        score = _reject_score(norm_dev, zmax, zmax_lower)
        score = torch.where(n_h > 2, score, torch.zeros_like(score))

        best = torch.full((n_active,), -1.0, dtype=DTYPE).scatter_reduce_(
            0, gid, score, reduce="amax", include_self=True
        )
        is_max = (score == best[gid]) & (score > 1.0)
        if not bool(is_max.any()):
            break

        # A tie inside a group would flag two rows in one round; DIALS
        # keeps whichever it met first, so keep the lowest row index.
        rows = torch.arange(gid.numel(), dtype=torch.int64)
        first = torch.full((n_active,), gid.numel(), dtype=torch.int64)
        first.scatter_reduce_(
            0, gid[is_max], rows[is_max], reduce="amin", include_self=True
        )
        flagged = torch.zeros(gid.numel(), dtype=torch.bool)
        chosen = first[first < gid.numel()]
        flagged[chosen] = True

        outliers[active[flagged]] = True
        has_outlier = torch.zeros(n_active, dtype=torch.bool)
        has_outlier[gid[flagged]] = True
        active = active[has_outlier[gid] & ~flagged]

    return outliers
