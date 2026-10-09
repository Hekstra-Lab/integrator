"""Unsupervised scale + merge via robust bilinear ALS with centric anchoring.

Extension module (does NOT modify the integrator, loss, factory, or registry).
It takes a trained neural integrator's per-observation intensities and recovers
merged structure-factor intensities -- including the anomalous (Bijvoet)
differences -- WITHOUT a hand-coded physical scale model and WITHOUT any DIALS
result, so the same code transfers across experiment types (rotation, Laue,
stills, ED): only the choice of `scale_group`/covariates changes.

Why this exists
---------------
Every learned-scale attempt so far fits the scale and the signal from one
objective, so a flexible scale absorbs the tiny anomalous signal. The cross-
domain fix (genomics batch-effect removal; Kepler half-sibling regression) is to
*estimate the systematic where the signal provably isn't*. Crystallography gives
a free, exact instance of that: CENTRIC reflections have I(+) = I(-) by symmetry
(no anomalous signal, ever). So:

    model:  I_i  ~  s[scale_group_i] * mu[merge_group_i]

    - acentric reflection: its Friedel mates are SEPARATE merge groups, so their
      recovered difference is the anomalous signal.
    - centric reflection: its Friedel mates are POOLED into ONE merge group
      (mu(+) == mu(-) enforced). The centrics thus pin the scale's Friedel-
      differentiating modes using observations that carry no anomalous signal,
      so the recovered acentric DANO cannot be eaten by the scale.

Estimation is closed-form alternating reweighted least squares -- a *solve*, not
SGD -- with Tukey-biweight outlier rejection and a chi^2 error-model inflation
(the two ingredients DIALS uses that the learned merges lacked). The scale model
is whatever grouping/covariate you pass as `scale_group`; nothing physical is
assumed.

This is a research prototype: numpy, CPU, dependency-free (no torch, no rs).
"""

from __future__ import annotations

import numpy as np


def tukey_biweight(z: np.ndarray, c: float = 6.0) -> np.ndarray:
    """Tukey biweight robustness weights for standardized residuals `z`.

    Returns `(1 - (z/c)^2)^2` inside `|z| < c` and 0 outside, so gross outliers
    are fully rejected while typical points are ~unweighted. `c=6` is lenient
    (rejects only ~6-sigma); lower it to reject harder.
    """
    a = np.abs(z) / c
    w = (1.0 - np.minimum(a, 1.0) ** 2) ** 2
    w[a >= 1.0] = 0.0
    return w


def _contiguous(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary integer ids to contiguous 0..K-1; return (inverse, unique)."""
    uniq, inv = np.unique(np.asarray(ids), return_inverse=True)
    return inv.astype(np.int64), uniq


def _group_sum(group: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(group, weights=weights, minlength=n)[:n]


def build_merge_groups(
    anom_id: np.ndarray, centric: np.ndarray, nonanom_id: np.ndarray
) -> np.ndarray:
    """Centric-anchored merge grouping.

    Acentric observations are grouped by `anom_id` (Friedel mates kept SEPARATE
    so their difference is the anomalous signal); centric observations are
    grouped by `nonanom_id` (Friedel mates POOLED, since symmetry forces them
    equal). Returns a contiguous merge-group index per observation.

    Args:
        anom_id: (N,) anomalous ASU id (F(+) and F(-) have distinct ids).
        centric: (N,) bool, True for centric reflections.
        nonanom_id: (N,) non-anomalous ASU id (F(+) and F(-) share an id).
    """
    anom_id = np.asarray(anom_id, dtype=np.int64)
    nonanom_id = np.asarray(nonanom_id, dtype=np.int64)
    centric = np.asarray(centric, dtype=bool)
    # Disjoint namespaces for the two id systems via a flag * offset + base.
    base = np.where(centric, nonanom_id, anom_id)
    flag = (~centric).astype(np.int64)
    key = flag * (int(base.max()) + 1) + base
    merge_group, _ = _contiguous(key)
    return merge_group


def scale_and_merge(
    intensity: np.ndarray,
    sigma: np.ndarray,
    scale_group: np.ndarray,
    merge_group: np.ndarray,
    *,
    n_iter: int = 100,
    tol: float = 1e-5,
    biweight_c: float = 6.0,
    robust_start: int = 3,
    calibrate_error: bool = True,
) -> dict:
    """Robust bilinear ALS:  intensity_i ~ scale[scale_group_i] * mu[merge_group_i].

    Closed-form alternating updates (a solve, not SGD):
        mu[g] = sum_i w_i s_i I_i / sum_i w_i s_i^2      (per merge group)
        s[k]  = sum_i w_i mu_i I_i / sum_i w_i mu_i^2     (per scale group)
        w_i   = (1/sigma_i^2) * biweight((I_i - s_i mu_i)/sigma_i)
    Gauge fixed each iter by geometric-mean(scale) = 1. Robustness is held off
    for `robust_start` iters so it does not reject everything before the scale
    has converged.

    Args:
        intensity, sigma: (N,) per-observation intensity and its 1-sigma.
        scale_group: (N,) int index into the scale parameter -- the generic,
            no-physics scale model (e.g. image id, or image x detector-bin id).
        merge_group: (N,) int index into the merged intensity (use
            `build_merge_groups` for the centric-anchored choice).

    Returns:
        dict with `mu` (per merge group), `mu_var`/`mu_sigma`, `scale` (per scale
        group), per-observation `weights`, `n_iter`, and the reduced `chi2`.
    """
    I = np.asarray(intensity, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sg, _ = _contiguous(scale_group)
    mg, _ = _contiguous(merge_group)
    n_scale = int(sg.max()) + 1
    n_merge = int(mg.max()) + 1
    inv_var = 1.0 / np.maximum(sigma, 1e-8) ** 2

    scale = np.ones(n_scale)
    w_rob = np.ones_like(I)
    mu = np.zeros(n_merge)
    prev_mu = None

    n_used = 0
    for it in range(n_iter):
        n_used = it + 1
        w = inv_var * w_rob

        s_i = scale[sg]
        num = _group_sum(mg, w * s_i * I, n_merge)
        den = _group_sum(mg, w * s_i * s_i, n_merge)
        mu = num / np.maximum(den, 1e-12)

        mu_i = mu[mg]
        nums = _group_sum(sg, w * mu_i * I, n_scale)
        dens = _group_sum(sg, w * mu_i * mu_i, n_scale)
        scale = nums / np.maximum(dens, 1e-12)

        # Gauge: geometric mean of populated scales = 1 (s*mu kept invariant).
        pop = dens > 0
        if pop.any():
            g = np.exp(np.mean(np.log(np.maximum(scale[pop], 1e-12))))
            scale = scale / g
            mu = mu * g

        # Robust reweighting (after a short warmup).
        if it + 1 >= robust_start:
            resid = I - scale[sg] * mu[mg]
            z = resid * np.sqrt(inv_var)  # resid / sigma
            w_rob = tukey_biweight(z, biweight_c)

        if prev_mu is not None:
            denom = np.abs(prev_mu) + 1e-8
            if np.max(np.abs(mu - prev_mu) / denom) < tol:
                break
        prev_mu = mu.copy()

    # Merged variance from the inverse curvature, then a global chi^2 inflation.
    w = inv_var * w_rob
    den = _group_sum(mg, w * scale[sg] * scale[sg], n_merge)
    mu_var = 1.0 / np.maximum(den, 1e-12)
    chi2 = None
    if calibrate_error:
        resid = I - scale[sg] * mu[mg]
        z2 = (resid**2) * inv_var * w_rob
        dof = max(float(w_rob.sum()) - n_scale - n_merge, 1.0)
        chi2 = float(z2.sum() / dof)
        mu_var = mu_var * max(chi2, 1.0)

    return {
        "mu": mu,
        "mu_var": mu_var,
        "mu_sigma": np.sqrt(mu_var),
        "scale": scale,
        "weights": w_rob,
        "n_iter": n_used,
        "chi2": chi2,
    }


def merge_anomalous(
    intensity: np.ndarray,
    sigma: np.ndarray,
    scale_group: np.ndarray,
    anom_id: np.ndarray,
    nonanom_id: np.ndarray,
    centric: np.ndarray,
    plus: np.ndarray,
    **kwargs,
) -> dict:
    """Centric-anchored scale + merge, read out as a per-reflection anomalous table.

    Builds the centric-anchored merge groups, runs `scale_and_merge`, then maps
    the merged intensities back to one row per unique reflection (`nonanom_id`)
    with F(+)/F(-) intensities and DANO. Centric rows have DANO == 0 by
    construction (their mates were pooled).

    Args:
        plus: (N,) bool, True for the F(+) member of a Friedel pair.
    Returns:
        dict with per-reflection arrays: `nonanom_id`, `I_plus`, `I_minus`,
        `dano`, `sig_plus`, `sig_minus`, `is_centric`, plus the raw solver
        `result` and per-observation `merge_group`.
    """
    plus = np.asarray(plus, dtype=bool)
    centric = np.asarray(centric, dtype=bool)
    merge_group = build_merge_groups(anom_id, centric, nonanom_id)
    res = scale_and_merge(intensity, sigma, scale_group, merge_group, **kwargs)
    mu, mu_sig = res["mu"], res["mu_sigma"]

    na_c, na_uniq = _contiguous(nonanom_id)
    n_u = len(na_uniq)
    plus_mg = np.full(n_u, -1, dtype=np.int64)
    minus_mg = np.full(n_u, -1, dtype=np.int64)
    # All observations of a given (reflection, sign) share one merge group, so a
    # scatter-assign (last wins) recovers it.
    plus_mg[na_c[plus]] = merge_group[plus]
    minus_mg[na_c[~plus]] = merge_group[~plus]
    is_centric = np.zeros(n_u, dtype=bool)
    is_centric[na_c[centric]] = True

    def gather(idx, arr):
        out = np.full(n_u, np.nan)
        ok = idx >= 0
        out[ok] = arr[idx[ok]]
        return out

    I_plus = gather(plus_mg, mu)
    I_minus = gather(minus_mg, mu)
    return {
        "nonanom_id": na_uniq,
        "I_plus": I_plus,
        "I_minus": I_minus,
        "dano": I_plus - I_minus,
        "sig_plus": gather(plus_mg, mu_sig),
        "sig_minus": gather(minus_mg, mu_sig),
        "is_centric": is_centric,
        "merge_group": merge_group,
        "result": res,
    }
