"""Estimate the Bijvoet ratio by conditioning on wavelength strata.

On a still a Friedel pair is never excited at the same wavelength: $h$ is
excited at $\\lambda = -2\\,\\mathbf{q}\\cdot\\mathbf{s}_0/|\\mathbf{q}|^2$, so
$-h$ needs the opposite sign and therefore a different orientation.  The
monochromatic guardrail -- both mates ride the same scale, so its error cancels
in $I(+) - I(-)$ -- does not exist here, and tying the mates' scales together
would assert something false, since their true scales differ by up to the
$18\\times$ span of the spectrum.

What does survive is conditioning.  Within a stratum of (reflection,
wavelength bin), every factor shared by the observations in it -- the merged
intensity, the spectrum's error, the resolution gauge $c(d)$, the per-image
terms to the extent they are common -- cancels exactly from the conditional
distribution of the signal counts given their total:

    S_- | S_+ + S_-  ~  Binomial(T, rho E_- / (E_+ + rho E_-)),

so the conditional likelihood identifies `rho = I(-)/I(+)` and nothing else.
This is the Mantel-Haenszel / conditional-Poisson construction (Andersen 1970;
Hausman, Hall & Griliches 1984), which Lancaster (2000) notes is one of the few
models free of incidental-parameter bias -- and per-observation scales are
incidental parameters in the Neyman-Scott sense, since their number grows with
the data.

Measured on hewl_1118 at 0.02 A bins: 95.6% of strata hold both signs,
covering 99.3% of observations, with about 1.03M observations of paired
information, so the estimator does not starve.

Input is the dump `merge_dump_terms` writes during `finalize_merge`: the signal
counts `S_i` and the exposure `E_i` per observation, with the keys saying which
reflection, which mate and which wavelength each belongs to.

Usage
-----
    uv run python scripts/stratified_bijvoet.py --dump merge_terms.pt \\
        --out merged_strat.mtz --cell 78.3121 78.3121 37.2343 90 90 90 \\
        --space-group "P 43 21 2"
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np
import reciprocalspaceship as rs
import torch


def _arr(dump: dict, key: str) -> np.ndarray:
    v = dump[key]
    return v.numpy() if torch.is_tensor(v) else np.asarray(v)


def _strata(group: np.ndarray, lam: np.ndarray, width: float) -> np.ndarray:
    """A stratum id per observation, from (pooled reflection, wavelength bin)."""
    b = np.floor((lam - lam.min()) / width).astype(np.int64)
    key = group.astype(np.int64) * (b.max() + 1) + b
    return np.unique(key, return_inverse=True)[1]


def _sums(index: np.ndarray, value: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, index, value)
    return out


def conditional_ratio(
    group_of_stratum: np.ndarray,
    s_plus: np.ndarray,
    s_minus: np.ndarray,
    e_plus: np.ndarray,
    e_minus: np.ndarray,
    n_groups: int,
    iters: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Conditional MLE of `rho` per group, and the Fisher information.

    The conditional log-likelihood is concave in `log rho`, so Newton from
    `rho = 1` converges in a few steps.  A stratum holding only one sign has
    `E = 0` on the other side, which sends its term to zero: the common-support
    restriction is automatic rather than imposed.
    """
    log_rho = np.zeros(n_groups, dtype=np.float64)
    total = s_plus + s_minus
    for _ in range(iters):
        rho = np.exp(log_rho[group_of_stratum])
        denom = e_plus + rho * e_minus
        pi = np.where(denom > 0, rho * e_minus / np.maximum(denom, 1e-300), 0.0)
        grad = _sums(group_of_stratum, s_minus * (1.0 - pi) - s_plus * pi, n_groups)
        info = _sums(group_of_stratum, total * pi * (1.0 - pi), n_groups)
        step = np.where(info > 1e-12, grad / np.maximum(info, 1e-12), 0.0)
        log_rho += np.clip(step, -2.0, 2.0)
        if np.abs(step).max() < 1e-10:
            break
    rho = np.exp(log_rho[group_of_stratum])
    denom = e_plus + rho * e_minus
    pi = np.where(denom > 0, rho * e_minus / np.maximum(denom, 1e-300), 0.0)
    info = _sums(group_of_stratum, total * pi * (1.0 - pi), n_groups)
    return np.exp(log_rho), info


def dispersion(
    group: np.ndarray, s: np.ndarray, e: np.ndarray, d: np.ndarray, shells: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-shell multiplicative scatter, by the same moments the model uses.

    `Var(S_i) = mu_i + eps^2 mu_i^2`, so pooling symmetry equivalents gives
    `eps^2` from the excess over Poisson.  Returns the per-shell values and a
    shell index per group.
    """
    n = group.max() + 1
    s_sum, e_sum = _sums(group, s, n), _sums(group, e, n)
    i_hat = np.where(e_sum > 0, s_sum / np.maximum(e_sum, 1e-30), 0.0)
    ss = _sums(group, s * s, n)
    se = _sums(group, s * e, n)
    ee = _sums(group, e * e, n)
    n_obs = _sums(group, np.ones_like(s), n)
    pearson = ss - 2.0 * i_hat * se + i_hat**2 * ee
    mu_sum = i_hat * e_sum
    excess = (pearson - mu_sum) * np.where(n_obs > 1, n_obs / (n_obs - 1.0), 1.0)
    mu_sq = i_hat**2 * ee

    d_group = _sums(group, d, n) / np.maximum(n_obs, 1.0)
    order = np.argsort(-d_group)
    shell_of = np.zeros(n, dtype=np.int64)
    edges = np.linspace(0, n, shells + 1).astype(int)
    for k in range(shells):
        shell_of[order[edges[k] : edges[k + 1]]] = k
    eps_sq = np.zeros(shells)
    for k in range(shells):
        sel = shell_of == k
        denom = mu_sq[sel].sum()
        if denom > 0:
            eps_sq[k] = max(float(excess[sel].sum() / denom), 0.0)
    return eps_sq, shell_of


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Estimate the Bijvoet ratio by conditioning on "
        "wavelength strata."
    )
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell", type=float, nargs=6, required=True)
    ap.add_argument("--space-group", type=str, required=True)
    ap.add_argument(
        "--lambda-bin",
        type=float,
        default=0.02,
        help="Stratum width in A. Narrow strata cancel more of the nuisance "
        "and hold less information; 0.02 keeps 99.3%% of hewl_1118's "
        "observations in two-signed strata.",
    )
    ap.add_argument("--shells", type=int, default=10)
    ap.add_argument(
        "--no-dispersion",
        action="store_true",
        help="Report Poisson-only errors, which are known to be optimistic.",
    )
    args = ap.parse_args()

    dump = torch.load(args.dump, map_location="cpu", weights_only=False)
    s = _arr(dump, "sig_counts").astype(np.float64)
    e = _arr(dump, "exposure").astype(np.float64)
    asu = _arr(dump, "asu_id").astype(np.int64)
    nonanom = _arr(dump, "nonanom_id").astype(np.int64)
    lam = _arr(dump, "wavelength").astype(np.float64)
    d = _arr(dump, "d").astype(np.float64)
    hkl = np.stack([_arr(dump, k) for k in ("H", "K", "L")], axis=1).astype(np.int32)
    centric = (
        _arr(dump, "centric").astype(bool)
        if "centric" in dump
        else np.zeros(len(s), dtype=bool)
    )
    ok = np.isfinite(s) & np.isfinite(e) & (e > 0)
    s, e, asu, nonanom, lam, d, hkl, centric = (
        v[ok] for v in (s, e, asu, nonanom, lam, d, hkl, centric)
    )
    print(f"{len(s)} observations, {len(np.unique(asu))} ids, "
          f"{len(np.unique(nonanom))} Friedel-pooled reflections")

    uniq_pool, pool = np.unique(nonanom, return_inverse=True)
    n_pool = len(uniq_pool)
    # the sign is which of the pair's two ids an observation carries; which one
    # is I(+) is settled downstream by `unstack_anomalous`, so the labelling
    # only has to be consistent
    lo_id = np.full(n_pool, asu.max() + 1, dtype=np.int64)
    np.minimum.at(lo_id, pool, asu)
    is_minus = (asu != lo_id[pool]).astype(np.float64)

    eps_sq, shell_of = dispersion(pool, s, e, d, args.shells)
    print("eps per shell (low to high resolution): "
          + " ".join(f"{v**0.5:.3f}" for v in eps_sq))

    strat = _strata(pool, lam, args.lambda_bin)
    n_strat = strat.max() + 1
    g_of_s = np.zeros(n_strat, dtype=np.int64)
    g_of_s[strat] = pool
    s_plus = _sums(strat, s * (1 - is_minus), n_strat)
    s_minus = _sums(strat, s * is_minus, n_strat)
    e_plus = _sums(strat, e * (1 - is_minus), n_strat)
    e_minus = _sums(strat, e * is_minus, n_strat)
    both = (e_plus > 0) & (e_minus > 0)
    print(f"{int(both.sum())} of {n_strat} strata hold both mates "
          f"({100 * both.mean():.1f}%), "
          f"{100 * (s_plus + s_minus)[both].sum() / (s_plus + s_minus).sum():.1f}% "
          "of the signal")

    rho, info = conditional_ratio(
        g_of_s, s_plus, s_minus, e_plus, e_minus, n_pool
    )
    # pooled intensity, prior-free: the prior's shrinkage would mix into rho
    s_sum, e_sum = _sums(pool, s, n_pool), _sums(pool, e, n_pool)
    i_hat = np.where(e_sum > 0, s_sum / np.maximum(e_sum, 1e-30), 0.0)
    n_obs = _sums(pool, np.ones_like(s), n_pool)
    ee = _sums(pool, e * e, n_pool)
    n_eff = np.where(ee > 0, e_sum**2 / np.maximum(ee, 1e-30), 1.0)

    rel_var = 1.0 / np.maximum(s_sum, 1.0)
    var_log_rho = 1.0 / np.maximum(info, 1e-12)
    if not args.no_dispersion:
        rel_var = rel_var + eps_sq[shell_of] / np.maximum(n_eff, 1.0)
        # quasi-likelihood: the conditional information is deflated by the same
        # overdispersion, with mu_bar the mean counts per observation
        mu_bar = np.where(n_obs > 0, s_sum / np.maximum(n_obs, 1.0), 0.0)
        var_log_rho = var_log_rho * (1.0 + eps_sq[shell_of] * mu_bar)
    sig_i = i_hat * np.sqrt(np.maximum(rel_var, 0.0))

    centric_pool = np.zeros(n_pool, dtype=bool)
    centric_pool[pool[centric]] = True
    acentric = ~centric_pool & (info > 0)
    print(f"\nrho on {int(acentric.sum())} acentric reflections: "
          f"median {np.median(rho[acentric]):.4f}, "
          f"IQR {np.percentile(rho[acentric], 25):.4f}-"
          f"{np.percentile(rho[acentric], 75):.4f}")
    z = np.log(rho[acentric]) / np.sqrt(np.maximum(var_log_rho[acentric], 1e-30))
    print(f"  |log rho| / sigma: median {np.median(np.abs(z)):.3f} "
          "(1.0 would mean the differences are all signal)")
    if centric_pool.sum() > 20:
        zc = np.log(np.clip(rho[centric_pool], 1e-6, 1e6))
        print(f"  centric control (rho must be 1): median {np.median(np.exp(zc)):.4f}, "
              f"sd of log {zc.std():.4f}")

    # rho = I_minus / I_plus with the mean held at i_hat
    i_plus = 2.0 * i_hat / (1.0 + rho)
    i_minus = 2.0 * i_hat * rho / (1.0 + rho)
    var_i = (i_hat * np.sqrt(np.maximum(rel_var, 0.0))) ** 2
    d_plus_rho = -2.0 * i_hat / (1.0 + rho) ** 2
    d_minus_rho = 2.0 * i_hat / (1.0 + rho) ** 2
    var_rho = rho**2 * var_log_rho
    sig_plus = np.sqrt((2.0 / (1.0 + rho)) ** 2 * var_i + d_plus_rho**2 * var_rho)
    sig_minus = np.sqrt(
        (2.0 * rho / (1.0 + rho)) ** 2 * var_i + d_minus_rho**2 * var_rho
    )
    # centrics have I(+) = I(-) by symmetry
    i_plus[centric_pool] = i_hat[centric_pool]
    i_minus[centric_pool] = i_hat[centric_pool]
    sig_plus[centric_pool] = sig_i[centric_pool]
    sig_minus[centric_pool] = sig_i[centric_pool]

    # one row per asu_id, then let unstack_anomalous place the columns
    first = np.full(asu.max() + 1, -1, dtype=np.int64)
    first[asu[::-1]] = np.arange(len(asu))[::-1]
    have = first >= 0
    rows_asu = np.flatnonzero(have)
    rows_obs = first[rows_asu]
    row_pool = pool[rows_obs]
    row_is_minus = is_minus[rows_obs].astype(bool)
    intensity = np.where(row_is_minus, i_minus[row_pool], i_plus[row_pool])
    sigma = np.where(row_is_minus, sig_minus[row_pool], sig_plus[row_pool])

    ds = rs.DataSet(
        {
            "H": hkl[rows_obs, 0],
            "K": hkl[rows_obs, 1],
            "L": hkl[rows_obs, 2],
            "I": intensity.astype(np.float32),
            "SIGI": np.maximum(sigma, 1e-6).astype(np.float32),
        },
        cell=gemmi.UnitCell(*args.cell),
        spacegroup=gemmi.SpaceGroup(args.space_group),
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)
    keep = np.isfinite(ds["I"].to_numpy()) & (ds["SIGI"].to_numpy() > 0)
    ds = ds[keep]
    out = ds.unstack_anomalous()
    out = out[["I(+)", "SIGI(+)", "I(-)", "SIGI(-)"]]
    out.rename(
        columns={"SIGI(+)": "SigI(+)", "SIGI(-)": "SigI(-)"}, inplace=True
    )
    before = len(out)
    out = out[np.isfinite(out["I(+)"].to_numpy()) | np.isfinite(out["I(-)"].to_numpy())]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_mtz(str(args.out), skip_problem_mtztypes=True)
    print(f"\nwrote {args.out}: {len(out)} reflections "
          f"({before - len(out)} dropped with neither mate)")


if __name__ == "__main__":
    main()
