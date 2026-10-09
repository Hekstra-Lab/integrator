"""Post-hoc anomalous merge on already-scaled, unmerged intensities.

Holds integration and scaling fixed (any scaler's unmerged MTZ) and varies only
the merge of Friedel mates, so the anomalous-preserving ideas can be compared
head to head against a plain inverse-variance merge on identical input:

  - "dials_like": inverse-variance merge each mate, no coupling. Reproduces a
    plain anomalous merge; the correctness control.
  - "eb_delta": the empirical-Bayes anomalous-difference latent. Writes
    I(+/-) = I0 * (1 +/- mu_delta), where the common mode I0 is the pooled
    inverse-variance mean and the signed fraction delta = (m+ - m-)/(m+ + m-)
    is shrunk by w = sigma_delta^2 / (sigma_delta^2 + v_h). sigma_delta^2 is a
    method-of-moments estimate over acentric pairs; centrics are pinned to 0.
    A faithful post-hoc port of SVAEDifferenceMergingIntegrator._eb_delta.
  - "double_wilson": the tractable double-Wilson coupling -- shrink the mates'
    log-ratio r = log(m+) - log(m-) toward zero by w = s^2 / (s^2 + var_r),
    s^2 method-of-moments over acentric pairs, then I(+/-) = I0 * exp(+/- mu_r/2)
    so the geometric mean stays I0. The log-space cousin of eb_delta.

Both shrinkages act ONLY on the Bijvoet difference; the common mode is the plain
merged intensity, so a Wilson prior on |F|^2 never touches the difference.
"""

from __future__ import annotations

import argparse

import numpy as np
import reciprocalspaceship as rs

_DELTA_CLAMP = 0.95
_EPS = 1e-8


def _pick(ds, names):
    for n in names:
        if n in ds.columns:
            return n
    raise KeyError(f"none of {names} in {list(ds.columns)}")


def _group_sum(group: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(group, weights=weights, minlength=n)[:n]


def _prepare(mtz_in: str):
    """Return per-observation arrays keyed to a contiguous pooled (ASU) id."""
    ds = rs.read_mtz(mtz_in)
    icol = _pick(ds, ["I", "IOBS", "I.scale", "IMEAN"])
    scol = _pick(ds, ["SIGI", "SIGIOBS", "SIGI.scale", "SIGIMEAN"])
    ds = ds.dropna(subset=[icol, scol])
    ds = ds.compute_dHKL().label_centrics()
    # Map to the reciprocal ASU: Friedel mates share the ASU hkl (the pooled id)
    # and are told apart by ISYM parity (odd = F(+)), as in add_friedel_metadata.
    ds = ds.hkl_to_asu()
    isym = np.asarray(ds[_pick(ds, ["M/ISYM"])]).astype(np.int64)
    plus = (isym % 2 == 1)

    hkl = ds.get_hkls().astype(np.int64)
    key = (hkl[:, 0].astype(np.int64) << 42) ^ (hkl[:, 1].astype(np.int64) << 21) ^ hkl[:, 2]
    uniq_key, pooled = np.unique(key, return_inverse=True)
    # representative hkl + resolution + centric per pooled group (first obs)
    order = np.argsort(pooled, kind="stable")
    first = np.zeros(uniq_key.size, np.int64)
    first[pooled[order]] = order  # last write per group under stable sort
    rep_hkl = hkl[first]
    centric = np.asarray(ds["CENTRIC"]).astype(bool)[first]
    d = np.asarray(ds["dHKL"]).astype(np.float64)[first]

    return {
        "cell": ds.cell,
        "spacegroup": ds.spacegroup,
        "pooled": pooled.astype(np.int64),
        "n_groups": uniq_key.size,
        "I": np.asarray(ds[icol]).astype(np.float64),
        "sig": np.asarray(ds[scol]).astype(np.float64),
        "plus": plus,
        "rep_hkl": rep_hkl,
        "centric": centric,
        "d": d,
    }


def _shell_sigma2(stat, v, acen, d, n_shell):
    """Per-resolution-shell method-of-moments prior variance for a Bijvoet
    statistic: sigma^2(shell) = <stat^2 - v> over that shell's acentric pairs,
    floored at a small positive value. A single global value collapses to the
    floor for weak S-SAD because the many noisy high-res pairs dominate the
    average; binning by resolution keeps the low-res shells where the signal is.
    Returns (sigma2_per_group, diagnostic_rows).
    """
    s2 = np.full(stat.shape, 1e-8)
    dd = np.where(np.isfinite(d), d, np.nanmedian(d[acen]))
    edges = np.quantile(dd[acen], np.linspace(0, 1, n_shell + 1))
    binidx = np.clip(np.searchsorted(edges[1:-1], dd), 0, n_shell - 1)
    rows = []
    for b in range(n_shell):
        inb = binidx == b
        m = acen & inb
        if m.sum() >= 50:
            val = float(np.clip(np.mean(stat[m] ** 2 - v[m]), 1e-8, None))
        else:
            val = 1e-8
        s2[inb] = val
        if m.sum() > 0:
            wt = val / (val + np.clip(float(np.median(v[m])), _EPS, None))
            rows.append((
                round(float(edges[b]), 2), round(float(edges[b + 1]), 2),
                int(m.sum()), float(np.mean(stat[m] ** 2)),
                float(np.mean(v[m])), float(np.sqrt(val)), float(wt),
            ))
    return s2, rows


def _print_diag(title, rows):
    """Per-shell anomalous SNR: is <stat^2> above the noise floor <v> anywhere?"""
    print(f"  [{title}] per-shell anomalous variance vs noise:")
    print(f"    {'d_lo':>6} {'d_hi':>6} {'n':>6} {'<stat^2>':>10} "
          f"{'<noise_v>':>10} {'sigma':>8} {'w':>6}")
    for dlo, dhi, nn, ms, mv, sg, wt in rows:
        print(f"    {dlo:6.2f} {dhi:6.2f} {nn:6d} {ms:10.2e} {mv:10.2e} "
              f"{sg:8.4f} {wt:6.3f}")


def _ivw(pooled, mask, I, w, n):
    """Inverse-variance mean and variance per group over the masked subset."""
    sw = _group_sum(pooled[mask], w[mask], n)
    swi = _group_sum(pooled[mask], (w * I)[mask], n)
    have = sw > 0
    m = np.full(n, np.nan)
    var = np.full(n, np.nan)
    m[have] = swi[have] / sw[have]
    var[have] = 1.0 / sw[have]
    return m, var, have


def _denoise(pooled, I, sig, plus, centric, d, n, mode, n_shell, obs_mask, quiet):
    """Merge Friedel mates (inverse-variance) then apply the mode's Bijvoet
    shrink, over the observations selected by `obs_mask`. Returns per-group
    arrays. Factored out so the whole pipeline can be re-run on random halves
    for a half-set cc_anom."""
    w = 1.0 / np.maximum(sig, _EPS) ** 2
    I0, var0, have0 = _ivw(pooled, obs_mask, I, w, n)
    m_p, var_p, have_p = _ivw(pooled, obs_mask & plus, I, w, n)
    m_m, var_m, have_m = _ivw(pooled, obs_mask & ~plus, I, w, n)
    both = have_p & have_m
    Ip = np.where(have_p, m_p, np.nan)
    Im = np.where(have_m, m_m, np.nan)
    Sp = np.where(have_p, np.sqrt(var_p), np.nan)
    Sm = np.where(have_m, np.sqrt(var_m), np.nan)
    mean_w = float("nan")

    if mode == "dials_like":
        pass  # raw mate means already in Ip/Im/Sp/Sm
    elif mode == "eb_delta":
        tot = (m_p + m_m).clip(min=_EPS)
        delta_hat = (m_p - m_m) / tot
        v_h = 4.0 / tot**4 * (m_m**2 * var_p + m_p**2 * var_m)
        acen = both & (~centric)
        s2, diag = _shell_sigma2(delta_hat, v_h, acen, d, n_shell)
        w_sh = s2 / (s2 + np.clip(v_h, _EPS, None))
        mu_d = np.clip(w_sh * delta_hat, -_DELTA_CLAMP, _DELTA_CLAMP)
        mu_d = np.where(centric, 0.0, mu_d)
        sd_d = np.sqrt(np.clip(s2 * v_h / (s2 + v_h), 0.0, None))
        Ip[both] = (I0 * (1.0 + mu_d))[both]
        Im[both] = (I0 * (1.0 - mu_d))[both]
        Sp[both] = np.sqrt(var0 * (1.0 + mu_d) ** 2 + I0**2 * sd_d**2)[both]
        Sm[both] = np.sqrt(var0 * (1.0 - mu_d) ** 2 + I0**2 * sd_d**2)[both]
        mean_w = float(np.nanmean(w_sh[acen]))
        if not quiet:
            _print_diag("eb_delta  delta=(I+ - I-)/(I+ + I-)", diag)
    elif mode == "double_wilson":
        r = np.log(m_p.clip(min=_EPS)) - np.log(m_m.clip(min=_EPS))
        var_r = var_p / m_p**2 + var_m / m_m**2
        acen = both & (~centric)
        s2, diag = _shell_sigma2(r, var_r, acen, d, n_shell)
        w_sh = s2 / (s2 + np.clip(var_r, _EPS, None))
        mu_r = np.where(centric, 0.0, w_sh * r)
        sd_r = np.sqrt(np.clip(s2 * var_r / (s2 + var_r), 0.0, None))
        Ip[both] = (I0 * np.exp(0.5 * mu_r))[both]
        Im[both] = (I0 * np.exp(-0.5 * mu_r))[both]
        rel0 = var0 / np.clip(I0**2, _EPS, None)
        Sp[both] = (I0 * np.exp(0.5 * mu_r) * np.sqrt(rel0 + 0.25 * sd_r**2))[both]
        Sm[both] = (I0 * np.exp(-0.5 * mu_r) * np.sqrt(rel0 + 0.25 * sd_r**2))[both]
        mean_w = float(np.nanmean(w_sh[acen]))
        if not quiet:
            _print_diag("double_wilson  r=log(I+/I-)", diag)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return {"Ip": Ip, "Im": Im, "Sp": Sp, "Sm": Sm, "I0": I0, "var0": var0,
            "have0": have0, "have_p": have_p, "have_m": have_m, "both": both,
            "mean_w": mean_w}


def _halfset_ccanom(p, mode, n_shell):
    """Half-set anomalous correlation: split observations at random, run the
    full merge+shrink on each half, correlate the two anomalous differences
    over acentric pairs present in both halves."""
    rng = np.random.default_rng(0)
    half = rng.random(p["I"].shape[0]) < 0.5
    args = (p["pooled"], p["I"], p["sig"], p["plus"], p["centric"], p["d"],
            p["n_groups"], mode, n_shell)
    A = _denoise(*args, half, True)
    B = _denoise(*args, ~half, True)
    dA, dB = A["Ip"] - A["Im"], B["Ip"] - B["Im"]
    sel = (A["both"] & B["both"] & ~p["centric"]
           & np.isfinite(dA) & np.isfinite(dB))
    if sel.sum() < 2:
        return float("nan"), 0
    return float(np.corrcoef(dA[sel], dB[sel])[0, 1]), int(sel.sum())


def merge_anomalous_posthoc(
    mtz_in: str, mtz_out: str, mode: str = "eb_delta", n_shell: int = 10
) -> dict:
    p = _prepare(mtz_in)
    n = p["n_groups"]
    res = _denoise(
        p["pooled"], p["I"], p["sig"], p["plus"], p["centric"], p["d"],
        n, mode, n_shell, np.ones(p["I"].shape[0], bool), False,
    )
    Ip, Im, Sp, Sm = res["Ip"], res["Im"], res["Sp"], res["Sm"]
    I0, var0 = res["I0"], res["var0"]
    have0, have_p, have_m = res["have0"], res["have_p"], res["have_m"]
    cc, n_cc = _halfset_ccanom(p, mode, n_shell)
    stats = {
        "mode": mode, "n_groups": int(n), "n_pairs": int(res["both"].sum()),
        "mean_shrink_w": res["mean_w"], "cc_anom_halfset": round(cc, 4),
        "n_ccanom": n_cc,
    }

    # assemble merged anomalous MTZ, keyed by the pooled ASU hkl
    out = rs.DataSet(
        {
            "H": p["rep_hkl"][:, 0], "K": p["rep_hkl"][:, 1], "L": p["rep_hkl"][:, 2],
            "I(+)": Ip, "SIGI(+)": Sp, "I(-)": Im, "SIGI(-)": Sm,
            "IMEAN": I0, "SIGIMEAN": np.sqrt(var0),
        },
        cell=p["cell"], spacegroup=p["spacegroup"],
    )
    for c in ("H", "K", "L"):
        out[c] = out[c].astype("H")  # rs Miller-index MTZ dtype
    out = out.set_index(["H", "K", "L"])
    keep = have0 & (have_p | have_m)
    out = out.loc[keep]
    for c, dt in [("I(+)", "K"), ("SIGI(+)", "M"), ("I(-)", "K"),
                  ("SIGI(-)", "M"), ("IMEAN", "J"), ("SIGIMEAN", "Q")]:
        out[c] = out[c].astype(dt)
    out.write_mtz(mtz_out)
    stats["n_written"] = int(keep.sum())
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mtz-in", required=True, help="scaled, unmerged MTZ")
    ap.add_argument("--mtz-out", required=True)
    ap.add_argument(
        "--mode", default="eb_delta",
        choices=["dials_like", "eb_delta", "double_wilson"],
    )
    ap.add_argument("--n-shell", type=int, default=10)
    a = ap.parse_args()
    stats = merge_anomalous_posthoc(a.mtz_in, a.mtz_out, a.mode, a.n_shell)
    print("  ".join(f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
