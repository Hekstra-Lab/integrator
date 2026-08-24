"""Diagnose where the anomalous signal is lost between a learned scale+merge
and a DIALS scale+merge, both run on the *same* integrator intensities.

Two parameterization-free diagnostics, motivated by the decomposition of any
per-observation scale `s_i` into a part that is symmetric under the Friedel
involution `h -> -h` (which cannot move the anomalous difference) and a part
that is antisymmetric under it (the only channel that can absorb anomalous
signal):

`regression`
    Errors-in-variables comparison of the merged anomalous differences
    `DANO = I(+) - I(-)` from the learned model against the DIALS oracle.
    Reports the ordinary-least-squares slope, the standardized-major-axis
    (SMA) slope, and the Pearson correlation, per resolution shell.  The SMA
    slope and the correlation jointly separate the two failure modes:

        SMA slope ~ 1, low correlation   -> unbiased but noisy
                                            (fix: error model / weighting /
                                             outlier rejection -- the merge)
        SMA slope < 1                    -> the scale is attenuating DANO
                                            (fix: anchored Friedel-odd channel)

`parity`
    Sizes the anomalous-dangerous channel directly from the DIALS
    per-observation scale field.  For each merged reflection it splits the
    observations into the two Friedel classes and measures the systematic
    difference of mean log-scale between them, `delta = 1/2 (s_bar(+) -
    s_bar(-))`, correcting for the finite-multiplicity noise floor with an
    analytic, finite-population-exact null.  The reported ratio
    `Var_systematic(delta) / Var(log s)` is the fraction of the total scale
    variation that lives in the Friedel-odd subspace -- i.e. how much the
    learned scale must be allowed to correct there, and no more.

Both diagnostics read MTZ files with `reciprocalspaceship`; no DIALS libraries
are required.  Stage the files on the cluster and run, e.g.::

    uv run python scripts/scale_anomalous_diagnostics.py regression \\
        --dials-mtz dials_scaled_on_my_intensities.mtz \\
        --learned-mtz merged.mtz --plot regression.png

    uv run python scripts/scale_anomalous_diagnostics.py parity \\
        --unmerged-mtz dials_scaled_unmerged.mtz --plot parity.png

The unmerged MTZ for `parity` must carry a per-observation scale column.  DIALS
writes one via `dials.scale ... output.unmerged_mtz=scaled_unmerged.mtz`
(column `SCALEUSED`) or `dials.export intensity=scale`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import reciprocalspaceship as rs

# Candidate column names, in priority order, for autodetection.
_SCALE_COLS = ["SCALEUSED", "SCALE", "inverse_scale_factor", "ISCALE"]
_IPLUS_COLS = ["I(+)", "IMEAN(+)", "Iplus"]
_IMINUS_COLS = ["I(-)", "IMEAN(-)", "Iminus"]
_FPLUS_COLS = ["F(+)", "Fplus"]
_FMINUS_COLS = ["F(-)", "Fminus"]
_SIGIPLUS_COLS = ["SIGI(+)", "SIGIplus"]
_SIGIMINUS_COLS = ["SIGI(-)", "SIGIminus"]


def _first_present(ds, candidates):
    """Return the first column name in `candidates` present in `ds`, else None."""
    cols = set(map(str, ds.columns))
    for c in candidates:
        if c in cols:
            return c
    return None


def _quantile_bins(values, n_bins):
    """Assign `values` to `n_bins` equal-count bins; return integer labels.

    Bins are ordered by increasing `values` (so for dHKL, bin 0 is low
    resolution).  Ties are broken arbitrarily; bins may be slightly uneven.
    """
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=int)
    edges = np.linspace(0, len(values), n_bins + 1).astype(int)
    for b in range(n_bins):
        labels[order[edges[b] : edges[b + 1]]] = b
    return labels


def _bin_label(dhkl, labels, b):
    """Human-readable resolution range for bin `b`."""
    m = labels == b
    if not m.any():
        return "empty"
    return f"{dhkl[m].max():5.2f}-{dhkl[m].min():4.2f} A"


# --------------------------------------------------------------------------- #
# Diagnostic 2: errors-in-variables regression of learned vs DIALS DANO        #
# --------------------------------------------------------------------------- #


def _load_dano(path, label):
    """Load merged anomalous data from an MTZ and return a frame with `DANO`.

    Returns a `rs.DataSet` indexed by ASU `(H, K, L)` with columns `DANO`
    (intensity difference `I(+) - I(-)`), `IMEAN` (the Friedel-even mean,
    used as a join sanity check), `SIGDANO` when sigmas are available, and
    `dHKL`.  Falls back to amplitude differences `F(+) - F(-)` if no intensity
    columns are found.
    """
    ds = rs.read_mtz(str(path))
    ds = ds.compute_dHKL()

    ip, im = _first_present(ds, _IPLUS_COLS), _first_present(ds, _IMINUS_COLS)
    fp, fm = _first_present(ds, _FPLUS_COLS), _first_present(ds, _FMINUS_COLS)

    if ip and im:
        plus, minus, kind = ds[ip].to_numpy(float), ds[im].to_numpy(float), "I"
    elif fp and fm:
        plus, minus, kind = ds[fp].to_numpy(float), ds[fm].to_numpy(float), "F"
    else:
        sys.exit(
            f"[{label}] {path}: no anomalous columns found.\n"
            f"  looked for {_IPLUS_COLS} / {_IMINUS_COLS} (or F variants).\n"
            f"  available columns: {list(map(str, ds.columns))}"
        )

    dano = plus - minus
    imean = 0.5 * (plus + minus)

    out = rs.DataSet(
        {"DANO": dano, "IMEAN": imean, "dHKL": ds["dHKL"].to_numpy(float)},
        cell=ds.cell,
        spacegroup=ds.spacegroup,
    )
    out.index = ds.index

    sp, sm = _first_present(ds, _SIGIPLUS_COLS), _first_present(ds, _SIGIMINUS_COLS)
    if sp and sm:
        out["SIGDANO"] = np.sqrt(
            ds[sp].to_numpy(float) ** 2 + ds[sm].to_numpy(float) ** 2
        )

    finite = np.isfinite(out["DANO"].to_numpy()) & np.isfinite(out["IMEAN"].to_numpy())
    out = out[finite]
    print(
        f"[{label}] {path.name}: {len(out):,} anomalous reflections "
        f"({kind}-based DANO), sg={ds.spacegroup.short_name()}"
    )
    return out


def _slopes(x, y, w=None):
    """Return `(ols, sma, r, n)` for `y ~ x`, optionally weighted by `w`.

    `ols` is the ordinary-least-squares slope of `y` on `x` (attenuated by
    noise in `x`).  `sma` is the standardized-major-axis slope `sign(cov) *
    std_y / std_x`, the symmetric errors-in-variables estimate.  `r` is the
    weighted Pearson correlation.  `ols = r * sma` exactly.
    """
    if w is None:
        w = np.ones_like(x)
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    dx, dy = x - mx, y - my
    vx, vy = (w * dx * dx).sum() / sw, (w * dy * dy).sum() / sw
    cov = (w * dx * dy).sum() / sw
    if vx <= 0 or vy <= 0:
        return float("nan"), float("nan"), float("nan"), len(x)
    ols = cov / vx
    sma = np.sign(cov) * np.sqrt(vy / vx)
    r = cov / np.sqrt(vx * vy)
    return ols, sma, r, len(x)


def diagnostic_regression(args):
    """Run the learned-vs-DIALS DANO regression, overall and per shell."""
    dials = _load_dano(args.dials_mtz, "dials")
    learned = _load_dano(args.learned_mtz, "learned")

    joined = dials.join(learned, how="inner", lsuffix="_d", rsuffix="_l")
    n = len(joined)
    if n < 50:
        sys.exit(
            f"Only {n} reflections in common after join on (H,K,L). "
            "Are the two MTZs in the same space-group setting / ASU? "
            "Check that both came from the same reduction."
        )

    xd = joined["DANO_d"].to_numpy(float)
    yl = joined["DANO_l"].to_numpy(float)
    dhkl = joined["dHKL_d"].to_numpy(float)

    xm_d = joined["IMEAN_d"].to_numpy(float)
    xm_l = joined["IMEAN_l"].to_numpy(float)
    # The Friedel-EVEN (mean) SMA slope is the global units factor between the
    # two datasets.  Dividing the DANO slope by it removes any overall scale /
    # normalization difference, leaving the anomalous-SPECIFIC attenuation --
    # the units-free bias number.  ratio ~ 1 => DANO preserved as well as the
    # mean (no signal-eating, just units);  ratio << 1 => real attenuation.
    _, sma_mean, r_mean, _ = _slopes(xm_d, xm_l)
    _, sma, r, _ = _slopes(xd, yl)
    ratio = sma / sma_mean if sma_mean else float("nan")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2  --  learned DANO  vs  DIALS DANO (same intensities)")
    print("=" * 70)
    print(f"  reflections in common : {n:,}")
    print(f"  Friedel-even mean     : SMA={sma_mean:8.4f}  r={r_mean:6.4f}  "
          f"(global units factor / join check)")
    print(f"  anomalous DANO        : SMA={sma:8.4f}  r={r:6.4f}  "
          f"(r = pooled CC_anom)")
    print(f"  units-free attenuation: SMA(DANO)/SMA(IMEAN) = {ratio:6.3f}  "
          f"(~1 good, <<1 = signal eaten)")

    print("\n  per resolution shell (low -> high):")
    print("    shell            n    r_dano  SMA_dano  SMA_mean   ratio")
    labels = _quantile_bins(dhkl, args.bins)
    shell_stats = []
    for b in range(args.bins):
        m = labels == b
        if m.sum() < 20:
            continue
        _, sb, rb, nb = _slopes(xd[m], yl[m])
        _, sbm, _, _ = _slopes(xm_d[m], xm_l[m])
        rt = sb / sbm if sbm else float("nan")
        rng = _bin_label(dhkl, labels, b)
        print(f"    {rng:14s} {nb:7d}  {rb:6.3f}  {sb:8.4f}  {sbm:7.3f}  {rt:6.3f}")
        if np.isfinite(rb) and rb > 0.05:
            shell_stats.append((rb, rt))

    print("\n  interpretation (per-shell medians; robust to pooling/bad shells):")
    if not shell_stats:
        print("    no usable shells -- check the (+)/(-) convention and the join.")
    else:
        med_r = float(np.median([s[0] for s in shell_stats]))
        med_ratio = float(np.median([s[1] for s in shell_stats]))
        print(f"    median within-shell CC_anom = {med_r:.3f}, "
              f"attenuation ratio = {med_ratio:.3f}")
        if r < 0.5 * med_r:
            print(f"    * pooled r={r:.2f} << within-shell r={med_r:.2f}: DANO is")
            print( "      mis-scaled ACROSS resolution (Wilson-B-like).  That alone")
            print( "      blurs the anomalous map -- fix the per-shell/overall scale.")
        if med_ratio < 0.5:
            print(f"    => anomalous signal ATTENUATED to ~{100 * med_ratio:.0f}% of the")
            print( "       oracle beyond the units factor: real bias.  Fix = restrained,")
            print( "       centric-anchored Friedel-odd channel.")
        elif med_r < 0.6:
            print(f"    => roughly UNBIASED (ratio~{med_ratio:.2f}) but NOISY "
                  f"(CC_anom={med_r:.2f}).")
            print( "       Fix = error model + inverse-variance weighting + outlier")
            print( "       rejection in the merge, NOT the scale function.")
        else:
            print(f"    => strong recovery (CC_anom={med_r:.2f}, ratio~{med_ratio:.2f});")
            print( "       remaining gap is the cross-resolution scale + bad shells.")

    if args.plot:
        _plot_regression(xd, yl, sma, r, Path(args.plot))


def _plot_regression(x, y, sma, r, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib unavailable; skipping {path})")
        return
    lim = np.nanpercentile(np.abs(np.concatenate([x, y])), 99)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=3, alpha=0.2, edgecolors="none")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y=x")
    ax.plot([-lim, lim], [-sma * lim, sma * lim], "r-", lw=1.2,
            label=f"SMA slope={sma:.2f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("DIALS DANO (oracle)")
    ax.set_ylabel("learned DANO")
    ax.set_title(f"r={r:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# Diagnostic 1: size the Friedel-odd channel of the DIALS per-obs scale field  #
# --------------------------------------------------------------------------- #


def diagnostic_parity(args):
    """Measure the systematic Friedel-odd content of the per-obs scale field."""
    ds = rs.read_mtz(str(args.unmerged_mtz))
    scol = args.scale_col or _first_present(ds, _SCALE_COLS)
    if scol is None:
        sys.exit(
            f"{args.unmerged_mtz}: no per-observation scale column found.\n"
            f"  looked for {_SCALE_COLS}.\n"
            f"  available: {list(map(str, ds.columns))}\n"
            "  Re-run DIALS with output.unmerged_mtz= (gives SCALEUSED), or "
            "pass --scale-col."
        )
    print(f"using scale column '{scol}' from {args.unmerged_mtz.name} "
          f"({len(ds):,} observations)")

    ds = ds.compute_dHKL()
    ds.label_centrics(inplace=True)
    # Map Friedel mates onto a shared (Laue) ASU index; M/ISYM parity then
    # labels the two Friedel classes.  Variance of `delta` is invariant to
    # which class we call (+), so the absolute ISYM convention is irrelevant.
    ds.hkl_to_asu(inplace=True)

    scale = ds[scol].to_numpy(float)
    good = np.isfinite(scale) & (scale > 0)
    df = rs.DataSet(
        {
            "LOGS": np.log(np.where(good, scale, 1.0)),
            "PAR": (ds["M/ISYM"].to_numpy(int) % 2).astype(int),
            "CEN": ds["CENTRIC"].to_numpy(bool).astype(int),
            "dHKL": ds["dHKL"].to_numpy(float),
        }
    )
    df.index = ds.index
    df = df[good]
    df = df.reset_index()
    hkl_cols = [c for c in ["H", "K", "L"] if c in df.columns][:3]

    total_var = float(np.var(df["LOGS"].to_numpy()))
    n_plus = int((df["PAR"] == 1).sum())
    print(f"  Friedel class split: {n_plus:,} / {len(df) - n_plus:,} "
          f"(should be roughly balanced)")
    print(f"  Var(log s) total          : {total_var:.5f}")

    _parity_report("ALL reflections", df, hkl_cols, total_var, args)
    cen = df[df["CEN"] == 1]
    acen = df[df["CEN"] == 0]
    if len(cen) > 100:
        _parity_report("centrics (true DANO = 0; pure scale)", cen, hkl_cols,
                       total_var, args)
    _parity_report("acentrics", acen, hkl_cols, total_var, args)

    if args.bins > 1:
        print("\n  acentric Friedel-odd fraction by resolution shell:")
        print("    shell            n_hkl   frac_odd   obs/null")
        labels = _quantile_bins(acen["dHKL"].to_numpy(float), args.bins)
        acen = acen.copy()
        acen["_BIN"] = labels
        for b in range(args.bins):
            sub = acen[acen["_BIN"] == b]
            if len(sub) < 200:
                continue
            frac, ratio, nh = _odd_fraction(sub, hkl_cols, total_var, args.min_mult)
            if nh == 0:
                continue
            rng = _bin_label(acen["dHKL"].to_numpy(float), labels, b)
            print(f"    {rng:14s} {nh:7d}   {frac:8.4f}   {ratio:6.2f}")


def _odd_fraction(df, hkl_cols, total_var, min_mult):
    """Return `(frac_odd, obs_over_null, n_hkl)` for one subset of observations.

    `frac_odd` is the noise-corrected systematic variance of the per-reflection
    antisymmetric log-scale `delta`, expressed as a fraction of `total_var`.
    `obs_over_null` is the observed `Var(delta)` divided by the analytic
    finite-multiplicity null; values near 1 mean the apparent asymmetry is pure
    sampling noise.
    """
    side = (
        df.groupby(hkl_cols + ["PAR"])["LOGS"]
        .agg(mean="mean", n="count")
        .reset_index()
    )
    piv = side.pivot_table(index=hkl_cols, columns="PAR", values=["mean", "n"])
    if ("mean", 0) not in piv.columns or ("mean", 1) not in piv.columns:
        return 0.0, float("nan"), 0
    mp, mm = piv[("mean", 1)], piv[("mean", 0)]
    npn, nmn = piv[("n", 1)], piv[("n", 0)]
    s2 = df.groupby(hkl_cols)["LOGS"].var(ddof=1)

    tab = rs.DataSet({"mp": mp, "mm": mm, "np": npn, "nm": nmn})
    tab["s2"] = s2
    tab = tab.dropna()
    tab = tab[(tab["np"] >= min_mult) & (tab["nm"] >= min_mult)]
    n_hkl = len(tab)
    if n_hkl < 20:
        return 0.0, float("nan"), n_hkl

    delta = 0.5 * (tab["mp"].to_numpy() - tab["mm"].to_numpy())
    obs = float(np.mean(delta**2))  # E[delta]~0, so this is Var(delta)
    null = float(np.mean(
        0.25 * tab["s2"].to_numpy()
        * (1.0 / tab["np"].to_numpy() + 1.0 / tab["nm"].to_numpy())
    ))
    systematic = max(0.0, obs - null)
    frac = systematic / total_var if total_var > 0 else float("nan")
    ratio = obs / null if null > 0 else float("nan")
    return frac, ratio, n_hkl


def _parity_report(title, df, hkl_cols, total_var, args):
    frac, ratio, n_hkl = _odd_fraction(df, hkl_cols, total_var, args.min_mult)
    print(f"\n  [{title}]  ({n_hkl:,} reflections with mult >= {args.min_mult}/side)")
    if n_hkl < 20:
        print("    too few multi-observed reflections to estimate.")
        return
    print(f"    Var(delta) observed / null : {ratio:6.2f}")
    print(f"    systematic Friedel-odd frac: {frac:.5f}  "
          f"({100 * frac:.2f}% of Var(log s))")
    if frac < 0.002:
        print("    => Friedel-odd channel is negligible: the scale barely")
        print("       differs between Friedel mates.  An odd-scale correction")
        print("       will NOT recover the gap; look to the merge instead.")
    else:
        print("    => a real Friedel-odd channel exists; a restrained, centric-")
        print("       anchored odd-scale term should help, and this fraction")
        print("       bounds how much capacity it needs.")


# --------------------------------------------------------------------------- #


def build_parser():
    p = argparse.ArgumentParser(
        description="Anomalous scale/merge diagnostics (regression, parity).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("regression", help="learned vs DIALS DANO (Diagnostic 2)")
    r.add_argument("--dials-mtz", type=Path, required=True,
                   help="merged anomalous MTZ: DIALS scaling on your intensities")
    r.add_argument("--learned-mtz", type=Path, required=True,
                   help="merged anomalous MTZ from your model (extract_merged_mtz.py)")
    r.add_argument("--bins", type=int, default=10, help="resolution shells")
    r.add_argument("--plot", type=str, default=None, help="output scatter PNG")
    r.set_defaults(func=diagnostic_regression)

    a = sub.add_parser("parity", help="size the Friedel-odd scale channel (Diagnostic 1)")
    a.add_argument("--unmerged-mtz", type=Path, required=True,
                   help="unmerged DIALS MTZ with a per-observation scale column")
    a.add_argument("--scale-col", type=str, default=None,
                   help=f"scale column name (auto: {_SCALE_COLS})")
    a.add_argument("--min-mult", type=int, default=2,
                   help="min observations per Friedel class to use a reflection")
    a.add_argument("--bins", type=int, default=8, help="resolution shells")
    a.add_argument("--plot", type=str, default=None, help="(reserved)")
    a.set_defaults(func=diagnostic_parity)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
