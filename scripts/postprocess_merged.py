"""Post-process a merged anomalous MTZ: ice-ring masking + resolution rescaling.

No retraining required -- operates on an existing `merged.mtz` from the model
and (optionally) a reference merged MTZ.  Two corrections motivated by the
`scale_anomalous_diagnostics` findings, each isolatable:

`--rescale` (needs `--reference-mtz`)
    Fit an isotropic `scale * exp(-2 B s^2)` between the model's mean
    intensities and the reference's (binned, smooth two-parameter fit -- it does
    NOT copy per-reflection values) and apply it, so the model data's resolution
    dependence matches the reference.  Targets the cross-resolution / pooled-
    correlation mismatch (pooled r << within-shell r).

`--ice`
    Drop reflections inside ice-ring resolution bands -- a uniformly
    contaminated shell that the in-merge attention gate cannot reject (no within-
    HKL outlier to gate).  Defaults to hexagonal-ice Ih d-spacings; pass
    `--ice-range LO HI` (repeatable) to add custom bands.

Writes a corrected MTZ for the downstream map/peak calculation.  Run, e.g.::

    uv run python scripts/postprocess_merged.py \\
        --learned-mtz merged.mtz --reference-mtz integrator_dials.mtz \\
        --rescale --ice --out merged_corrected.mtz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import reciprocalspaceship as rs

# Hexagonal ice (Ih) d-spacings (Angstrom), strongest rings.  Half-width applied
# in `_ice_mask`.  The 1.34-1.42 A band flagged by the regression diagnostic is
# covered by the 1.372 / 1.444 entries.
_ICE_D = [
    3.897, 3.669, 3.441, 2.671, 2.249, 2.072, 1.948, 1.918, 1.883,
    1.721, 1.522, 1.473, 1.444, 1.372, 1.304,
]

_IPLUS = ["I(+)", "Iplus"]
_IMINUS = ["I(-)", "Iminus"]
_SIGIPLUS = ["SIGI(+)", "SIGIplus"]
_SIGIMINUS = ["SIGI(-)", "SIGIminus"]
_FPLUS = ["F(+)", "Fplus"]
_FMINUS = ["F(-)", "Fminus"]
_SIGFPLUS = ["SIGF(+)", "SIGFplus"]
_SIGFMINUS = ["SIGF(-)", "SIGFminus"]


def _first(ds, names):
    cols = set(map(str, ds.columns))
    for n in names:
        if n in cols:
            return n
    return None


def _imean(ds):
    """Mean (Friedel-even) intensity per reflection, or None if no I columns."""
    ip, im = _first(ds, _IPLUS), _first(ds, _IMINUS)
    if ip and im:
        return 0.5 * (ds[ip].to_numpy(float) + ds[im].to_numpy(float))
    fp, fm = _first(ds, _FPLUS), _first(ds, _FMINUS)
    if fp and fm:
        return 0.5 * (ds[fp].to_numpy(float) ** 2 + ds[fm].to_numpy(float) ** 2)
    return None


def _fit_scale_b(ds_learn, ds_ref, n_bins):
    """Fit `log(I_ref/I_learn) = c0 - 2 B s^2` on shell-mean intensities.

    Joins the two datasets on `(H, K, L)`, bins the common reflections by
    resolution, and regresses the log shell-mean-intensity ratio on `s^2 =
    1/(4 d^2)`.  Returns `(c0, B)`; the per-reflection correction is then
    `k = exp(c0 - 2 B s^2)`.
    """
    il, ir = _imean(ds_learn), _imean(ds_ref)
    if il is None or ir is None:
        sys.exit("--rescale needs intensity (or amplitude) columns in both MTZs.")
    a = rs.DataSet(
        {"il": il, "d": ds_learn.compute_dHKL()["dHKL"].to_numpy(float)},
        cell=ds_learn.cell,
        spacegroup=ds_learn.spacegroup,
    )
    a.index = ds_learn.index
    b = rs.DataSet({"ir": ir}, cell=ds_ref.cell, spacegroup=ds_ref.spacegroup)
    b.index = ds_ref.index
    j = a.join(b, how="inner")
    j = j[(j["il"] > 0) & (j["ir"] > 0)]
    if len(j) < 200:
        sys.exit(f"only {len(j)} common reflections -- check ASU/setting.")

    d = j["d"].to_numpy(float)
    s2 = 1.0 / (4.0 * d**2)
    logr = np.log(j["ir"].to_numpy(float) / j["il"].to_numpy(float))
    order = np.argsort(s2)
    edges = np.linspace(0, len(s2), n_bins + 1).astype(int)
    xs, ys = [], []
    for k in range(n_bins):
        idx = order[edges[k]:edges[k + 1]]
        if len(idx) < 20:
            continue
        xs.append(float(np.mean(s2[idx])))
        ys.append(float(np.median(logr[idx])))  # median: robust to outliers
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([np.ones_like(xs), xs]).T
    c0, slope = np.linalg.lstsq(A, ys, rcond=None)[0]
    B = -0.5 * slope
    print(f"  rescale fit: scale=exp({c0:.3f})={np.exp(c0):.4g}, "
          f"relative B={B:.2f} A^2  (over {len(xs)} shells)")
    return float(c0), float(B)


def _scale_col(ds, col, factor):
    """Multiply `ds[col]` by `factor` (array) in place, PRESERVING its MTZ dtype.

    Reassigning a raw float array would strip the rs MTZ dtype (e.g.
    `FriedelIntensity`), and `write_mtz(skip_problem_mtztypes=True)` would then
    silently drop the column -- which breaks downstream label matching.
    """
    dt = ds[col].dtype
    ds[col] = rs.DataSeries(
        ds[col].to_numpy(float) * factor, index=ds.index
    ).astype(dt)


def _apply_scale(ds, c0, B):
    """Multiply intensity columns by `k=exp(c0-2Bs^2)` (amplitudes by sqrt(k))."""
    d = ds.compute_dHKL()["dHKL"].to_numpy(float)
    s2 = 1.0 / (4.0 * d**2)
    k = np.exp(c0 - 2.0 * B * s2)
    for col in [_first(ds, c) for c in (_IPLUS, _IMINUS, _SIGIPLUS, _SIGIMINUS)]:
        if col:
            _scale_col(ds, col, k)
    sk = np.sqrt(k)
    for col in [_first(ds, c) for c in (_FPLUS, _FMINUS, _SIGFPLUS, _SIGFMINUS)]:
        if col:
            _scale_col(ds, col, sk)
    return ds


def _ice_mask(ds, bands):
    """Boolean mask of reflections to KEEP (outside every ice band)."""
    d = ds.compute_dHKL()["dHKL"].to_numpy(float)
    keep = np.ones(len(d), dtype=bool)
    for lo, hi in bands:
        keep &= ~((d >= lo) & (d <= hi))
    print(f"  ice masking: removing {(~keep).sum():,} / {len(d):,} reflections "
          f"in {len(bands)} bands")
    return keep


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--learned-mtz", type=Path, required=True)
    p.add_argument("--reference-mtz", type=Path, default=None,
                   help="DIALS merged MTZ (required for --rescale)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rescale", action="store_true",
                   help="fit & apply isotropic scale*exp(-2Bs^2) to match reference")
    p.add_argument("--ice", action="store_true",
                   help="mask built-in hexagonal-ice bands")
    p.add_argument("--ice-range", type=float, nargs=2, action="append",
                   metavar=("LO", "HI"), default=[],
                   help="extra d-range (A) to mask; repeatable")
    p.add_argument("--ice-halfwidth", type=float, default=0.02,
                   help="half-width (A) around each built-in ice d-spacing")
    p.add_argument("--bins", type=int, default=20)
    args = p.parse_args()

    ds = rs.read_mtz(str(args.learned_mtz))
    n0 = len(ds)
    print(f"loaded {n0:,} reflections from {args.learned_mtz.name}")

    if args.rescale:
        if args.reference_mtz is None:
            sys.exit("--rescale requires --reference-mtz")
        ref = rs.read_mtz(str(args.reference_mtz))
        c0, B = _fit_scale_b(ds, ref, args.bins)
        ds = _apply_scale(ds, c0, B)

    if args.ice or args.ice_range:
        bands = list(args.ice_range)
        if args.ice:
            bands += [(d - args.ice_halfwidth, d + args.ice_halfwidth)
                      for d in _ICE_D]
        keep = _ice_mask(ds, bands)
        ds = ds[keep]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("  output columns:", {c: str(ds[c].dtype) for c in ds.columns})
    ds.write_mtz(str(args.out), skip_problem_mtztypes=True)
    print(f"wrote {args.out}: {len(ds):,} reflections "
          f"({n0 - len(ds):,} removed)")


if __name__ == "__main__":
    main()
