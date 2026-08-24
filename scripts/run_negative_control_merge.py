"""Run the centric-anchored, unsupervised scale+merge on integrator predictions.

Extension driver (imports only the new `integrator.merge_ext` module + rs/gemmi
for symmetry; touches nothing in the training pipeline). It:

  1. loads per-observation intensities from a predictions parquet,
  2. uses reciprocalspaceship to label centrics and map each reflection to its
     anomalous / non-anomalous ASU id and its F(+)/F(-) flag,
  3. builds the scale grouping from a geometry covariate (default: image/frame;
     pass --scale-bins to also split each image into radial detector bins),
  4. solves scale + merge by robust ALS with the centric anchor, and
  5. writes an unmerged-by-sign MTZ [I(+), SIGI(+), I(-), SIGI(-)] you can feed
     straight to French-Wilson / phenix.

Nothing about this is rotation-specific except the choice of covariate in (3):
for Laue add wavelength, for stills add a per-crystal index, etc.

Validate the install/algorithm on the cluster with no data:

    uv run python scripts/run_negative_control_merge.py --self-test

Real run:

    uv run python scripts/run_negative_control_merge.py \
        --preds preds.parquet --spacegroup "P 43 21 2" \
        --cell 79 79 38 90 90 90 --out merged_nc.mtz \
        --i-col qi_mean --sigi-col qi_var --sigi-is-variance \
        --frame-col "xyzcal.px.2"

Expected parquet columns: H, K, L, the intensity/sigma columns, and the frame
column. If your predictions live in a .refl or are split across files, join them
to H/K/L/frame first (the integrator's metadata.pt carries H,K,L and
xyzcal.px.*).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from integrator.merge_ext.negative_control_merge import merge_anomalous


def _self_test() -> int:
    """Run the synthetic confound recovery and print the headline numbers."""
    sys.path.insert(0, "tests")
    from test_negative_control_merge import _simulate_confounded  # type: ignore

    d = _simulate_confounded()
    out = merge_anomalous(
        d["I"], d["sig"], d["img"], d["anom"], d["nonanom"],
        d["centric"], d["plus"],
    )
    s = out["result"]["scale"]
    ratio_est = s[:20].mean() / s[20:].mean()
    ratio_true = d["s_img"][:20].mean() / d["s_img"][20:].mean()
    ac = ~out["is_centric"]
    nid = out["nonanom_id"]
    e = out["dano"][ac] - d["dano_true"][nid][ac]
    print("[self-test] centric-anchored scale+merge on synthetic confound:")
    print(f"  recovered A/B scale ratio = {ratio_est:.3f}  (true {ratio_true:.3f})")
    print(f"  acentric DANO error: median={np.median(e):+.3f}  p90|e|="
          f"{np.percentile(np.abs(e), 90):.3f}")
    print(f"  reduced chi^2 = {out['result']['chi2']:.2f}, "
          f"iters = {out['result']['n_iter']}")
    ok = abs(ratio_est / ratio_true - 1) < 0.03 and abs(np.median(e)) < 0.5
    print("  RESULT:", "OK" if ok else "UNEXPECTED")
    return 0 if ok else 1


def _scale_groups(frame, x, y, n_radial_bins):
    """Per-frame scale group, optionally x per-frame x radial-shell bins."""
    frame = np.asarray(frame)
    fid = np.unique(frame, return_inverse=True)[1]
    if n_radial_bins <= 1 or x is None or y is None:
        return fid
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2)
    rb = np.clip((r / (r.max() + 1e-9) * n_radial_bins).astype(int), 0,
                 n_radial_bins - 1)
    return fid * n_radial_bins + rb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic recovery check and exit")
    ap.add_argument("--preds", help="predictions parquet")
    ap.add_argument("--spacegroup", help="e.g. 'P 43 21 2'")
    ap.add_argument("--cell", nargs=6, type=float,
                    help="a b c alpha beta gamma")
    ap.add_argument("--out", help="output MTZ path")
    ap.add_argument("--i-col", default="qi_mean")
    ap.add_argument("--sigi-col", default="qi_var")
    ap.add_argument("--sigi-is-variance", action="store_true",
                    help="treat --sigi-col as a variance (take sqrt)")
    ap.add_argument("--frame-col", default="xyzcal.px.2")
    ap.add_argument("--x-col", default="xyzcal.px.0")
    ap.add_argument("--y-col", default="xyzcal.px.1")
    ap.add_argument("--scale-bins", type=int, default=1,
                    help="radial detector bins per frame (1 = per-frame scalar)")
    ap.add_argument("--biweight-c", type=float, default=6.0)
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if not (args.preds and args.spacegroup and args.cell and args.out):
        ap.error("real run needs --preds --spacegroup --cell --out "
                 "(or use --self-test)")

    import pandas as pd
    import reciprocalspaceship as rs

    df = pd.read_parquet(args.preds)
    H = df[["H", "K", "L"]].to_numpy().astype(np.int32)
    I = df[args.i_col].to_numpy().astype(np.float64)
    sig = df[args.sigi_col].to_numpy().astype(np.float64)
    if args.sigi_is_variance:
        sig = np.sqrt(np.maximum(sig, 0.0))
    frame = df[args.frame_col].to_numpy()
    x = df[args.x_col].to_numpy() if args.x_col in df else None
    y = df[args.y_col].to_numpy() if args.y_col in df else None

    sg = rs.utils.canonicalize_spacegroup(args.spacegroup) \
        if hasattr(rs.utils, "canonicalize_spacegroup") else args.spacegroup

    # Symmetry: ASU mapping (ISYM odd = F(+), even = F(-)) + centric labels.
    hasu, isym = rs.utils.hkl_to_asu(H, sg)
    plus = (isym % 2 == 1)
    # non-anomalous id = the reciprocal-ASU hkl (Friedel mates merged)
    _, nonanom_id = np.unique(
        np.ascontiguousarray(hasu).view(
            np.dtype((np.void, hasu.dtype.itemsize * 3))
        ).ravel(), return_inverse=True,
    )
    # anomalous id = (asu hkl, sign)
    anom_key = nonanom_id * 2 + plus.astype(np.int64)
    _, anom_id = np.unique(anom_key, return_inverse=True)

    ds = rs.DataSet(
        {"H": H[:, 0], "K": H[:, 1], "L": H[:, 2]},
        cell=args.cell, spacegroup=sg,
    ).set_index(["H", "K", "L"])
    centric = ds.label_centrics()["CENTRIC"].to_numpy()

    scale_group = _scale_groups(frame, x, y, args.scale_bins)

    out = merge_anomalous(
        I, sig, scale_group, anom_id, nonanom_id, centric, plus,
        biweight_c=args.biweight_c,
    )
    res = out["result"]
    n_ref = len(out["nonanom_id"])
    n_cent = int(out["is_centric"].sum())
    print(f"merged {n_ref} reflections ({n_cent} centric anchors); "
          f"reduced chi^2 = {res['chi2']:.2f}; scale spread "
          f"[{res['scale'].min():.2f}, {res['scale'].max():.2f}]; "
          f"iters = {res['n_iter']}")

    # Reconstruct asu hkl per unique reflection for the output index.
    first = np.zeros(n_ref, dtype=np.int64)
    seen = np.full(n_ref, False)
    for i, g in enumerate(nonanom_id):
        if not seen[g]:
            seen[g] = True
            first[g] = i
    out_h = hasu[first]
    merged = rs.DataSet(
        {
            "H": out_h[:, 0], "K": out_h[:, 1], "L": out_h[:, 2],
            "I(+)": out["I_plus"], "SIGI(+)": out["sig_plus"],
            "I(-)": out["I_minus"], "SIGI(-)": out["sig_minus"],
        },
        cell=args.cell, spacegroup=sg,
    ).set_index(["H", "K", "L"])
    for col, dt in (("I(+)", "K"), ("SIGI(+)", "M"),
                    ("I(-)", "K"), ("SIGI(-)", "M")):
        merged[col] = merged[col].astype(dt)
    merged = merged[np.isfinite(merged[["I(+)", "I(-)"]].to_numpy()).all(1)]
    merged.write_mtz(args.out)
    print(f"wrote {args.out}  ({len(merged)} rows). "
          "Next: French-Wilson (rs.algorithms.scale_merged_intensities) "
          "then your anomalous-map / peak-height pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
