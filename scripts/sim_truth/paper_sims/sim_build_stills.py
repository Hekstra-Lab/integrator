"""Attach the simulation's EXACT crystal to each imported still, and check it.

Why this exists. laue-dials cannot process these frames as a rotation series:
dxtbx resolves the CBFs to `FormatCBFFullStill` (the headers carry no rotation),
so `geometry.scan.oscillation` is accepted and then silently ignored -- the
import reports "num stills: N, sweep: 0" and `from_templates` dies in
`scan.set_image_range` on None. Steps 5-8 of the laue pipeline run on stills
anyway, after `laue.sequence_to_stills`, so stills are the right entry point;
what is missing is a crystal per still, and the h5 beside each cbf carries the
exact `Umat` and `Bmat`.

The convention was established by measurement, not assumption. With

    q = A h,  A = U B          (U, B read as row-major 3x3 from the flat (9,))

and an incident unit direction of -z, the Laue condition

    lambda = -2 (q . s0_hat) / |q|^2

reproduces `wave_data` for all 3,984 well-sampled reflections of shot 1 to a
median |dlambda| of 0.00032 A. The three alternative pairings of U/B with their
transposes miss by 0.65 A and give no solution at all for a third of the
reflections. The residual 0.00032 A is the mosaic excitation width -- the same
width that shows up as |h - round(h)| median 0.011 -- not a convention error.

Two further facts that contradict the Oct 9 script: `delta_phi` is 1.0 degree
but stored in RADIANS (0.017453), and consecutive `Umat` differ by exactly 1.0
degree about x = (1,0,0), where that script set
`geometry.goniometer.axes=0,1,0`.

THE CHECK. Getting the crystal in without error is not evidence it is right, so
this predicts every reflection's detector position through dxtbx's own panel
model and compares it to the ground-truth footprint centroid computed from
h_data/k_data/l_data. That closes the loop the lambda test leaves open: the
lambda test validates the crystal, this validates crystal AND detector together,
in pixels, against an answer that was never fitted.
"""

import argparse
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

RESID_CUT = 0.05
SG = "P212121"   # DHFR, space group 19


def read_UB(h5_path):
    with h5py.File(h5_path, "r") as f:
        U = np.asarray(f["Umat"][()]).reshape(3, 3)
        B = np.asarray(f["Bmat"][()]).reshape(3, 3)
    return U, B


def crystal_from_UB(U, B):
    """dxtbx Crystal whose A matrix is U @ B.

    dxtbx builds A from the real-space vectors, and for q = A h the columns of A
    are a*, b*, c*, so the ROWS of inv(A) are a, b, c.
    """
    from dxtbx.model import Crystal

    A = U @ B
    Ai = np.linalg.inv(A)
    return Crystal(tuple(Ai[0]), tuple(Ai[1]), tuple(Ai[2]),
                   space_group_symbol=SG), A


def truth_centroids(h5_path, min_pix=200):
    """hkl, mean lambda and footprint centroid in DIALS (fast, slow) pixels."""
    with h5py.File(h5_path, "r") as f:
        h = f["h_data"][()]; k = f["k_data"][()]
        l = f["l_data"][()]; w = f["wave_data"][()]
    hi, ki, li = np.round(h), np.round(k), np.round(l)
    res = np.maximum.reduce([abs(h - hi), abs(k - ki), abs(l - li)])
    ok = (res < RESID_CUT) & np.isfinite(h) & ~((hi == 0) & (ki == 0) & (li == 0))
    rr, cc = np.nonzero(ok)
    trip = np.stack([hi[ok], ki[ok], li[ok]], 1).astype(np.int64)
    uq, inv = np.unique(trip, axis=0, return_inverse=True)
    n = np.bincount(inv)
    lam = np.bincount(inv, weights=w[ok]) / np.maximum(n, 1)
    # DIALS pixel order is (fast, slow) = (column, row)
    cf = np.bincount(inv, weights=cc.astype(float)) / np.maximum(n, 1)
    cs = np.bincount(inv, weights=rr.astype(float)) / np.maximum(n, 1)
    sel = n >= min_pix
    return uq[sel].astype(float), lam[sel], cf[sel], cs[sel], n[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--imported", required=True, help="stills .expt from dials.import")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-shots", type=int, nargs="*", default=[1, 90, 180])
    args = ap.parse_args()

    from dxtbx.model.experiment_list import ExperimentListFactory

    el = ExperimentListFactory.from_json_file(args.imported, check_format=False)
    print(f"imported {len(el)} experiments")

    beam = el[0].beam
    s0 = np.asarray(beam.get_s0(), float)
    us0 = s0 / np.linalg.norm(s0)
    # dxtbx's own unit_s0 is (0,0,-1), which is the incident direction the
    # lambda test independently picked out, so A = U B needs no sign fix here.
    print(f"beam.get_unit_s0() {np.asarray(beam.get_unit_s0(), float)}")
    print(f"beam.get_s0() {s0}  -> unit {us0}")

    for i, expt in enumerate(el):
        n = i + 1
        U, B = read_UB(os.path.join(args.data_dir, f"shot_1_{n:05d}.h5"))
        cr, A = crystal_from_UB(U, B)
        if i == 0:
            got = np.asarray(cr.get_A(), float).reshape(3, 3)
            err = np.abs(got - A).max()
            print(f"crystal round-trip: max|get_A() - U@B| = {err:.3e}")
            print(f"  cell {tuple(round(x, 3) for x in cr.get_unit_cell().parameters())}")
        expt.crystal = cr
    el.as_file(args.out)
    print(f"wrote {args.out}")

    print("\nposition check against ground-truth footprint centroids")
    print(f"{'shot':>5} {'n':>6} {'dlam med':>9} {'dfast med':>10} {'dslow med':>10} "
          f"{'dr med':>8} {'dr p90':>8} {'dr>3px':>8}")
    for n in args.check_shots:
        if n > len(el):
            continue
        expt = el[n - 1]
        panel = expt.detector[0]
        H, lam_t, cf, cs, npix = truth_centroids(
            os.path.join(args.data_dir, f"shot_1_{n:05d}.h5"))
        A = np.asarray(expt.crystal.get_A(), float).reshape(3, 3)
        q = H @ A.T
        n2 = (q * q).sum(1)
        lam_p = -2.0 * (q @ us0) / np.maximum(n2, 1e-30)
        good = np.isfinite(lam_p) & (lam_p > 0.5) & (lam_p < 2.0)
        pf, ps = [], []
        for j in np.nonzero(good)[0]:
            s1 = us0 / lam_p[j] + q[j]
            try:
                x, y = panel.get_ray_intersection_px(tuple(s1))
            except RuntimeError:
                x = y = np.nan
            pf.append(x); ps.append(y)
        pf = np.asarray(pf); ps = np.asarray(ps)
        df = pf - cf[good]; ds = ps - cs[good]
        dr = np.hypot(df, ds)
        fin = np.isfinite(dr)
        print(f"{n:5d} {fin.sum():6d} {np.median(np.abs(lam_p[good] - lam_t[good])):9.5f} "
              f"{np.median(np.abs(df[fin])):10.2f} {np.median(np.abs(ds[fin])):10.2f} "
              f"{np.median(dr[fin]):8.2f} {np.quantile(dr[fin], 0.9):8.2f} "
              f"{100 * (dr[fin] > 3).mean():7.1f}%")


if __name__ == "__main__":
    main()
