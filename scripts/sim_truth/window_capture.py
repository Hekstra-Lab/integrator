"""Fraction of each reflection's true photons inside a square window, per window size.

The true spot of reflection r on a shot is the set of pixels whose datsimx map
names r (`truth_table.load_shot`), with their noiseless Bragg photons B_p, and
I_r = sum_p B_p. For a window of n x n pixels placed as the shoeboxes place it
(centred on floor(xyzcal.px) of the reflection table) the script reports

    capture_pre_n  = sum_{p in window} B_p / I_r
    capture_post_n = sum_{p in window} (K * B^(r))_p / I_r

where K is nanoBragg's PSF kernel as `add_noise` applies it, measured by the
`kernel` subcommand from a single bright pixel. K keeps 0.854 of a pixel's
photons within its 5-px radius and spreads the rest over the whole detector
(docs/sim_truth_protocol.md section 9), so capture_post_n <= sum(K) for every n;
that remainder is lost to every window alike and is reported separately.
The window is not slid inward at the panel edge here; reflections whose window
would leave the panel are dropped and counted.

Two stages, because nanoBragg and the DIALS reflection reader live in
different environments:

    kernel   simtbx env:     the point-source response of add_noise -> kernel.npy
    measure  integrator env: one row per (shot, hkl) of the reflection table ->
             long CSV (method, dataset, shot, h, k, l, i_true, capture_pre_<n>,
             capture_post_<n>), and a summary by i_true bin on stdout

    python window_capture.py kernel  --geom noisy/geom_run1.expt --out kernel.npy
    python window_capture.py measure TRUTH_DIR --refl R.refl --expt R.expt \\
        --kernel kernel.npy --method NAME --dataset NAME --out capture.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

EDGES = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e9]


def kernel(a):
    """add_noise on a single bright pixel, every noise term but Poisson off."""
    from dxtbx.model.experiment_list import ExperimentListFactory
    from scitbx.array_family import flex
    from simtbx.nanoBragg import nanoBragg

    el = ExperimentListFactory.from_json_file(a.geom, check_format=False)
    det, beam = el[0].detector, el[0].beam
    n_fast, n_slow = det[0].get_image_size()
    img = np.zeros((n_slow, n_fast))
    s0, f0, total = n_slow // 2, n_fast // 2, 1e10
    img[s0, f0] = total
    S = nanoBragg(detector=det, beam=beam)
    S.adc_offset_adu = 0
    S.detector_psf_kernel_radius_pixels = a.radius
    S.detector_calibration_noise_pct = 0
    S.detector_psf_fwhm_mm = a.psf_mm
    S.quantum_gain = 1
    S.readout_noise_adu = 0
    S.seed = 1
    S.raw_pixels = flex.double(img.ravel())
    S.add_noise()
    out = S.raw_pixels.as_numpy_array().reshape(img.shape) / total
    S.free_all()
    r = a.radius
    k = out[s0 - r:s0 + r + 1, f0 - r:f0 + r + 1].copy()
    np.save(a.out, k)
    print(f"kernel {k.shape}, sum {k.sum():.4f} (local), image total {out.sum():.4f}, "
          f"centre {k[r, r]:.4f}; wrote {a.out}")


def measure(a):
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import pandas as pd
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory
    from scipy.signal import fftconvolve

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from truth_table import hkl_key, load_shot

    sizes = sorted(a.sizes)
    K = np.load(a.kernel) if a.kernel else None
    kr = K.shape[0] // 2 if K is not None else 0
    el = ExperimentListFactory.from_json_file(a.expt, check_format=False)
    frame = np.array([e.imageset.indices()[0] for e in el])
    refl = flex.reflection_table.from_file(a.refl)
    ids = refl["id"].as_numpy_array()
    hkl = np.asarray(refl["miller_index"], dtype=np.int64).reshape(-1, 3)
    xy = refl["xyzcal.px"].as_numpy_array()[:, :2]
    shots_file = os.path.join(a.truth_dir, f"shots_{a.run}_rank0.h5")

    rows, n_edge, n_absent = [], 0, 0
    eids = np.unique(ids)[:: a.stride]
    hmax = sizes[-1] // 2
    for eid in eids:
        shot = int(frame[eid]) + 1
        B, key, _, _ = load_shot(shots_file, f"shot_{a.run}_{shot:05d}")
        n_slow, n_fast = B.shape
        for j in np.where(ids == eid)[0]:
            own = key == hkl_key(*hkl[j])
            i_true = B[own].sum()
            if i_true <= 0:
                n_absent += 1
                continue
            x, y = int(np.floor(xy[j, 0])), int(np.floor(xy[j, 1]))
            if x - hmax < 0 or y - hmax < 0 or x + hmax >= n_fast or y + hmax >= n_slow:
                n_edge += 1
                continue
            # pixels farther than the kernel radius from every window cannot reach one
            pad = hmax + kr
            y0, y1, x0, x1 = max(y - pad, 0), min(y + pad + 1, n_slow), max(x - pad, 0), min(x + pad + 1, n_fast)
            crop = np.where(own[y0:y1, x0:x1], B[y0:y1, x0:x1], 0.0)
            post = fftconvolve(crop, K, mode="same") if K is not None else None
            cy, cx = y - y0, x - x0
            row = dict(method=a.method, dataset=a.dataset, shot=shot, h=hkl[j, 0], k=hkl[j, 1], l=hkl[j, 2],
                       i_true=i_true)
            for n in sizes:
                h = n // 2
                win = np.s_[cy - h:cy + h + 1, cx - h:cx + h + 1]
                row[f"capture_pre_{n}"] = crop[win].sum() / i_true
                if post is not None:
                    row[f"capture_post_{n}"] = post[win].sum() / i_true
            rows.append(row)
        print(f"shot {shot}: {len(rows):,} rows so far", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    print(f"{len(df):,} reflections on {len(eids)} shots; dropped {n_edge:,} whose {sizes[-1]} px window "
          f"leaves the panel and {n_absent:,} with no true photons; appended to {a.out}")
    if K is not None:
        print(f"PSF kernel keeps {K.sum():.4f} locally; capture_post is bounded by that")
    df["bin"] = pd.cut(df.i_true, EDGES, right=False)
    cols = [f"capture_{s}_{n}" for s in (("pre", "post") if K is not None else ("pre",)) for n in sizes]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("median by i_true bin:")
        print(df.groupby("bin", observed=True)[cols].median().round(4).assign(
            n=df.groupby("bin", observed=True).size()).to_string())
        print("5th percentile by i_true bin:")
        print(df.groupby("bin", observed=True)[cols].quantile(0.05).round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kernel")
    k.add_argument("--geom", required=True)
    k.add_argument("--psf-mm", type=float, default=0.0847)
    k.add_argument("--radius", type=int, default=5)
    k.add_argument("--out", required=True)
    m = sub.add_parser("measure")
    m.add_argument("truth_dir")
    m.add_argument("--refl", required=True)
    m.add_argument("--expt", required=True, help="experiment list of --refl; maps its id to the image-file frame")
    m.add_argument("--kernel", default=None, help="kernel.npy from the kernel subcommand (default: pre-PSF only)")
    m.add_argument("--sizes", type=int, nargs="+", default=[15, 21, 25, 31, 41])
    m.add_argument("--run", type=int, default=1)
    m.add_argument("--stride", type=int, default=1, help="every stride-th experiment (default all)")
    m.add_argument("--method", required=True)
    m.add_argument("--dataset", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    kernel(a) if a.cmd == "kernel" else measure(a)


if __name__ == "__main__":
    main()
