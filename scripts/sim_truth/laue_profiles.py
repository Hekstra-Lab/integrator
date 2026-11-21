"""laue.integrate's fitted per-reflection profiles, laid on our shoebox windows.

laue-dials' `Integrator` (laue_dials/algorithms/integration.py) fits, per
reflection, a 2-D Gaussian whose softmax over the valid pixels of a disc window
of radius r is the profile p used as profile-fitting weights,

    w = (p / v) / sum(p^2 / v),   I = sum((c - b) w),   v = max(0, I) p + b,

and keeps p, the Gaussian's centre and covariance only in memory: the command
writes I, SigI and the background. This script runs laue-dials' own per-image
function `integrate_image` unchanged (same mask dilation, radius, window
validity and refit) with the `Integrator` it builds replaced by a subclass whose
only addition is to keep a reference to itself after `fit`. Nothing in
laue-dials is edited, so the profiles are those of the run that wrote
integrated.refl, given the same inputs, parameters and laue-dials commit.

Per reflection the npz holds

    image_num, h, k, l, x, y          the input table's row (x, y = xyzcal.px)
    I, sigI, bg                       laue's intensity, sigma, per-pixel background
    radius, n_valid                   window radius and usable pixels in the disc
    loc (y, x), scale (2x2)           the fitted Gaussian, image coordinates
    strong                            used as a profile source for neighbours
    profile_box (n, box*box)          p summed onto our box, placed as
                                      make_sim_shoeboxes places it (centred on
                                      floor(xyzcal.px), slid inward at the edge)
    mass_in_box                       sum of profile_box; p sums to 1 over the
                                      disc, so 1 - mass_in_box is the part of
                                      laue's profile outside our box

Rows are the reflections laue.integrate fitted, before its final selection of
I != 0 and SigI^2 > 0, matched to the input table by their predicted centroid.
With --check-refl the fitted I are compared with an integrated.refl written by
the same chain, which says whether this reproduces that run.

    PYTHONPATH=<laue-dials>/src python laue_profiles.py --expt poly_refined.expt \\
        --refl predicted.refl --out laue_profiles.npz [--check-refl integrated.refl]
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

_KEPT: list = []


def _install():
    import laue_dials.command_line.integrate as li

    if getattr(li.Integrator, "_keeps_self", False):
        return li

    class KeepingIntegrator(li.Integrator):
        _keeps_self = True

        def fit(self, *args, **kwargs):
            out = super().fit(*args, **kwargs)
            _KEPT.append(self)
            return out

    li.Integrator = KeepingIntegrator
    return li


def _one_image(job):
    img_set, table, params, box, image_num = job
    li = _install()
    _KEPT.clear()
    li.integrate_image(img_set, table, *params)
    if not _KEPT:
        return None
    it = _KEPT[-1]
    xy_in = table["xyzcal.px"].as_numpy_array()[:, :2].astype("float32")
    hkl_in = np.asarray(table["miller_index"], dtype=np.int64).reshape(-1, 3)
    index = {(float(x), float(y)): i for i, (x, y) in enumerate(xy_in)}
    cen = it.centroids[:, ::-1].astype("float32")  # (x, y)
    rows = np.array([index[(float(x), float(y))] for x, y in cen])

    n_slow, n_fast = it.pixels.shape[1:]
    half = box // 2
    xi, yi = np.floor(cen[:, 0]).astype(int), np.floor(cen[:, 1]).astype(int)
    x0 = np.clip(xi - half, 0, n_fast - box)
    y0 = np.clip(yi - half, 0, n_slow - box)
    p = it.profile_values  # (n, m), zero on invalid window pixels
    wy, wx = it.window_idx[0], it.window_idx[1]  # (n, m)
    by, bx = wy - y0[:, None], wx - x0[:, None]
    inside = (by >= 0) & (by < box) & (bx >= 0) & (bx < box) & it.window_valid
    prof = np.zeros((len(cen), box * box))
    r_idx = np.broadcast_to(np.arange(len(cen))[:, None], p.shape)
    np.add.at(prof, (r_idx[inside], (by * box + bx)[inside]), p[inside])
    return dict(
        image_num=np.full(len(cen), image_num), h=hkl_in[rows, 0], k=hkl_in[rows, 1], l=hkl_in[rows, 2],
        x=cen[:, 0], y=cen[:, 1], I=it.intensity, sigI=it.uncertainty, bg=it.background.squeeze(-1),
        radius=np.full(len(cen), it.radius), n_valid=it.window_valid.sum(-1), loc=it.profile_loc,
        scale=it.profile_scale, strong=it.strong.astype(bool), profile_box=prof.astype(np.float32),
        mass_in_box=prof.sum(-1), box_x0=x0, box_y0=y0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expt", required=True, help="the experiment list laue.integrate read (poly_refined.expt)")
    ap.add_argument("--refl", required=True, help="the prediction table laue.integrate read")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=int, default=25)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--check-refl", default=None, help="an integrated.refl from the same chain, to compare I")
    a = ap.parse_args()
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    import laue_dials
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    li = _install()
    ph = li.working_phil.extract()
    params = (ph.isigi_cutoff, ph.integration_radius, ph.knn, ph.maxiter)
    print(f"laue_dials from {laue_dials.__file__}; isigi_cutoff {params[0]}, integration_radius {params[1]}, "
          f"knn {params[2]}, maxiter {params[3]} (the command's defaults, as the chain ran it)")

    expts = ExperimentListFactory.from_json_file(a.expt, check_format=True)
    preds = flex.reflection_table.from_file(a.refl)
    imagesets = expts.imagesets()
    ids = list(np.unique(preds["id"]).astype(np.int32))
    assert len(imagesets) == len(ids), "imagesets and prediction ids differ, as laue.integrate would refuse"
    frames = [int(e.imageset.indices()[0]) for e in expts]
    jobs = [(imagesets[k], preds.select(preds["id"] == i), params, a.box, frames[i] + 1)
            for k, i in enumerate(ids)]
    if a.max_images:
        jobs = jobs[: a.max_images]
    t0 = time.time()
    with Pool(a.nproc) as pool:
        parts = [r for r in pool.imap(_one_image, jobs, chunksize=1) if r is not None]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(a.out, **out)
    print(f"{len(out['I']):,} reflections on {len(parts)} images in {time.time() - t0:.0f} s -> {a.out}")
    print(f"radius per image: {sorted(set(out['radius'].tolist()))}; mass of laue's profile inside our "
          f"{a.box}x{a.box} box: median {np.median(out['mass_in_box']):.4f}, p5 {np.percentile(out['mass_in_box'], 5):.4f}")

    if a.check_refl:
        r = flex.reflection_table.from_file(a.check_refl)
        rid = r["id"].as_numpy_array()
        rf = np.array(frames)[rid] + 1
        hkl = np.asarray(r["miller_index"], dtype=np.int64).reshape(-1, 3)
        key = lambda f, h, k, l: ((f.astype(np.int64) * 1024 + h + 256) * 1024 + k + 256) * 1024 + l + 256
        ref = dict(zip(key(rf, hkl[:, 0], hkl[:, 1], hkl[:, 2]).tolist(), r["intensity.sum.value"].as_numpy_array()))
        mine = key(out["image_num"], out["h"], out["k"], out["l"])
        both = np.array([m in ref for m in mine.tolist()])
        d = np.abs(out["I"][both] - np.array([ref[m] for m in mine[both].tolist()]))
        print(f"--check-refl: {both.sum():,} of {len(mine):,} rows in {a.check_refl}; |I - I_refl| max {d.max():.3g}, "
              f"median {np.median(d):.3g}")


if __name__ == "__main__":
    main()
