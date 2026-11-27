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

    image_num, h, k, l, x, y          the input table's row (x, y = xyzcal.px);
                                      image_num = the imageset frame + 1, which is 1
                                      on every row when each image is its own file
    expt_id                           the experiment's index in --reference-expt,
                                      found by image path (default: in --expt);
    file_index                        the trailing integer of the image file's name,
                                      minus 1 (HEWL_NaI_3_2_0248.mccd -> 247); -1 when
                                      the name has none. This is the `image_num` of the
                                      hewl_1118 shoeboxes, which differs from the
                                      experiment id wherever files were not indexed
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

On a real dataset the full npz is gigabytes (hewl_1118: about 2.6 million rows of
625 pixels), too large to take to the laptop where figures are drawn. `--reduce-from`
fits nothing; it reads a full npz and a `dump_profile_maps.py` npz and writes what
`fig_profile_gate_examples.py --laue-profiles` draws:

    image_num, h, k, l, x, y, profile_box    the rows matching any arm's named
                                             reflections (`<arm>_named_*` or `<arm>_ex_*`)
    cells (grid, grid, box*box)              laue's mean profile_box per detector
    cells_n (grid, grid)                     cell, on the first arm's `<arm>_extent`
                                             and grid, the edges the figure uses

    python laue_profiles.py --reduce-from laue_profiles.npz --maps profile_maps.npz \\
        --out laue_profiles_reduced.npz [--join expt_id]

`--join` names the laue column matched to the maps' `<arm>_named_image`: `image_num`
(default) for a single multi-frame file, `file_index` for hewl_1118 (one file per
image, numbered from 1), `expt_id` where the shoeboxes number images by experiment.
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
    img_set, table, params, box, image_num, expt_id, file_index = job
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
        image_num=np.full(len(cen), image_num), expt_id=np.full(len(cen), expt_id),
        file_index=np.full(len(cen), file_index), h=hkl_in[rows, 0], k=hkl_in[rows, 1], l=hkl_in[rows, 2],
        x=cen[:, 0], y=cen[:, 1], I=it.intensity, sigI=it.uncertainty, bg=it.background.squeeze(-1),
        radius=np.full(len(cen), it.radius), n_valid=it.window_valid.sum(-1), loc=it.profile_loc,
        scale=it.profile_scale, strong=it.strong.astype(bool), profile_box=prof.astype(np.float32),
        mass_in_box=prof.sum(-1), box_x0=x0, box_y0=y0,
    )


def reduce_for_figure(full_path: str, maps_path: str, out_path: str, join: str = "image_num") -> None:
    """Named rows and per-cell means of a full laue_profiles npz; see the module docstring."""
    full = np.load(full_path)
    maps = np.load(maps_path)
    # REFL_IDS runs store the named reflections as `<arm>_named_*`, which the figure
    # presents as `<arm>_ex_*`; other runs store `<arm>_ex_*` directly
    sets = [(k[: -len(f"_{tag}_image")], tag) for tag in ("named", "ex")
            for k in maps.files if k.endswith(f"_{tag}_image")]
    if not sets:
        raise SystemExit(f"{maps_path} has no <arm>_named_image or <arm>_ex_image keys; rerun dump_profile_maps.py")
    arms = sorted({a for a, _ in sets})
    key = lambda f, h, k, l: ((np.asarray(f, np.int64) * 1024 + h + 256) * 1024 + k + 256) * 1024 + l + 256
    if join not in full.files:
        raise SystemExit(f"{full_path} has no {join!r} column; rerun laue_profiles.py")
    have = key(full[join], full["h"], full["k"], full["l"])
    want = np.unique(np.concatenate([key(*(maps[f"{a}_{tag}_{c}"] for c in ("image", "h", "k", "l")))
                                     for a, tag in sets]))
    rows = np.flatnonzero(np.isin(have, want))
    out = {k: full[k][rows] for k in ("image_num", "expt_id", "file_index", "h", "k", "l", "x", "y", "profile_box")
           if k in full.files}
    if join != "image_num":
        out["image_num"] = full[join][rows]  # the figure joins on image_num

    cell_arm = next((a for a in arms if f"{a}_extent" in maps.files and f"{a}_cells" in maps.files), None)
    if cell_arm is not None:
        x0, x1, y0, y1 = (float(v) for v in maps[f"{cell_arm}_extent"])
        grid = maps[f"{cell_arm}_cells"].shape[0]
        x, y = full["x"], full["y"]
        ix = np.clip(((x - x0) / max(x1 - x0, 1e-9) * grid).astype(int), 0, grid - 1)
        iy = np.clip(((y - y0) / max(y1 - y0, 1e-9) * grid).astype(int), 0, grid - 1)
        prof = full["profile_box"]
        cells = np.full((grid, grid, prof.shape[1]), np.nan, dtype=np.float64)
        cells_n = np.zeros((grid, grid), dtype=np.int64)
        for r in range(grid):
            for c in range(grid):
                sel = (iy == r) & (ix == c)
                cells_n[r, c] = int(sel.sum())
                if cells_n[r, c]:
                    cells[r, c] = prof[sel].astype(np.float64).mean(0)
        out.update(cells=cells, cells_n=cells_n, cells_extent=np.array([x0, x1, y0, y1]))
    np.savez_compressed(out_path, **out)
    print(f"{len(rows)} of {len(want)} named reflections found among {len(have):,} laue rows; "
          f"cells {'from ' + cell_arm if cell_arm else 'not written (no <arm>_cells)'} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expt", help="the experiment list laue.integrate read (poly_refined.expt)")
    ap.add_argument("--refl", help="the prediction table laue.integrate read")
    ap.add_argument("--reduce-from", default=None,
                    help="a full npz of this script: write only the figure's rows and cell means to --out")
    ap.add_argument("--maps", default=None, help="with --reduce-from: the dump_profile_maps.py npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=int, default=25)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--check-refl", default=None, help="an integrated.refl from the same chain, to compare I")
    ap.add_argument("--reference-expt", default=None,
                    help="the experiment list whose indices expt_id reports, matched by image path "
                         "(for a subset of it); default --expt")
    ap.add_argument("--join", choices=("image_num", "expt_id", "file_index"), default="image_num",
                    help="with --reduce-from: the laue column matched to the maps' image number")
    a = ap.parse_args()
    if a.reduce_from:
        if not a.maps:
            ap.error("--reduce-from needs --maps")
        reduce_for_figure(a.reduce_from, a.maps, a.out, a.join)
        return
    if not (a.expt and a.refl):
        ap.error("--expt and --refl are required unless --reduce-from is given")
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
    if a.reference_expt:
        ref = ExperimentListFactory.from_json_file(a.reference_expt, check_format=False)
        where = {os.path.basename(e.imageset.paths()[0]): j for j, e in enumerate(ref)}
        expt_ids = [where[os.path.basename(e.imageset.paths()[0])] for e in expts]
    else:
        expt_ids = list(range(len(expts)))
    import re

    def file_index(e):
        m = re.search(r"(\d+)\.[^.\d]+$", os.path.basename(e.imageset.paths()[0]))
        return int(m.group(1)) - 1 if m else -1

    file_indices = [file_index(e) for e in expts]
    jobs = [(imagesets[k], preds.select(preds["id"] == i), params, a.box, frames[i] + 1, expt_ids[i],
             file_indices[i]) for k, i in enumerate(ids)]
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
        # keyed on the experiment, not the frame: one file per image puts every row on frame 0
        rf = np.array(expt_ids)[rid]
        hkl = np.asarray(r["miller_index"], dtype=np.int64).reshape(-1, 3)
        key = lambda f, h, k, l: ((f.astype(np.int64) * 1024 + h + 256) * 1024 + k + 256) * 1024 + l + 256
        ref = dict(zip(key(rf, hkl[:, 0], hkl[:, 1], hkl[:, 2]).tolist(), r["intensity.sum.value"].as_numpy_array()))
        mine = key(out["expt_id"], out["h"], out["k"], out["l"])
        both = np.array([m in ref for m in mine.tolist()])
        d = np.abs(out["I"][both] - np.array([ref[m] for m in mine[both].tolist()]))
        print(f"--check-refl: {both.sum():,} of {len(mine):,} rows in {a.check_refl}; |I - I_refl| max {d.max():.3g}, "
              f"median {np.median(d):.3g}")


if __name__ == "__main__":
    main()
