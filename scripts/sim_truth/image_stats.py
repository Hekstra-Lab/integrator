"""Per-image statistics of a simulated Laue shot, by one estimator for the laue-dials paper sims and for datsimx.

Every statistic is computed from the NOISY image and the per-pixel Miller-index
maps only, so the paper shots (which have no noiseless render) and the datsimx
shots are measured the same way:

    photons     = (ADU - adc) / gain
    footprint F = pixels with a finite, positive wave_data
    background  = mean photons of the pixels outside F in 32 x 32 px blocks
                  (blocks entirely inside F take the mean of their neighbours)
    net         = photons - background, on F
    bragg_net   = sum of net over F; per reflection, sum of net over its pixels
                  (label = rounded h, k, l)
    noise a, b  = least squares Var(net) = a * bg + b over pixels outside F,
                  in background deciles (photon units; a = 1 and b = (readout/gain)^2
                  for pure Poisson plus readout)
    radius      = for the 100 reflections of largest net, sqrt of the
                  net-weighted second moment (weights max(net, 0)) over the
                  reflection's pixels within 10 px of its brightest pixel;
                  reported in px and in mrad (px * pixel / distance)
    peak_over_bg = brightest net pixel / background there, same reflections
    bg_by_2theta = median background per pixel and per steradian
                   (photons / (pixel^2 cos^3(2theta) / distance^2)) in 2.5 deg bins

usage:
  python image_stats.py paper   SHOT.cbf SHOT.h5         --label NAME [--gain 0.7 --adc 10]
  python image_stats.py datsimx NOISY_DIR TRUTH_DIR SHOT --label NAME [--gain 0.7 --adc 0]
Appends one row to --out (default image_stats.csv) and the 2theta profile to
--out-profile (default image_stats_bg.csv); both long-format with a `dataset` key.
Runs in the simtbx environment (numpy < 2, dxtbx, h5py).
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings

import h5py
import numpy as np

BLK = 32
NTOP = 100
HALF = 10


def load_paper(cbf, h5):
    import dxtbx

    img = dxtbx.load(cbf)
    adu = img.get_raw_data().as_numpy_array().astype(np.float64)
    with h5py.File(h5, "r") as f:
        maps = {c: f[f"{c}_data"][()] for c in ("h", "k", "l", "wave")}
    return adu, maps, img.get_detector()[0], img.get_beam()


def load_datsimx(noisy, truth, shot):
    from dxtbx.model.experiment_list import ExperimentListFactory

    run = shot.split("_")[1]
    el = ExperimentListFactory.from_json_file(os.path.join(noisy, f"geom_run{run}.expt"), check_format=False)
    with h5py.File(os.path.join(noisy, f"shots_{run}_rank0.h5"), "r") as f:
        adu = f[f"sim_image/{shot}"][()].astype(np.float64)
    with h5py.File(os.path.join(truth, f"shots_{run}_rank0.h5"), "r") as f:
        maps = {c: f[f"{c}_data/{shot}"][()] for c in ("h", "k", "l", "wave")}
    return adu, maps, el[0].detector[0], el[0].beam


def two_theta(panel, beam, shape):
    ny, nx = shape
    p = np.array(panel.get_pixel_size())
    o, fa, sl = (np.array(v) for v in (panel.get_origin(), panel.get_fast_axis(), panel.get_slow_axis()))
    xs, ys = (np.arange(nx) + 0.5) * p[0], (np.arange(ny) + 0.5) * p[1]
    lab = o[None, None] + xs[None, :, None] * fa[None, None] + ys[:, None, None] * sl[None, None]
    s0 = np.array(beam.get_unit_s0())
    cos2t = (lab @ s0) / np.linalg.norm(lab, axis=2)
    return np.degrees(np.arccos(np.clip(cos2t, -1, 1))), cos2t


def block_background(ph, foot):
    ny, nx = ph.shape
    out = np.where(foot, np.nan, ph)
    by, bx = ny // BLK, nx // BLK
    blocks = out[: by * BLK, : bx * BLK].reshape(by, BLK, bx, BLK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bg = np.nanmean(blocks, axis=(1, 3))
        while np.isnan(bg).any():
            pad = np.pad(bg, 1, constant_values=np.nan)
            nb = np.nanmean(np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]]), axis=0)
            bg = np.where(np.isnan(bg), nb, bg)
    full = np.kron(bg, np.ones((BLK, BLK)))
    return np.pad(full, ((0, ny - full.shape[0]), (0, nx - full.shape[1])), mode="edge")


def stats(adu, maps, panel, beam, gain, adc):
    ph = (adu - adc) / gain
    lam = maps["wave"]
    foot = np.isfinite(lam) & (lam > 0)
    bg = block_background(ph, foot)
    net = ph - bg

    out_px = ~foot
    q = np.quantile(bg[out_px], np.linspace(0, 1, 11))
    mb, vb = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        s = out_px & (bg >= lo) & (bg < hi)
        if s.sum() > 1000:
            mb.append(bg[s].mean())
            vb.append(net[s].var())
    a, b = np.polyfit(mb, vb, 1)

    ys, xs = np.nonzero(foot)
    hkl = np.stack([np.rint(maps[c][foot]).astype(np.int64) for c in "hkl"])
    key = ((hkl[0] + 512) * 1024 + hkl[1] + 512) * 1024 + hkl[2] + 512
    _, inv = np.unique(key, return_inverse=True)
    nf = net[foot]
    I = np.bincount(inv, weights=nf)
    order = np.argsort(I)[::-1]

    radii, pob = [], []
    for j in order[:NTOP]:
        m = inv == j
        yy, xx, w = ys[m], xs[m], nf[m]
        k = np.argmax(w)
        near = (np.abs(yy - yy[k]) <= HALF) & (np.abs(xx - xx[k]) <= HALF)
        yy, xx, ww = yy[near], xx[near], np.clip(w[near], 0, None)
        if ww.sum() <= 0:
            continue
        cy, cx = (ww * yy).sum() / ww.sum(), (ww * xx).sum() / ww.sum()
        radii.append(np.sqrt((ww * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / ww.sum()))
        pob.append(w[k] / max(bg[ys[m][k], xs[m][k]], 1e-9))

    pix = panel.get_pixel_size()[0]
    dist = panel.get_distance()
    tth, cos2t = two_theta(panel, beam, ph.shape)
    omega = pix**2 * cos2t**3 / dist**2
    edges = np.arange(0, 62.5, 2.5)
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = out_px & (tth >= lo) & (tth < hi)
        if s.sum() > 1000:
            prof.append(dict(two_theta_lo=lo, two_theta_hi=hi, n_px=int(s.sum()),
                             bg_per_px=float(np.median(bg[s])), bg_per_sr=float(np.median(bg[s] / omega[s]))))

    Is = I[order]
    row = dict(
        distance_mm=dist, pixel_mm=pix, ny=ph.shape[0], nx=ph.shape[1],
        footprint_frac=float(foot.mean()), n_refl=len(I),
        bg_median_per_px=float(np.median(bg[out_px])),
        bg_median_per_sr=float(np.median(bg[out_px] / omega[out_px])),
        bragg_net=float(nf.sum()), top100_net=float(Is[:100].sum()),
        net_rank1=float(Is[0]), net_rank10=float(Is[9]), net_rank100=float(Is[99]), net_rank300=float(Is[299]),
        noise_a=float(a), noise_b=float(b),
        radius_px_median=float(np.median(radii)), radius_mrad_median=float(np.median(radii) * pix / dist * 1e3),
        peak_over_bg_median=float(np.median(pob)),
        frac_negative_px=float((adu - adc < 0).mean()),
    )
    return row, prof


def append(path, rows):
    new = not os.path.exists(path)
    cols = list(rows[0].keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["paper", "datsimx"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--label", required=True)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--adc", type=float, default=None, help="default 10 for paper, 0 for datsimx")
    ap.add_argument("--out", default="image_stats.csv")
    ap.add_argument("--out-profile", default="image_stats_bg.csv")
    a = ap.parse_args()

    if a.kind == "paper":
        adu, maps, panel, beam = load_paper(*a.paths)
        adc = 10.0 if a.adc is None else a.adc
    else:
        adu, maps, panel, beam = load_datsimx(*a.paths)
        adc = 0.0 if a.adc is None else a.adc
    row, prof = stats(adu, maps, panel, beam, a.gain, adc)
    src = " ".join(a.paths)
    append(a.out, [dict(dataset=a.label, source=src, **row)])
    append(a.out_profile, [dict(dataset=a.label, **p) for p in prof])
    for k, v in row.items():
        print(f"{k:22s} {v:.4g}" if isinstance(v, float) else f"{k:22s} {v}")


if __name__ == "__main__":
    main()
