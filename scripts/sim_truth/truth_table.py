"""Per-reflection truth from a datsimx `--noNoise --waveImg` run: one row per (shot, hkl).

A pixel belongs to reflection hkl when its noiseless Bragg value B_p > 0 and
round(h_data, k_data, l_data) = hkl at that pixel (the maps are NaN wherever
B_p = 0). With that labelling

    i_true   = sum_p B_p                      (photons, before noise and gain)
    n_px     = number of pixels with B_p > 0
    n_eff    = 1 / sum_p (B_p / i_true)^2     (of the truth profile)
    lam_mean = sum_p B_p lam_p / i_true       (Angstrom)
    x_cen, y_cen = B-weighted fast / slow pixel centroid
    n_comp   = connected components (8-connectivity) of the pixels with
               B_p >= REL_THR * max_p B_p of the reflection

The maps carry one hkl per pixel, so where two reflections (or harmonics) land
on the same pixel its whole B_p goes to the one the map names.
n_comp counts components of the thresholded footprint of all reflections, so
two parts of one reflection joined only through a neighbouring reflection count
as one.

usage: python truth_table.py TRUTH_DIR [--run 1] [--out TRUTH_DIR/truth_reflections.csv]
Needs numpy, h5py and scipy only, so it runs in the simtbx environment.
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np
from scipy import ndimage

REL_THR = 1e-3
OFFSET = 512  # hkl -> one int64 key; |h|, |k|, |l| < 512
COLUMNS = ["shot", "h", "k", "l", "i_true", "n_px", "n_eff", "lam_mean", "x_cen", "y_cen", "n_comp"]


def hkl_key(h, k, l):
    return ((h + OFFSET) * 1024 + (k + OFFSET)) * 1024 + (l + OFFSET)


def key_hkl(key):
    l = key % 1024 - OFFSET
    k = key // 1024 % 1024 - OFFSET
    h = key // (1024 * 1024) - OFFSET
    return h, k, l


def shot_names(truth_dir, run):
    """(shots file, shot name) for every shot of the run, in shot order."""
    out = []
    for path in sorted(glob.glob(os.path.join(truth_dir, f"shots_{run}_*.h5"))):
        with h5py.File(path, "r") as f:
            out += [(path, name) for name in f["sim_image"].keys()]
    return sorted(out, key=lambda t: t[1])


def load_shot(path, name):
    """Noiseless Bragg image, per-pixel hkl key (-1 off the footprint) and wavelength."""
    with h5py.File(path, "r") as f:
        bragg = f[f"sim_image/{name}"][()].astype(np.float64)
        hkl = [f[f"{c}_data/{name}"][()] for c in "hkl"]
        lam = f[f"wave_data/{name}"][()]
    foot = bragg > 0
    key = np.full(bragg.shape, -1, dtype=np.int64)
    h, k, l = (np.rint(a[foot]).astype(np.int64) for a in hkl)
    key[foot] = hkl_key(h, k, l)
    return bragg, key, lam


def reflections(bragg, key, lam):
    """Per-reflection truth for one image; returns (keys, dict of columns)."""
    foot = key >= 0
    keys, inv = np.unique(key[foot], return_inverse=True)
    b = bragg[foot]
    ys, xs = np.nonzero(foot)
    i_true = np.bincount(inv, weights=b)
    n_px = np.bincount(inv)
    n_eff = i_true**2 / np.bincount(inv, weights=b * b)
    lam_mean = np.bincount(inv, weights=b * lam[foot]) / i_true
    x_cen = np.bincount(inv, weights=b * xs) / i_true
    y_cen = np.bincount(inv, weights=b * ys) / i_true

    bmax = np.zeros(len(keys))
    np.maximum.at(bmax, inv, b)
    strong = b >= REL_THR * bmax[inv]
    mask = np.zeros(bragg.shape, dtype=bool)
    mask[ys[strong], xs[strong]] = True
    comp, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    pairs = np.unique(np.stack([inv[strong], comp[ys[strong], xs[strong]]]), axis=1)
    n_comp = np.bincount(pairs[0], minlength=len(keys))

    cols = dict(i_true=i_true, n_px=n_px, n_eff=n_eff, lam_mean=lam_mean,
                x_cen=x_cen, y_cen=y_cen, n_comp=n_comp)
    return keys, cols


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in COLUMNS) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("truth_dir")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.truth_dir, "truth_reflections.csv")

    rows = []
    for path, name in shot_names(a.truth_dir, a.run):
        bragg, key, lam = load_shot(path, name)
        keys, cols = reflections(bragg, key, lam)
        h, k, l = key_hkl(keys)
        for j in range(len(keys)):
            rows.append(dict(shot=name, h=h[j], k=k[j], l=l[j],
                             **{c: (f"{v[j]:.6g}" if v.dtype.kind == "f" else v[j]) for c, v in cols.items()}))
        print(f"{name}: {len(keys)} reflections, {cols['i_true'].sum():.4g} Bragg photons, "
              f"{(bragg > 0).sum()} footprint pixels")
    write_csv(out, rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
