"""Compare datsimx sampling settings (enSteps, mosDoms) against a reference setting, shot by shot.

For each test row and the reference row at the same mosaic spread, on one shot:

    total_ratio = sum B_X / sum B_R            (all footprint pixels)
    dev         = (I_X / I_R) / total_ratio - 1  per reflection present in both
    l1          = sum_p |B_X,p / I_X - B_R,p / I_R|  per reflection (0 = same profile, 2 = disjoint)
    n_comp      = components of the 1e-3-of-max footprint (truth_table.py)

Statistics are over reflections with I_R >= --min-photons, and separately over
the 100 brightest by I_R. If datsimx averages over spectrum samples, total_ratio
is 1 between enSteps values; if it sums, total_ratio = enSteps_X / enSteps_R.

usage: python compare_sampling.py GRID.tsv ROOT [--ref-en 322 --ref-doms 100] [--shot shot_1_00001]
Reads ROOT/<name>/truth for each grid row; writes ROOT/sampling_summary.csv and
ROOT/sampling_reflections.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from truth_table import load_shot, reflections, shot_names  # noqa: E402


def read_grid(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t")]
    return rows


def profile_l1(bx, kx, ix, br, kr, ir, keys):
    """Per-reflection L1 distance between the two truth profiles, indexed like `keys`."""
    union = (kx >= 0) | (kr >= 0)
    kx, kr, bx, br = kx[union], kr[union], bx[union], br[union]
    idx_x = np.searchsorted(keys, kx).clip(0, len(keys) - 1)
    idx_r = np.searchsorted(keys, kr).clip(0, len(keys) - 1)
    ok_x = (kx >= 0) & (keys[idx_x] == kx)
    ok_r = (kr >= 0) & (keys[idx_r] == kr)
    px = np.where(ok_x, bx / np.where(ok_x, ix[idx_x], 1), 0.0)
    pr = np.where(ok_r, br / np.where(ok_r, ir[idx_r], 1), 0.0)
    l1 = np.zeros(len(keys))
    same = ok_x & ok_r & (kx == kr)
    np.add.at(l1, idx_x[same], np.abs(px[same] - pr[same]))
    np.add.at(l1, idx_x[ok_x & ~same], px[ok_x & ~same])
    np.add.at(l1, idx_r[ok_r & ~same], pr[ok_r & ~same])
    return l1


def summarise(sel, dev, l1, ncx, ncr):
    if not sel.any():
        return dict(n=0)
    return dict(n=int(sel.sum()),
                dev_median_abs=float(np.median(np.abs(dev[sel]))),
                dev_p95_abs=float(np.percentile(np.abs(dev[sel]), 95)),
                l1_median=float(np.median(l1[sel])),
                l1_p95=float(np.percentile(l1[sel], 95)),
                frac_split_test=float((ncx[sel] > 1).mean()),
                frac_split_ref=float((ncr[sel] > 1).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grid")
    ap.add_argument("root")
    ap.add_argument("--ref-en", type=int, default=322)
    ap.add_argument("--ref-doms", type=int, default=100)
    ap.add_argument("--shot", default="shot_1_00001")
    ap.add_argument("--min-photons", type=float, default=1.0)
    a = ap.parse_args()

    grid = read_grid(a.grid)
    cache = {}

    def get(row):
        name = row["name"]
        if name not in cache:
            tdir = os.path.join(a.root, name, "truth")
            path = dict((n, p) for p, n in shot_names(tdir, int(row.get("run", 1))))[a.shot]
            bragg, key, lam = load_shot(path, a.shot)
            keys, cols = reflections(bragg, key, lam)
            cache[name] = (bragg, key, keys, cols)
        return cache[name]

    summary, per_refl = [], []
    for row in grid:
        refs = [r for r in grid if r["mos_spread"] == row["mos_spread"]
                and int(r["en_steps"]) == a.ref_en and int(r["mos_doms"]) == a.ref_doms]
        if not refs:
            print(f"{row['name']}: no reference row at mos_spread {row['mos_spread']}; skipped")
            continue
        ref = refs[0]
        bx, kx, keys_x, cx = get(row)
        br, kr, keys_r, cr = get(ref)
        common, jx, jr = np.intersect1d(keys_x, keys_r, return_indices=True)
        ix, ir = cx["i_true"][jx], cr["i_true"][jr]
        total_ratio = bx[bx > 0].sum() / br[br > 0].sum()
        dev = ix / ir / total_ratio - 1
        l1 = profile_l1(bx, kx, cx["i_true"][jx], br, kr, cr["i_true"][jr], common)
        ncx, ncr = cx["n_comp"][jx], cr["n_comp"][jr]

        top = np.zeros(len(common), dtype=bool)
        top[np.argsort(ir)[::-1][:100]] = True
        base = dict(name=row["name"], ref=ref["name"], mos_spread=row["mos_spread"],
                    en_steps=row["en_steps"], mos_doms=row["mos_doms"], xtal_size=row["xtal_size"],
                    total_bragg=float(bx[bx > 0].sum()), total_ratio=float(total_ratio),
                    n_refl_test=len(keys_x), n_refl_ref=len(keys_r), n_common=len(common))
        for label, sel in ((f"I_R>={a.min_photons:g}", ir >= a.min_photons), ("top100", top)):
            summary.append(dict(base, subset=label, **summarise(sel, dev, l1, ncx, ncr)))
        for j in range(len(common)):
            per_refl.append(dict(name=row["name"], key=int(common[j]), i_ref=ir[j], i_test=ix[j],
                                 dev=dev[j], l1=l1[j], n_comp_test=int(ncx[j]), n_comp_ref=int(ncr[j])))

    for fname, rows in (("sampling_summary.csv", summary), ("sampling_reflections.csv", per_refl)):
        path = os.path.join(a.root, fname)
        cols = list(dict.fromkeys(c for r in rows for c in r))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")

    hdr = ["name", "subset", "total_bragg", "total_ratio", "n", "dev_median_abs", "dev_p95_abs",
           "l1_median", "l1_p95", "frac_split_test", "frac_split_ref"]
    print("\t".join(hdr))
    for r in summary:
        print("\t".join(f"{r.get(c, ''):.4g}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in hdr))


if __name__ == "__main__":
    main()
