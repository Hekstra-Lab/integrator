"""Example shoeboxes from several shoebox datasets, chosen at the same laue I/sigma, for fig_sim_shoeboxes.py.

Each dataset is a directory written by make_sim_shoeboxes.py (counts.npy,
masks.npy, metadata.npy). Per dataset and per target value of laue's
`intensity.sum.value / sqrt(intensity.sum.variance)` the reflection closest
to the target is taken, among those with that ratio within 20 % of it (or
the nearest overall when none is), drawn with a fixed seed. A box larger
than --box is cropped to its central --box x --box, so every panel covers
the same detector area; counts are left in the units the dataset stores,
which are the units training reads.

Per dataset the npz also carries summary numbers over --n-stats boxes drawn
at random: the fraction of valid stored values that are not whole numbers,
the median over boxes of the median of the 1-px border, and the median over
boxes of the border's variance / mean.

    python dump_shoebox_examples.py --out shoebox_examples.npz \\
        name=/path/to/shoeboxes_a name2=/path/to/shoeboxes_b
"""

from __future__ import annotations

import argparse

import numpy as np


def load(path):
    c = np.load(f"{path}/counts.npy", mmap_mode="r")
    m = np.load(f"{path}/masks.npy", mmap_mode="r")
    md = np.load(f"{path}/metadata.npy", allow_pickle=True)
    md = md.item() if md.dtype == object else md
    return c, m, md


def crop(rows, side, box):
    rows = np.asarray(rows).reshape(len(rows), side, side)
    o = (side - box) // 2
    return rows[:, o:o + box, o:o + box]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+", help="name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", type=int, default=25)
    ap.add_argument("--isigi", default="0,2,5,10,30,100", help="target laue I/sigma values, one column each")
    ap.add_argument("--n-stats", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    targets = np.array([float(t) for t in a.isigi.split(",")])
    rng = np.random.default_rng(a.seed)
    out = dict(names=np.array([d.split("=", 1)[0] for d in a.datasets]), targets=targets, box=a.box)
    for spec in a.datasets:
        name, path = spec.split("=", 1)
        c, m, md = load(path)
        side = int(round(np.sqrt(c.shape[1])))
        var = np.asarray(md["intensity.sum.variance"], dtype=float)
        isigi = np.where(var > 0, np.asarray(md["intensity.sum.value"], dtype=float) / np.sqrt(np.maximum(var, 1e-30)), np.nan)
        pick = []
        for t in targets:
            tol = max(0.2 * t, 0.5)
            cand = np.flatnonzero(np.abs(isigi - t) <= tol)
            pick.append(int(rng.choice(cand)) if len(cand) else int(np.nanargmin(np.abs(isigi - t))))
        pick = np.array(pick)
        order = np.argsort(pick)
        rows_c = np.empty((len(pick), c.shape[1]), dtype=np.float64)
        rows_m = np.empty((len(pick), c.shape[1]), dtype=bool)
        rows_c[order] = c[pick[order]]
        rows_m[order] = m[pick[order]]

        s = np.sort(rng.choice(c.shape[0], size=min(a.n_stats, c.shape[0]), replace=False))
        sc, sm = crop(c[s].astype(np.float64), side, a.box), crop(m[s], side, a.box)
        valid = sm & np.isfinite(sc)
        nonint = float(np.mean((sc[valid] % 1) != 0))
        border = np.zeros((a.box, a.box), bool)
        border[[0, -1], :] = border[:, [0, -1]] = True
        bvals = np.where(valid & border, sc, np.nan).reshape(len(s), -1)
        bmed = np.nanmedian(bvals, axis=1)
        bvm = np.nanvar(bvals, axis=1) / np.nanmean(bvals, axis=1)

        out[f"{name}_counts"] = crop(rows_c, side, a.box)
        out[f"{name}_mask"] = crop(rows_m, side, a.box)
        out[f"{name}_isigi"] = isigi[pick]
        out[f"{name}_d"] = np.asarray(md["d"], dtype=float)[pick]
        out[f"{name}_dtype"] = str(c.dtype)
        out[f"{name}_side"] = side
        out[f"{name}_n"] = c.shape[0]
        out[f"{name}_nonint"] = nonint
        out[f"{name}_border_median"] = float(np.nanmedian(bmed))
        out[f"{name}_border_var_over_mean"] = float(np.nanmedian(bvm))
        print(f"{name}: {c.shape[0]:,} boxes {side}x{side} {c.dtype}; non-integer {nonint:.4f}; border median "
              f"{np.nanmedian(bmed):.2f}, var/mean {np.nanmedian(bvm):.3f}; picked I/sigI "
              + ", ".join(f"{v:.1f}" for v in isigi[pick]))
    np.savez_compressed(a.out, **out)
    print("->", a.out)


if __name__ == "__main__":
    main()
