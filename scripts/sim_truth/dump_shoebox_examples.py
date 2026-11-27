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

With --same-reflections the targets are met in the FIRST dataset only, among the
reflections every dataset holds (matched on image_num, H, K, L), and the same
reflections are taken from each dataset, in the same order; the npz then carries
same_reflections = 1 and each dataset's image_num, h, k, l. For two sims that
differ in one parameter (mos010 and mos025, mosaic spread 0.10 and 0.25) this
shows one reflection under each value.

--truth name=TRUTH_SHOTS (repeatable; datsimx datasets only) adds, for the picked
rows of that dataset, `<name>_truth`: the noiseless Bragg image of the run's TRUTH
shots file (`sim_image/shot_<run>_<image_num:05d>`, photons, no background), cut
at the row's own bbox and cropped like the counts, so it covers the same pixels.
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
    ap.add_argument("--truth", action="append", default=[], help="name=TRUTH_SHOTS h5 (datsimx), repeatable")
    ap.add_argument("--same-reflections", action="store_true",
                    help="pick in the first dataset among reflections all datasets hold; take the same ones from each")
    a = ap.parse_args()

    targets = np.array([float(t) for t in a.isigi.split(",")])
    rng = np.random.default_rng(a.seed)
    out = dict(names=np.array([d.split("=", 1)[0] for d in a.datasets]), targets=targets, box=a.box,
               same_reflections=int(a.same_reflections))

    def keys(md):
        f = lambda c: np.asarray(md[c], dtype=np.int64)
        return ((f("image_num") * 1024 + f("H") + 256) * 1024 + f("K") + 256) * 1024 + f("L") + 256

    want = None
    if a.same_reflections:
        common = None
        for spec in a.datasets:
            k = keys(load(spec.split("=", 1)[1])[2])
            _, first = np.unique(k, return_index=True)  # a key held twice is ambiguous: drop it
            k_once = k[first][np.bincount(np.unique(k, return_inverse=True)[1]) == 1]
            common = k_once if common is None else np.intersect1d(common, k_once)
        print(f"reflections held once by every dataset: {len(common):,}")

    for di, spec in enumerate(a.datasets):
        name, path = spec.split("=", 1)
        c, m, md = load(path)
        side = int(round(np.sqrt(c.shape[1])))
        var = np.asarray(md["intensity.sum.variance"], dtype=float)
        isigi = np.where(var > 0, np.asarray(md["intensity.sum.value"], dtype=float) / np.sqrt(np.maximum(var, 1e-30)), np.nan)
        if a.same_reflections and di > 0:
            pos = {v: i for i, v in enumerate(keys(md).tolist())}
            pick = np.array([pos[v] for v in want.tolist()])
        else:
            allowed = np.isin(keys(md), common) if a.same_reflections else np.ones(len(isigi), bool)
            pick = []
            for t in targets:
                tol = max(0.2 * t, 0.5)
                cand = np.flatnonzero(allowed & (np.abs(isigi - t) <= tol))
                pick.append(int(rng.choice(cand)) if len(cand)
                            else int(np.nanargmin(np.where(allowed, np.abs(isigi - t), np.nan))))
            pick = np.array(pick)
            if a.same_reflections:
                want = keys(md)[pick]
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
        for col, src in (("image_num", "image_num"), ("h", "H"), ("k", "K"), ("l", "L")):
            out[f"{name}_{col}"] = np.asarray(md[src], dtype=np.int64)[pick]
        truth_of = dict(t.split("=", 1) for t in a.truth)
        if name in truth_of:
            import h5py

            run = None
            rows_t = np.empty((len(pick), c.shape[1]), dtype=np.float64)
            with h5py.File(truth_of[name], "r") as f:
                keys_h5 = sorted(f["sim_image"].keys())
                run = keys_h5[0].split("_")[1]
                for r, i in enumerate(pick):
                    x0, x1, y0, y1 = (int(md[f"bbox.{j}"][i]) for j in range(4))
                    img = f[f"sim_image/shot_{run}_{int(md['image_num'][i]):05d}"]
                    rows_t[r] = img[y0:y1, x0:x1].astype(np.float64).ravel()
            out[f"{name}_truth"] = crop(rows_t, side, a.box)
            print(f"{name}: noiseless Bragg photons in the picked boxes "
                  + ", ".join(f"{v:.0f}" for v in out[f"{name}_truth"].sum((1, 2))))
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
