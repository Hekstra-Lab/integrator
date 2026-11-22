"""Shoeboxes beside their true profile and each arm's predicted profile and gate, for fig_profile_truth_examples.py.

Reflections are chosen by the truth intensity `i_true` of `truth_table.py`,
never by an estimate: per target in --targets the reflection whose `i_true`
is nearest, drawn with a fixed seed among those within 20 % of it, inside
the scoring domain of `score_against_truth.py` (lam_mean in [--lam-lo,
--lam-hi], mixed_0p25 <= --max-mixed), so the truth-mask profile rests on
pixels this one reflection owns.

Per pick, all on the shoebox window `bbox` of the dataset (the window the
model read):

  * `counts`, the stored counts (photons for the photon-counting images);
  * `bragg`, the noiseless datsimx `sim_image` of that shot on the window;
  * `own`, whether a pixel's true (h, k, l) (`h_data`, `k_data`, `l_data`,
    rounded) is this reflection's;
  * `true_profile`, `bragg * own` divided by its sum;
  * `background`, the `background` image of the background file on the window;
  * per arm, `qp_mean` and `signal_prob` from that arm's prediction parquet
    (`rb_predict_config.py KEYS=both`), joined on `refl_ids`;
  * with --laue-profiles, laue.integrate's fitted profile on the same window
    (`profile_box` of `laue_profiles.py`), as method `laue_integrate`, joined
    on (image_num, h, k, l); it has no gate. Its window origin (`box_x0`,
    `box_y0`) is checked against the shoebox `bbox`.

Over every in-domain reflection it also writes, per arm and per bin of
i_true, the median of n_eff = 1 / sum(p^2) of `qp_mean` beside the median
of the truth table's own `n_eff`, as a long CSV (--stats-out).

    python dump_profile_truth_examples.py --data SHOEBOXES --truth-dir TRUTH \\
        --truth-csv TRUTH/truth_reflections.csv --out profile_truth_examples.npz \\
        --stats-out profile_neff_vs_truth.csv --dataset mos010_pc_default \\
        svae=PARQUET svae_bg=PARQUET svae_conjugate=PARQUET
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

EDGES = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e9]


def key(shot, h, k, l):
    return (((np.asarray(shot, np.int64) * 1024 + (h + 256)) * 1024 + (k + 256)) * 1024 + (l + 256))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="+", help="name=parquet path or glob (the flushed parts of one export)")
    ap.add_argument("--data", required=True, help="shoebox dataset directory")
    ap.add_argument("--truth-dir", required=True, help="datsimx truth directory (shots_<run>_rank0.h5, background_<run>.h5)")
    ap.add_argument("--truth-csv", required=True)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--laue-profiles", default=None, help="laue_profiles.py npz of the same prediction set")
    ap.add_argument("--targets", default="3,10,30,100,300,1000,3000")
    ap.add_argument("--lam-lo", type=float, default=1.035)
    ap.add_argument("--lam-hi", type=float, default=1.165)
    ap.add_argument("--max-mixed", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import h5py
    import pandas as pd
    import polars as pl

    md = np.load(f"{a.data}/metadata.npy", allow_pickle=True)
    md = md.item() if md.dtype == object else md
    counts = np.load(f"{a.data}/counts.npy", mmap_mode="r")
    side = int(round(np.sqrt(counts.shape[1])))
    rid = np.asarray(md["refl_ids"]).astype(np.int64)
    shot = np.asarray(md["image_num"]).astype(np.int64)
    H, K, L = (np.asarray(md[c]).astype(np.int64) for c in ("H", "K", "L"))

    t = pd.read_csv(a.truth_csv)
    t["shot_n"] = t["shot"].str.rsplit("_", n=1).str[1].astype(int)
    tk = pd.Index(key(t["shot_n"].to_numpy(), t["h"].to_numpy(), t["k"].to_numpy(), t["l"].to_numpy()))
    j = tk.get_indexer(key(shot, H, K, L))
    ok = j >= 0
    i_true = np.full(len(rid), np.nan)
    n_eff_true = np.full(len(rid), np.nan)
    i_true[ok] = t["i_true"].to_numpy()[j[ok]]
    n_eff_true[ok] = t["n_eff"].to_numpy()[j[ok]]
    lam = np.full(len(rid), np.nan)
    mixed = np.full(len(rid), np.nan)
    lam[ok] = t["lam_mean"].to_numpy()[j[ok]]
    mixed[ok] = t["mixed_0p25"].to_numpy()[j[ok]]
    dom = ok & (lam >= a.lam_lo) & (lam <= a.lam_hi) & (mixed <= a.max_mixed)
    print(f"{len(rid):,} shoeboxes; {ok.sum():,} joined to the truth table; {dom.sum():,} in the domain")

    rng = np.random.default_rng(a.seed)
    targets = np.array([float(x) for x in a.targets.split(",")])
    pick = []
    for tv in targets:
        cand = np.flatnonzero(dom & (np.abs(i_true - tv) <= 0.2 * tv))
        pick.append(int(rng.choice(cand)) if len(cand) else int(np.nanargmin(np.where(dom, np.abs(i_true - tv), np.inf))))
    pick = np.array(pick)

    out = dict(dataset=a.dataset, targets=targets, side=side, arms=np.array([s.split("=", 1)[0] for s in a.arms]),
               refl_ids=rid[pick], i_true=i_true[pick], n_eff_true=n_eff_true[pick], shot=shot[pick],
               hkl=np.stack([H[pick], K[pick], L[pick]], 1), d=np.asarray(md["d"], float)[pick],
               counts=np.asarray(counts[np.sort(pick)])[np.argsort(np.argsort(pick))].reshape(-1, side, side))
    bb = np.stack([np.asarray(md[f"bbox.{k}"]).astype(np.int64) for k in range(4)], 1)[pick]
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    truth_dir = Path(a.truth_dir)
    bragg, own, bgw = [], [], []
    with h5py.File(truth_dir / f"shots_{a.run}_rank0.h5", "r") as f, \
            h5py.File(truth_dir / f"background_{a.run}.h5", "r") as fb:
        for n, (x0, x1, y0, y1) in zip(range(len(pick)), bb):
            name = f"shot_{a.run}_{shot[pick[n]]:05d}"
            bragg.append(f[f"sim_image/{name}"][y0:y1, x0:x1].astype(np.float64))
            hkl_px = [np.rint(f[f"{c}_data/{name}"][y0:y1, x0:x1]).astype(np.int64) for c in "hkl"]
            own.append((hkl_px[0] == H[pick[n]]) & (hkl_px[1] == K[pick[n]]) & (hkl_px[2] == L[pick[n]]))
            bgw.append(fb["background"][y0:y1, x0:x1].astype(np.float64))
    out["bragg"], out["own"], out["background"] = np.array(bragg), np.array(own), np.array(bgw)
    tp = out["bragg"] * out["own"]
    out["true_profile"] = tp / np.maximum(tp.sum(axis=(1, 2), keepdims=True), 1e-30)

    rows = []

    def add(name, prof_all, ids, gate_all=None):
        """Store the picks' profile (and gate) and the n_eff rows of one method."""
        pos = pd.Index(ids).get_indexer(rid[pick])
        if (pos < 0).any():
            raise SystemExit(f"{name}: {int((pos < 0).sum())} picked reflections missing")
        out[f"{name}_profile"] = prof_all[pos].reshape(-1, side, side)
        if gate_all is not None:
            out[f"{name}_gate"] = gate_all[pos].reshape(-1, side, side)
        neff = 1.0 / np.sum(prof_all ** 2, axis=1)
        at = pd.Index(rid).get_indexer(ids)
        sel = (at >= 0)
        it, nt, dm = i_true[at[sel]], n_eff_true[at[sel]], dom[at[sel]]
        nm = neff[sel]
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = dm & (it >= lo) & (it < hi)
            if s.sum() < 10:
                continue
            rows.append(dict(method=name, dataset=a.dataset, i_true_lo=lo, i_true_hi=hi, n=int(s.sum()),
                             n_eff_model_median=float(np.median(nm[s])), n_eff_true_median=float(np.median(nt[s]))))
        print(f"{name}: {len(ids):,} profiles; picked n_eff " + ", ".join(f"{v:.1f}" for v in neff[pos]))

    if a.laue_profiles:
        lp = np.load(a.laue_profiles)
        lk = key(lp["image_num"], lp["h"].astype(np.int64), lp["k"].astype(np.int64), lp["l"].astype(np.int64))
        at = pd.Index(key(shot, H, K, L)).get_indexer(lk)
        m = at >= 0
        print(f"laue_integrate: {m.sum():,} of {len(lk):,} fitted profiles match a shoebox")
        bx0 = np.asarray(md["bbox.0"]).astype(np.int64)[at[m]]
        by0 = np.asarray(md["bbox.2"]).astype(np.int64)[at[m]]
        moved = int(((lp["box_x0"][m] != bx0) | (lp["box_y0"][m] != by0)).sum())
        if moved:
            raise SystemExit(f"laue_integrate: {moved} windows placed differently from the shoeboxes")
        add("laue_integrate", np.asarray(lp["profile_box"][m], dtype=np.float64), rid[at[m]])
        arm_names = ["laue_integrate"]
    else:
        arm_names = []

    for spec in a.arms:
        name, path = spec.split("=", 1)
        df = pl.read_parquet(path, columns=["refl_ids", "qp_mean", "signal_prob"])
        ids = df["refl_ids"].to_numpy().astype(np.int64)
        qp = np.stack(df["qp_mean"].to_numpy()) if df["qp_mean"].dtype != pl.Array else df["qp_mean"].to_numpy()
        qp = np.asarray(qp, dtype=np.float64).reshape(len(ids), -1)
        gate = np.asarray(np.stack(df["signal_prob"].to_numpy()), dtype=np.float64).reshape(len(ids), -1)
        add(name, qp, ids, gate)
        arm_names.append(name)
    out["arms"] = np.array(arm_names)
    print("truth n_eff of the picks: " + ", ".join(f"{v:.1f}" for v in n_eff_true[pick]))
    np.savez_compressed(a.out, **out)
    print("->", a.out)
    if a.stats_out:
        pd.DataFrame(rows).to_csv(a.stats_out, mode="a", header=not os.path.exists(a.stats_out), index=False)
        print(f"appended {len(rows)} rows to {a.stats_out}")


if __name__ == "__main__":
    main()
