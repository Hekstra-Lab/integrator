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
    on (image_num, h, k, l); it has no gate. Rows whose window origin
    (`box_x0`, `box_y0`) differs from the shoebox `bbox` are dropped (24 of
    162,117 and 81 of 462,541 on mos010_pc).

With --sample N, the picks are N in-domain reflections spread evenly over
log10 i_true (0.25-decade bins from 3 to 10^4 photons) instead of --targets; the
npz then serves `fit_profile_family.py`.

With --bg-out, over every in-domain reflection, each method's background is
compared with the truth: b_true is the mean of the `background` image over
the reflection's window (photons per pixel), the method's b is laue's `bg` or
the arm's `qbg_mean`, db = b - b_true, and dI_bg = -db * n_eff of the method's
own profile on the window, the intensity error a background error of db
causes in a profile fit. Per method, per bin of i_true and per resolution
shell (D_EDGES of score_against_truth.py), the long CSV holds n and the
median b_true, median and mean db, sd db, mean db / b_true and median dI_bg.
It also prints sum(counts) / sum(sim_image + background) over all windows,
which is 1 when the noisy images are Poisson draws of exactly those rates.

With --fit-out, every method's profile p (and the truth profile, as method
`truth_mask_profile`: `sim_image * own` on the window, normalised) is put through
ONE estimator, the profile fit with the true background,

    I_fit = sum p (c - b_true) / sum p^2,

on the stored counts c, so methods differ only in p. Against the truth rates
lam = sim_image + background on the window it also computes the expectation
E = sum p (lam - b_true) / sum p^2 and variance V = sum p^2 lam / (sum p^2)^2,
the Hellinger distance sqrt(1 - sum sqrt(p p_true)) from the truth profile,
and I_own = sum p (c - b) / sum p^2 with the method's own background b. Per
method, per bin of i_true and per resolution shell, the long CSV holds n,
sum_ratio = sum I_fit / sum i_true, rel_dev_mean (rows with i_true >= 1),
dev_sd of I_fit - i_true, mean E / i_true, median sqrt(V), median of
V_truth_mask_profile / V (the efficiency of p relative to the truth-mask profile), mean
Hellinger and sum_ratio_own; also the median and interquartile range of
I_fit - i_true (the residual is heavy-tailed, so the sd alone misleads), the
p5 / median / p95 of kappa = sum p p_true / sum p^2 (the scale a profile fit
puts on this reflection's own signal), the median of cos(p, p_true), and the
median and p95 of the neighbour pickup sum p s_other / sum p^2, with s_other
the noiseless signal of every other reflection on the window (`sim_image` on
pixels not owned), which is additive and does not scale with this
reflection's intensity. The reference is named `truth_mask_profile`: the
noiseless owned-pixel shape put through this profile fit. With --fit-mask the
shoebox mask (the overlap mask of make_sim_shoeboxes.py finalize) is applied
to every profile, the truth-mask profile included, and renormalised, so all
methods see the same pixels; it is printed per set.

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


def fit_test(a, md, counts, side, rid, shot, H, K, L, i_true, dom, d_all, truth_dir, profiles) -> None:
    """The fixed-estimator profile test of --fit-out; see the module docstring."""
    import sys

    import h5py
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_against_truth import D_EDGES

    n = len(rid)
    x0 = np.asarray(md["bbox.0"]).astype(np.int64)
    y0 = np.asarray(md["bbox.2"]).astype(np.int64)
    dy, dx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    names = ["truth_mask_profile"] + list(profiles)
    keys = ("fit", "E", "V", "hel", "own", "kappa", "cos", "pick")
    res = {m: {k: np.full(n, np.nan) for k in keys} for m in names}
    masks = np.load(f"{a.data}/masks.npy", mmap_mode="r") if a.fit_mask else None
    if masks is not None:
        sub = np.sort(np.random.default_rng(0).choice(n, min(n, 20000), replace=False))
        print(f"--fit-mask: masked pixel fraction {1 - np.asarray(masks[sub]).astype(bool).mean():.4f}")
    with h5py.File(truth_dir / f"background_{a.run}.h5", "r") as fb:
        bg_img = fb["background"][()]
    with h5py.File(truth_dir / f"shots_{a.run}_rank0.h5", "r") as f:
        for sh in np.unique(shot):
            sel = np.flatnonzero((shot == sh) & dom)
            if not len(sel):
                continue
            nm = f"shot_{a.run}_{sh:05d}"
            rows = (y0[sel, None, None] + dy).reshape(len(sel), -1)
            cols = (x0[sel, None, None] + dx).reshape(len(sel), -1)
            img = f[f"sim_image/{nm}"][()]
            bragg = img[rows, cols].astype(np.float64)
            with np.errstate(invalid="ignore"):
                own = np.ones_like(bragg, dtype=bool)
                for c, ref in zip("hkl", (H, K, L)):
                    own &= np.rint(f[f"{c}_data/{nm}"][()][rows, cols]) == ref[sel, None]
            bg = bg_img[rows, cols].astype(np.float64)
            b_true = bg.mean(1, keepdims=True)
            lam = bragg + bg
            c = np.asarray(counts[np.sort(sel)], dtype=np.float64)[np.argsort(np.argsort(sel))]
            order = np.argsort(np.argsort(sel))
            mk = (np.asarray(masks[np.sort(sel)]).astype(bool)[order] if masks is not None
                  else np.ones_like(bragg, dtype=bool))
            other = bragg * (~own)
            pt = bragg * own * mk
            pt = pt / np.maximum(pt.sum(1, keepdims=True), 1e-30)
            for m in names:
                if m == "truth_mask_profile":
                    P, bm = pt, np.full(len(sel), np.nan)
                else:
                    P = profiles[m][0][sel].astype(np.float64)
                    bm = profiles[m][1][sel]
                ok = np.isfinite(P).all(1)
                P = np.where(np.isfinite(P), P, 0.0) * mk
                P = P / np.maximum(P.sum(1, keepdims=True), 1e-30)
                s2 = np.maximum((P ** 2).sum(1), 1e-30)
                r = res[m]
                r["fit"][sel] = np.where(ok, (P * (c - b_true)).sum(1) / s2, np.nan)
                r["E"][sel] = np.where(ok, (P * (lam - b_true)).sum(1) / s2, np.nan)
                r["V"][sel] = np.where(ok, (P ** 2 * lam).sum(1) / s2 ** 2, np.nan)
                r["hel"][sel] = np.where(ok, np.sqrt(np.clip(1 - np.sqrt(P * pt).sum(1), 0, None)), np.nan)
                r["own"][sel] = np.where(ok, (P * (c - bm[:, None])).sum(1) / s2, np.nan)
                r["kappa"][sel] = np.where(ok, (P * pt).sum(1) / s2, np.nan)
                r["cos"][sel] = np.where(ok, (P * pt).sum(1) / np.sqrt(s2 * np.maximum((pt ** 2).sum(1), 1e-30)), np.nan)
                r["pick"][sel] = np.where(ok, (P * other).sum(1) / s2, np.nan)
    vt = res["truth_mask_profile"]["V"]
    out_rows = []
    for m in names:
        r = res[m]
        bins = [("i_true", lo, hi, (i_true >= lo) & (i_true < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        bins += [("d", d_lo, d_hi, (d_all < d_hi) & (d_all >= d_lo)) for d_hi, d_lo in zip(D_EDGES[:-1], D_EDGES[1:])]
        bins += [("all", 0, np.inf, np.ones(n, bool))]
        for kind, lo, hi, s in bins:
            s = s & dom & np.isfinite(r["fit"])
            if s.sum() < 10:
                continue
            pos = s & (i_true >= 1)
            own_ok = s & np.isfinite(r["own"])
            out_rows.append(dict(
                method=m, dataset=a.dataset, bin_kind=kind, lo=lo, hi=hi, n=int(s.sum()),
                sum_ratio=float(r["fit"][s].sum() / i_true[s].sum()),
                rel_dev_mean=float(((r["fit"][pos] - i_true[pos]) / i_true[pos]).mean()) if pos.sum() >= 10 else np.nan,
                dev_sd=float((r["fit"][s] - i_true[s]).std()),
                expected_ratio_mean=float((r["E"][pos] / i_true[pos]).mean()) if pos.sum() >= 10 else np.nan,
                sigma_median=float(np.median(np.sqrt(r["V"][s]))),
                efficiency_median=float(np.median(vt[s] / r["V"][s])),
                hellinger_mean=float(r["hel"][s].mean()),
                sum_ratio_own=float(r["own"][own_ok].sum() / i_true[own_ok].sum()) if own_ok.sum() >= 10 else np.nan,
                dev_median=float(np.median(r["fit"][s] - i_true[s])),
                dev_iqr=float(np.subtract(*np.percentile(r["fit"][s] - i_true[s], [75, 25]))),
                kappa_p5=float(np.percentile(r["kappa"][s], 5)), kappa_median=float(np.median(r["kappa"][s])),
                kappa_p95=float(np.percentile(r["kappa"][s], 95)), cos_median=float(np.median(r["cos"][s])),
                pickup_median=float(np.median(r["pick"][s])), pickup_p95=float(np.percentile(r["pick"][s], 95)),
                masked=bool(a.fit_mask)))
        s = dom & np.isfinite(r["fit"])
        print(f"fit {m}: n {s.sum():,}; sum I_fit / sum i_true {r['fit'][s].sum() / i_true[s].sum():.4f}; "
              f"sd(I_fit - i_true) {(r['fit'][s] - i_true[s]).std():.1f}; median efficiency vs truth profile "
              f"{np.median(vt[s] / r['V'][s]):.3f}; mean Hellinger {r['hel'][s].mean():.3f}")
    pd.DataFrame(out_rows).to_csv(a.fit_out, mode="a", header=not os.path.exists(a.fit_out), index=False)
    print(f"appended {len(out_rows)} rows to {a.fit_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="*", help="name=parquet path or glob (the flushed parts of one export); may be empty with --sample")
    ap.add_argument("--data", required=True, help="shoebox dataset directory")
    ap.add_argument("--truth-dir", required=True, help="datsimx truth directory (shots_<run>_rank0.h5, background_<run>.h5)")
    ap.add_argument("--truth-csv", required=True)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--fit-mask", action="store_true",
                    help="apply the shoebox mask to every profile in the --fit-out test (default: not)")
    ap.add_argument("--fit-out", default=None, help="append the fixed-estimator profile test here (long CSV)")
    ap.add_argument("--bg-out", default=None, help="append the background comparison here (long CSV)")
    ap.add_argument("--laue-profiles", default=None, help="laue_profiles.py npz of the same prediction set")
    ap.add_argument("--targets", default="3,10,30,100,300,1000,3000")
    ap.add_argument("--pick", default=None,
                    help="draw these reflections instead of one per --targets value: comma-separated "
                         "shot:h:k:l (shot = image_num), in that order; outside the domain is allowed")
    ap.add_argument("--lam-lo", type=float, default=1.035)
    ap.add_argument("--lam-hi", type=float, default=1.165)
    ap.add_argument("--max-mixed", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="instead of --targets, N in-domain reflections spread evenly over log10 i_true in [0.5, 4] "
                         "(N/14 per 0.25-decade bin), for fitting a profile family to the true profiles")
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
    if a.pick:
        mk = pd.Index(key(shot, H, K, L))
        for spec in a.pick.split(","):
            sh, h, k, l_ = (int(v) for v in spec.split(":"))
            i = mk.get_indexer(key(np.array([sh]), np.array([h]), np.array([k]), np.array([l_])))[0]
            if i < 0:
                raise SystemExit(f"--pick {spec}: no shoebox with that shot and hkl")
            pick.append(int(i))
        targets = np.array([i_true[i] for i in pick])
    elif a.sample:
        edges = 10 ** np.arange(0.5, 4.01, 0.25)
        per = max(1, a.sample // (len(edges) - 1))
        for lo, hi in zip(edges[:-1], edges[1:]):
            cand = np.flatnonzero(dom & (i_true >= lo) & (i_true < hi))
            if len(cand):
                pick += rng.choice(cand, size=min(per, len(cand)), replace=False).tolist()
        targets = np.array([i_true[i] for i in pick])
    else:
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

    rows, bg_rows = [], []
    d_all = np.asarray(md["d"], float)
    b_true = None
    if a.bg_out:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from score_against_truth import D_EDGES

        x0a = np.asarray(md["bbox.0"]).astype(np.int64)
        y0a = np.asarray(md["bbox.2"]).astype(np.int64)

        def window_sums(img, sel):
            ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1))
            ii[1:, 1:] = np.cumsum(np.cumsum(img, 0, dtype=np.float64), 1)
            x, y = x0a[sel], y0a[sel]
            return ii[y + side, x + side] - ii[y, x + side] - ii[y + side, x] + ii[y, x]

        with h5py.File(truth_dir / f"background_{a.run}.h5", "r") as fb:
            bg_img = fb["background"][()].astype(np.float64)
        b_true = window_sums(bg_img, slice(None)) / side ** 2
        bragg_sum = np.zeros(len(rid))
        with h5py.File(truth_dir / f"shots_{a.run}_rank0.h5", "r") as f:
            for sh in np.unique(shot):
                sel = np.flatnonzero(shot == sh)
                bragg_sum[sel] = window_sums(f[f"sim_image/shot_{a.run}_{sh:05d}"][()].astype(np.float64), sel)
        csum = np.concatenate([np.asarray(counts[i:i + 20000], dtype=np.float64).sum(1)
                               for i in range(0, counts.shape[0], 20000)])
        ratio = csum / (bragg_sum + b_true * side ** 2)
        print(f"sum(counts) / sum(sim_image + background) per window: median {np.median(ratio):.4f}, "
              f"p5 {np.percentile(ratio, 5):.4f}, p95 {np.percentile(ratio, 95):.4f}; over all windows "
              f"{csum.sum() / (bragg_sum + b_true * side ** 2).sum():.5f}")
        print(f"b_true per pixel over windows: median {np.median(b_true):.2f}, p5 {np.percentile(b_true, 5):.2f}, "
              f"p95 {np.percentile(b_true, 95):.2f}")
        out["b_true"] = b_true[pick]

        def bg_compare(name, b_hat, at, neff):
            """Background of one method against b_true, binned by i_true and by d."""
            bt, it, dd, dm = b_true[at], i_true[at], d_all[at], dom[at]
            db = b_hat - bt
            dI = -db * neff
            bins = [("i_true", lo, hi, dm & (it >= lo) & (it < hi)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
            bins += [("d", d_lo, d_hi, dm & (dd < d_hi) & (dd >= d_lo)) for d_hi, d_lo in zip(D_EDGES[:-1], D_EDGES[1:])]
            bins += [("all", 0, np.inf, dm)]
            for kind, lo, hi, s in bins:
                if s.sum() < 10:
                    continue
                bg_rows.append(dict(method=name, dataset=a.dataset, bin_kind=kind, lo=lo, hi=hi, n=int(s.sum()),
                                    b_true_median=float(np.median(bt[s])), db_median=float(np.median(db[s])),
                                    db_mean=float(db[s].mean()), db_sd=float(db[s].std()),
                                    rel_db_mean=float((db[s] / bt[s]).mean()), dI_bg_median=float(np.median(dI[s]))))
            print(f"{name}: background db median {np.median(db[dm]):+.3f} photons/px, mean db/b_true "
                  f"{(db[dm] / bt[dm]).mean():+.4f}, median dI_bg {np.median(dI[dm]):+.1f} photons")

    fit_profiles = {}

    def add(name, prof_all, ids, gate_all=None, bg_all=None):
        """Store the picks' profile (and gate) and the n_eff rows of one method."""
        if a.fit_out:
            ix = pd.Index(rid).get_indexer(ids)
            P = np.full((len(rid), side * side), np.nan, dtype=np.float32)
            P[ix[ix >= 0]] = prof_all[ix >= 0]
            B = np.full(len(rid), np.nan)
            if bg_all is not None:
                B[ix[ix >= 0]] = np.asarray(bg_all, float)[ix >= 0]
            fit_profiles[name] = (P, B)
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
        if bg_all is not None:
            out[f"{name}_bg"] = np.asarray(bg_all, float)[pos]
            if b_true is not None:
                bg_compare(name, np.asarray(bg_all, float)[sel], at[sel], neff[sel])

    if a.laue_profiles:
        lp = np.load(a.laue_profiles)
        lk = key(lp["image_num"], lp["h"].astype(np.int64), lp["k"].astype(np.int64), lp["l"].astype(np.int64))
        at = pd.Index(key(shot, H, K, L)).get_indexer(lk)
        m = at >= 0
        print(f"laue_integrate: {m.sum():,} of {len(lk):,} fitted profiles match a shoebox")
        bx0 = np.asarray(md["bbox.0"]).astype(np.int64)[at[m]]
        by0 = np.asarray(md["bbox.2"]).astype(np.int64)[at[m]]
        same = (lp["box_x0"][m] == bx0) & (lp["box_y0"][m] == by0)
        # laue floors a float32 centroid, the cutter a float64 one, so a centroid
        # on a pixel boundary can land one pixel apart; those rows are dropped
        print(f"laue_integrate: {int((~same).sum())} windows placed differently from the shoeboxes, dropped")
        idx = np.flatnonzero(m)[same]
        add("laue_integrate", np.asarray(lp["profile_box"][idx], dtype=np.float64), rid[at[m]][same],
            bg_all=np.asarray(lp["bg"][idx], dtype=np.float64).reshape(len(idx), -1).mean(1))
        arm_names = ["laue_integrate"]
    else:
        arm_names = []

    for spec in a.arms:
        name, path = spec.split("=", 1)
        df = pl.read_parquet(path, columns=["refl_ids", "qp_mean", "signal_prob", "qbg_mean"])
        ids = df["refl_ids"].to_numpy().astype(np.int64)
        qp = np.stack(df["qp_mean"].to_numpy()) if df["qp_mean"].dtype != pl.Array else df["qp_mean"].to_numpy()
        qp = np.asarray(qp, dtype=np.float64).reshape(len(ids), -1)
        gate = np.asarray(np.stack(df["signal_prob"].to_numpy()), dtype=np.float64).reshape(len(ids), -1)
        add(name, qp, ids, gate, bg_all=df["qbg_mean"].to_numpy().astype(np.float64))
        arm_names.append(name)
    out["arms"] = np.array(arm_names)
    print("truth n_eff of the picks: " + ", ".join(f"{v:.1f}" for v in n_eff_true[pick]))
    np.savez_compressed(a.out, **out)
    print("->", a.out)
    if a.fit_out:
        fit_test(a, md, counts, side, rid, shot, H, K, L, i_true, dom, d_all, truth_dir, fit_profiles)
    if a.bg_out and bg_rows:
        pd.DataFrame(bg_rows).to_csv(a.bg_out, mode="a", header=not os.path.exists(a.bg_out), index=False)
        print(f"appended {len(bg_rows)} rows to {a.bg_out}")
    if a.stats_out:
        pd.DataFrame(rows).to_csv(a.stats_out, mode="a", header=not os.path.exists(a.stats_out), index=False)
        print(f"appended {len(rows)} rows to {a.stats_out}")


if __name__ == "__main__":
    main()
