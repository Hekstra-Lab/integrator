"""Score an unmerged integrated MTZ against the per-reflection truth of a datsimx run.

Join: (shot, H, K, L), with truth shot = the number in `shot_<run>_<NNNNN>`.
laue.integrate writes BATCH = experiment id + 1, and the experiment ids are
renumbered whenever `laue.refine` drops an experiment, so BATCH is the shot only
until the first dropped one. --expt EXPT (the experiment list the MTZ was
integrated from) maps each experiment to its frame in the image file instead,
shot = imageset index + 1. Without --expt, shot = BATCH (the earlier behaviour,
correct only when no experiment was dropped). An MTZ written by
`integrator.predict` has BATCH = image_num + 1, where the shoeboxes of
`make_sim_shoeboxes.py` already store image_num = shot; --batch-convention
image_num then takes shot = BATCH - 1. The indices of both
sides are in the true crystal's setting when the geometry was injected, so no
asymmetric-unit mapping is applied. Two controls are printed beside the join:

  * the join rate of the sign-flipped key (-H, -K, -L), which a correct join beats
  * |lambda_mtz - lam_mean| on joined rows: both sides carry a wavelength and
    neither was fitted to the other

Per observation, in photons (I_hat = I / gain; the ADC offset is removed by the
background subtraction):

    dev = I_hat - i_true            z = dev / (SIGI / gain)

Reported per bin of i_true (noiseless, so binning on it carries no
errors-in-variables bias): count, median and mean of dev, mean of dev / i_true,
sd of dev, mean and sd of z (sd(z) = 1 for calibrated sigmas).

Domain, fixed before scoring and from the truth alone:
  * lam_mean within [--lam-lo, --lam-hi], inside the simulated spectrum, where
    reflections are fully excited
  * mixed_0p25 <= --max-mixed, so the truth sum rests on single-reflection pixels

usage: python score_against_truth.py TRUTH_CSV INTEGRATED_MTZ --method NAME --dataset NAME
           [--gain 0.7 --lam-lo 1.035 --lam-hi 1.165 --max-mixed 0.05 --out scores.csv]
Appends long-format rows (method, dataset, bin, ...) to --out. With
--joined-out PATH it also appends every joined row, raw MTZ values in ADU
beside the truth columns (method, dataset, shot, h, k, l, i_true, lam_mean,
mixed_0p25, n_eff, in_domain, gain, I, SIGI, BG, d), for
`scripts/figures/fig_sim_truth_scatter.py`. BG is laue.integrate's per-pixel
mean background of the window in ADU, ADC offset included; d is the resolution
from the MTZ's cell. Needs
reciprocalspaceship (integrator environment).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import reciprocalspaceship as rs

EDGES = [0, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e9]


def key(shot, h, k, l):
    return (((shot.astype(np.int64) * 1024 + (h + 256)) * 1024 + (k + 256)) * 1024 + (l + 256))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("truth_csv")
    ap.add_argument("mtz")
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--lam-lo", type=float, default=1.035)
    ap.add_argument("--lam-hi", type=float, default=1.165)
    ap.add_argument("--max-mixed", type=float, default=0.05)
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--expt", default=None,
                    help="experiment list of the MTZ; maps BATCH to the image-file frame (default: shot = BATCH)")
    ap.add_argument("--batch-convention", choices=("laue", "image_num"), default="laue",
                    help="laue: BATCH from laue.integrate (default); image_num: BATCH = shot + 1, "
                         "as integrator.predict writes it")
    ap.add_argument("--joined-out", default=None, help="append the joined rows here (default: not written)")
    a = ap.parse_args()

    t = pd.read_csv(a.truth_csv)
    t["shot_n"] = t["shot"].str.rsplit("_", n=1).str[1].astype(int)
    m = rs.read_mtz(a.mtz).reset_index()
    hkl = m[["H", "K", "L"]].to_numpy().astype(np.int64)
    batch = m["BATCH"].to_numpy().astype(np.int64)
    if a.batch_convention == "image_num":
        if a.expt:
            raise SystemExit("--expt maps the laue BATCH convention; it does not apply to image_num")
        batch = batch - 1
    if a.expt:
        from dxtbx.model.experiment_list import ExperimentListFactory

        el = ExperimentListFactory.from_json_file(a.expt, check_format=False)
        frame = np.array([e.imageset.indices()[0] for e in el], dtype=np.int64)
        assert batch.min() >= 1 and batch.max() <= len(frame), "BATCH outside the experiment list"
        moved = int((frame + 1 != np.arange(1, len(frame) + 1)).sum())
        print(f"--expt: {len(frame)} experiments, {moved} with frame + 1 != id + 1")
        batch = frame[batch - 1] + 1

    tk = key(t["shot_n"].to_numpy(), t["h"].to_numpy(), t["k"].to_numpy(), t["l"].to_numpy())
    mk = key(batch, hkl[:, 0], hkl[:, 1], hkl[:, 2])
    mk_flip = key(batch, -hkl[:, 0], -hkl[:, 1], -hkl[:, 2])
    tset = pd.Index(tk)
    join = tset.get_indexer(mk)
    rate = (join >= 0).mean()
    rate_flip = (tset.get_indexer(mk_flip) >= 0).mean()
    ok = join >= 0
    tj = t.iloc[join[ok]].reset_index(drop=True)
    mj = m[ok].reset_index(drop=True)
    dlam = np.abs(mj["wavelength"].to_numpy() - tj["lam_mean"].to_numpy())
    print(f"{len(m):,} MTZ rows; joined {ok.sum():,} ({rate:.3f}); sign-flipped key joins {rate_flip:.3f}")
    print(f"|lambda_mtz - lam_mean| on joined rows: median {np.median(dlam):.5f} A, p95 {np.percentile(dlam, 95):.5f} A")

    i_true = tj["i_true"].to_numpy()
    i_hat = mj["I"].to_numpy() / a.gain
    s_hat = mj["SIGI"].to_numpy() / a.gain
    dom = ((tj["lam_mean"] >= a.lam_lo) & (tj["lam_mean"] <= a.lam_hi)
           & (tj["mixed_0p25"] <= a.max_mixed)).to_numpy() & np.isfinite(i_hat) & (s_hat > 0)
    print(f"domain: {dom.sum():,} of {len(dom):,} joined (lam in [{a.lam_lo}, {a.lam_hi}], mixed_0p25 <= {a.max_mixed})")

    if a.joined_out:
        cols = {c: mj[c].to_numpy() for c in ("I", "SIGI", "BG") if c in mj.columns}
        cols["d"] = rs.read_mtz(a.mtz).compute_dHKL().reset_index()["dHKL"].to_numpy()[ok]
        j = pd.DataFrame(dict(method=a.method, dataset=a.dataset, shot=tj["shot_n"], h=tj["h"], k=tj["k"],
                              l=tj["l"], i_true=i_true, lam_mean=tj["lam_mean"], mixed_0p25=tj["mixed_0p25"],
                              n_eff=tj["n_eff"], in_domain=dom, gain=a.gain, **cols))
        j.to_csv(a.joined_out, mode="a", header=not os.path.exists(a.joined_out), index=False)
        print(f"appended {len(j):,} joined rows to {a.joined_out}")

    dev = i_hat - i_true
    z = dev / s_hat
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = dom & (i_true >= lo) & (i_true < hi)
        if s.sum() < 10:
            continue
        rows.append(dict(method=a.method, dataset=a.dataset, i_true_lo=lo, i_true_hi=hi, n=int(s.sum()),
                         i_true_median=float(np.median(i_true[s])),
                         dev_median=float(np.median(dev[s])), dev_mean=float(dev[s].mean()),
                         rel_dev_mean=float((dev[s] / i_true[s]).mean()), dev_sd=float(dev[s].std()),
                         z_mean=float(z[s].mean()), z_sd=float(z[s].std())))
    s = dom
    rows.append(dict(method=a.method, dataset=a.dataset, i_true_lo=0, i_true_hi=np.inf, n=int(s.sum()),
                     i_true_median=float(np.median(i_true[s])), dev_median=float(np.median(dev[s])),
                     dev_mean=float(dev[s].mean()), rel_dev_mean=np.nan, dev_sd=float(dev[s].std()),
                     z_mean=float(z[s].mean()), z_sd=float(z[s].std())))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.drop(columns=["method", "dataset"]).round(4).to_string(index=False))
    print(f"appended {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
