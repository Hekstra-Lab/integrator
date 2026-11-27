"""One simulation integrated on two shoebox sizes: per method, a line per size, against the true intensity.

Rows, all binned on the true intensity I_true (noiseless, so the binning carries no
errors-in-variables bias):

  photons recovered   sum I_fit / sum I_true of the fixed-estimator profile fit
                      (dump_profile_truth_examples.py --fit-out: the profile fit with the true
                      background, I_fit = sum p (c - b_true) / sum p^2, on the stored counts),
                      so methods differ only in their profile p. The true profile cut to each
                      window is drawn in every panel as that window's ceiling.
  relative deviation  mean (I_hat - I_true) / I_true of the method's own export
                      (score_against_truth.py; i_rb for the arms, laue.integrate's intensities)
  CRPS / I_true       the export's predictive distribution against I_true
                      (crps_against_truth.py)

One figure per row above (`<out>_photons`, `<out>_reldev`, `<out>_crps`), one panel per
method; values outside the fixed y range of the first two are drawn at its edge. Inputs, under docs/figures/data, per window W given as
LABEL=PROFILE_DATASET:SCORES_CSV:CRPS_CSV:

    profile_fit_vs_truth_<PROFILE_DATASET>.csv, SCORES_CSV, CRPS_CSV

    uv run --with-editable /Users/luis/master/software/figtex python scripts/figures/fig_sim_window_compare.py \\
        --window "25x25=mos025_pc_m0_default:sim_truth_scores_mos025_pc.csv:sim_crps_mos025_pc.csv" \\
        --window "41x41=mos025_pc_m0_default41:sim_truth_scores_mos025_41_pc.csv:sim_crps_mos025_41_pc.csv" \\
        --out sim_window_mos025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "sim"
# profile_fit method name -> (title, export method in the score/CRPS tables)
METHODS = {
    "laue_integrate": ("laue.integrate", "laue_integrate"),
    "svae_bg": ("svae_bg (learned basis)", "svae_bg_rb"),
    "t0": (r"T0, $i_\mathrm{rb}$", "t0_rb"),
    "p2": (r"P2, $i_\mathrm{rb}$", "p2_rb"),
    "b1": (r"B1, $i_\mathrm{rb}$", "b1_rb"),
    "geom_both": ("geom_both (geometric basis)", "geom_both_rb"),
}
CRPS_BINS = {"10-30": (10, 30), "30-100": (30, 100), "100-300": (100, 300), "300-1000": (300, 1000),
             "1000-3000": (1000, 3000), ">3000": (3000, 1e4)}


def centre(lo, hi):
    hi = np.asarray(hi, dtype=float)
    hi = np.where(np.isfinite(hi) & (hi < 1e8), hi, 3 * np.asarray(lo, float))
    return np.sqrt(np.maximum(np.asarray(lo, float), 0.3) * hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", action="append", required=True, help="LABEL=PROFILE_DATASET:SCORES_CSV:CRPS_CSV")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", default="sim_window_compare")
    ap.add_argument("--min-itrue", type=float, default=3.0, help="leftmost I_true bin edge drawn")
    a = ap.parse_args()

    wins = []
    for spec in a.window:
        lab, rest = spec.split("=", 1)
        ds, sc, cr = rest.split(":")
        wins.append((lab, pl.read_csv(a.data / f"profile_fit_vs_truth_{ds}.csv").filter(bin_kind="i_true"),
                     pl.read_csv(a.data / sc), pl.read_csv(a.data / cr)))
    methods = [m for m in METHODS if any(w[1].filter(method=m).height for w in wins)]
    figstyle.use()
    col = figstyle.arm_colors([w[0] for w in wins])

    def series(kind, m, lab, fit, sc, cr):
        title, exp = METHODS[m]
        if kind == "photons":
            f = fit.filter((pl.col("method") == m) & (pl.col("lo") >= a.min_itrue)).sort("lo")
            return centre(f["lo"], f["hi"]), f["sum_ratio"].to_numpy()
        if kind == "reldev":
            s = sc.filter((pl.col("method") == exp) & (pl.col("i_true_lo") >= a.min_itrue)).sort("i_true_lo")
            return s["i_true_median"].to_numpy(), s["rel_dev_mean"].to_numpy()
        q = cr.filter(method=exp)
        pts = [(np.sqrt(CRPS_BINS[b][0] * CRPS_BINS[b][1]), float(q.filter(bin=b)["crps_rel"][0]))
               for b in CRPS_BINS if q.filter(bin=b).height]
        return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts])) if pts else (np.array([]), np.array([]))

    SPEC = {
        "photons": (r"$\sum \hat I_\mathrm{fit} / \sum I_\mathrm{true}$ $\uparrow$ to the ceiling", (0.0, 1.5), "linear",
                    "photons recovered by the profile fit with the true background (dotted: the true profile cut to the window)"),
        "reldev": (r"mean $(\hat I - I_\mathrm{true})/I_\mathrm{true}$, 0 best", (-1.0, 1.5), "linear",
                   "the export's mean relative deviation from the true intensity"),
        "crps": (r"CRPS$/I_\mathrm{true}$ $\downarrow$", None, "log", "CRPS of the export against the true intensity"),
    }
    cols = 3
    rows = (len(methods) + cols - 1) // cols
    for kind, (ylab, ylim, yscale, what) in SPEC.items():
        fig, ax = plt.subplots(rows, cols, figsize=figstyle.figsize("full", aspect=0.42 * rows), sharex=True,
                               sharey=True, layout="constrained", squeeze=False)
        flat = ax.ravel()
        for k, m in enumerate(methods):
            x_ = flat[k]
            for lab, fit, sc, cr in wins:
                x, y = series(kind, m, lab, fit, sc, cr)
                if ylim is not None:
                    y = np.clip(y, *ylim)
                x_.plot(x, y, "-o", ms=2, lw=1.0, color=col[lab], label=lab)
                if kind == "photons":
                    t = fit.filter((pl.col("method") == "truth_mask_profile") & (pl.col("lo") >= a.min_itrue)).sort("lo")
                    x_.plot(centre(t["lo"], t["hi"]), t["sum_ratio"], ls=":", lw=0.9, color=col[lab],
                            label=f"true profile, {lab}")
            if kind == "photons":
                x_.axhline(1, color=figstyle.roles()["frame"], lw=0.6)
            if kind == "reldev":
                x_.axhline(0, color=figstyle.roles()["frame"], lw=0.6)
            x_.set_title(METHODS[m][0], fontsize="small")
            x_.set_xscale("log")
            x_.set_yscale(yscale)
            if ylim is not None:
                x_.set_ylim(*ylim)
            if k // cols == rows - 1:
                x_.set_xlabel(r"$I_\mathrm{true}$ (photons)")
            if k % cols == 0:
                x_.set_ylabel(ylab, fontsize="small")
        for x_ in flat[len(methods):]:
            x_.set_visible(False)
        h, l = flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside upper center", ncols=len(l), frameon=True, fontsize="small")
        for p in figstyle.save(fig, OUT / f"{a.out}_{kind}"):
            print(p)
        plt.close(fig)


if __name__ == "__main__":
    main()
