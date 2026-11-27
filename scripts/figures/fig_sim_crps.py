"""CRPS of each integrator's exported intensities against the simulation's true intensity, by true-intensity bin.

Reads `sim_crps_mos010_pc.csv` of `scripts/sim_truth/crps_against_truth.py`
(long: method, dataset, bin, n, crps_rel, crps_abs). CRPS is the expected
absolute error of the export's predictive distribution, so it is lowest only
for an estimate that is both unbiased and calibrated; plotted as the mean of
CRPS / I_true per bin of I_true. Left: the value; right: the ratio to B1's
i_rb, the reference export of the svae_kmixv2_post run. One line per family,
the representative chosen in LABELS; laue.integrate as the reference style.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_crps.py --src docs/figures/data/sim_crps_mos010_pc.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "sim"
LABELS = {
    "svae_posterior": "svae, conjugate posterior",
    "svae_streak_rb": "N1, i_rb",
    "svae_mixture_rb": "M1, i_rb",
    "svae_kmix_base_rb": "T0, i_rb",
    "svae_kmixv2_post_avg_avg": "B1, i_rb",
    "svae_kmixv2_post_avg_lc": "B1, Laplace + collapsed",
    "svae_kmixv2_post_collapsed_geom_avg_geom_posterior": "geometric prior, collapsed",
    # score_exports.sh method names of the mos025 arms (<arm>_<estimator>)
    "svae_bg_rb": "svae_bg (learned basis), i_rb",
    "svae_bg_posterior": "svae_bg (learned basis), posterior",
    "t0_rb": "T0, i_rb",
    "p2_rb": "P2, i_rb",
    "b1_rb": "B1, i_rb",
    "b1_posterior": "B1, posterior",
}
REF = "laue_integrate"
BASE = "svae_kmixv2_post_avg_avg"
BINS = ["10-30", "30-100", "100-300", "300-1000", "1000-3000", ">3000"]
CENTRES = np.sqrt([10 * 30, 30 * 100, 100 * 300, 300 * 1000, 1000 * 3000, 3000 * 1e4])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--base", default=BASE, help="the method the right panel divides by (B1, i_rb)")
    a = ap.parse_args()
    figstyle.use()
    d = pl.read_csv(a.src)
    dataset = d["dataset"][0]
    names = [m for m in LABELS if d.filter(method=m).height]
    col = figstyle.arm_colors([LABELS[m] for m in names])
    val = {m: np.array([d.filter(method=m, bin=b)["crps_rel"][0] for b in BINS]) for m in names + [REF]}
    fig, ax = plt.subplots(1, 2, figsize=figstyle.figsize("full", aspect=0.42), layout="constrained")
    ax[0].plot(CENTRES, val[REF], label="laue.integrate", **figstyle.reference_style())
    for m in names:
        ax[0].plot(CENTRES, val[m], "-o", ms=2.5, lw=1.0, color=col[LABELS[m]], label=LABELS[m])
        ax[1].plot(CENTRES, val[m] / val[a.base], "-o", ms=2.5, lw=1.0, color=col[LABELS[m]])
    ax[1].plot(CENTRES, val[REF] / val[a.base], **figstyle.reference_style())
    ax[1].axhline(1, color=figstyle.roles()["frame"], lw=0.6)
    for x in ax:
        x.set_xscale("log")
        x.set_xlabel("true intensity (photons)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("mean CRPS / $I_\\mathrm{true}$")
    ax[1].set_ylabel(f"ratio to {LABELS.get(a.base, a.base)}")
    ax[1].set_ylim(0.4, 2.0)
    ax[0].legend(fontsize="xx-small", frameon=False)
    for p in figstyle.save(fig, OUT / f"sim_crps_{dataset}"):
        print(p)


if __name__ == "__main__":
    main()
