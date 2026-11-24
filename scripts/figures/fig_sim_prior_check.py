"""Prior checks of the Wilson intensity prior: against the simulation's true intensities, and prior-predictively.

Reads `prior_check_summary.csv` and, when present, `prior_predictive.csv` of
`scripts/sim_truth/prior_check.py`.

Figure `sim_prior_check_<dataset>`: (a) the ratio of the mean true intensity to
the prior mean, per resolution bin, for each run (solid acentric, dashed
centric; 1 means the prior mean is right); (b) 12 var(u) of the PIT values
u = F_prior(I_true) in the same bins (1 when the true intensities have the
prior's dispersion); (c) the ratio per wavelength bin, every reflection.
Resolution bins are drawn at their mean 1/d^2 with d labelled.

Figure `sim_prior_predictive_<dataset>`: for each shoebox statistic (total count
over valid pixels, maximum pixel, median pixel) and each resolution bin, the
5..95 % quantiles of the prior draw against those of the real shoeboxes on log
axes; points on the diagonal mean the prior predictive matches the data there.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_prior_check.py --src DIR
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
STATS = (("total", "total count"), ("max", "maximum pixel"), ("median", "median pixel"))


def x_of(lo, hi):
    """Bin position on 1/d^2: the midpoint of the edges, the open low-resolution bin at 1/(1.25 lo)^2."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    hi = np.where(np.isfinite(hi), hi, 1.25 * lo)
    return 0.5 * (1 / lo**2 + 1 / hi**2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    figstyle.use()
    roles = figstyle.roles()
    s = pl.read_csv(a.src / "prior_check_summary.csv")
    dataset = s["dataset"][0]
    runs = list(dict.fromkeys(s["method"].to_list()))
    col = figstyle.arm_colors(runs)

    fig, ax = plt.subplots(1, 3, figsize=figstyle.figsize("full", aspect=0.34), layout="constrained")
    for r in runs:
        for group, ls in (("acentric", "-"), ("centric", "--")):
            d = s.filter(method=r, binning="d", group=group).sort("lo", descending=True)
            x = x_of(d["lo"], d["hi"])
            ax[0].plot(x, d["ratio"], ls, marker="o", ms=2.5, color=col[r], lw=1.0,
                       label=r if group == "acentric" else None)
            ax[1].plot(x, d["var12"], ls, marker="o", ms=2.5, color=col[r], lw=1.0)
        w = s.filter(method=r, binning="wavelength", group="all").sort("lo")
        ax[2].plot(0.5 * (w["lo"] + w["hi"]), w["ratio"], "-", marker="o", ms=2.5, color=col[r], lw=1.0)
    for i in range(3):
        ax[i].axhline(1, color=roles["frame"], lw=0.6)
    d0 = s.filter(method=runs[0], binning="d", group="all").sort("lo", descending=True)
    ticks = x_of(d0["lo"], d0["hi"])
    for i in (0, 1):
        ax[i].set_xticks(ticks, [f"{v:g}" for v in d0["lo"]], fontsize="xx-small", rotation=90)
        ax[i].set_xlabel("$d$ bin, low edge (\\AA)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("mean $I_\\mathrm{true}$ / prior mean")
    ax[1].set_ylabel("$12\\,\\mathrm{var}\\,u$")
    ax[2].set_xlabel("wavelength (\\AA)")
    ax[2].set_ylabel("mean $I_\\mathrm{true}$ / prior mean")
    ax[0].legend(fontsize="xx-small", frameon=False)
    for p in figstyle.save(fig, OUT / f"sim_prior_check_{dataset}{a.tag}"):
        print(p)

    pp = a.src / "prior_predictive.csv"
    if not pp.exists():
        return
    q = pl.read_csv(pp)
    sims = list(dict.fromkeys(q["method"].to_list()))
    bins = q.select("lo", "hi").unique().sort("lo", descending=True).rows()
    ramp = figstyle.epoch_ramp(len(bins))[1]
    fig, ax = plt.subplots(len(sims), 3, figsize=figstyle.figsize("full", aspect=0.34 * len(sims)),
                           layout="constrained", squeeze=False)
    for i, sname in enumerate(sims):
        for j, (stat, lab) in enumerate(STATS):
            sub = q.filter(method=sname, statistic=stat)
            lo_all, hi_all = [], []
            for b, (lo, hi) in enumerate(bins):
                d = sub.filter(lo=lo).sort("q")
                if d.height == 0:
                    continue
                ax[i, j].plot(d["real"], d["sim"], "-o", ms=2, lw=0.8, color=ramp[b],
                              label=f"{lo:g}" if (i, j) == (0, 0) else None)
                lo_all.append(min(d["real"].min(), d["sim"].min()))
                hi_all.append(max(d["real"].max(), d["sim"].max()))
            lim = (max(min(lo_all), 0.5), max(hi_all) * 1.1)
            ax[i, j].plot(lim, lim, color=roles["frame"], lw=0.6)
            ax[i, j].set_xscale("log")
            ax[i, j].set_yscale("log")
            ax[i, j].set_xlabel(f"real, {lab}")
            ax[i, j].set_ylabel(f"{sname}, {lab}" if j == 0 else "")
    ax[0, 0].legend(title="$d$ bin (\\AA)", fontsize="xx-small", title_fontsize="xx-small", frameon=False)
    for p in figstyle.save(fig, OUT / f"sim_prior_predictive_{dataset}{a.tag}"):
        print(p)


if __name__ == "__main__":
    main()
