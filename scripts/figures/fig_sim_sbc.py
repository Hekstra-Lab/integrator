"""Simulation-based calibration of trained runs: ECDF of the PIT values minus the uniform, with simultaneous 95 % bands.

Reads `sbc_ecdf.csv` of `scripts/sim_truth/sbc_ranks.py` (one or more files,
concatenated; `method` and `dataset` key columns). For each quantity the curve
is ECDF(z) - z of the PIT values u = F_q(theta_true) on the grid z = 0..1. A
calibrated posterior stays inside the band; a curve above zero at small z (an
upward bump, then back to zero) means u piles up low, i.e. the truth sits
below the posterior; a curve that is positive below z = 1/2 and negative above
(an S on its side) means the posterior is too narrow; the mirror image, too
wide.

The band is the simultaneous 95 % envelope for n uniform PIT values at the 101
grid points: c sqrt(z (1 - z) / n), with c the 95th percentile, over 4000 draws
of the uniform ECDF, of max_z |ECDF(z) - z| / sqrt(z (1 - z) / n) on 0 < z < 1
(the equal-precision band of Saeilynoja et al. 2022; the ECDF on the grid is
drawn exactly as multinomial bin counts). It is computed here from n, so its
definition lives in one place.

Figure `sim_sbc_<dataset>_<method>`: rows are the quantities I, b, the joint
log-likelihood, the profile's n_eff and i_rb; columns are intensity bins, the
first column every reflection. The posterior rows are binned by the posterior
mean qi_mean, a function of the counts, because an exact posterior is uniform
in a subset selected by the data and is not in one selected by the truth; the
i_rb row, a point estimate, is binned by I_true. Each panel prints n and 12 var(u) (1 when
calibrated, > 1 posterior too narrow).

Figure `sim_sbc_eta_<dataset>_<method>`: the components of the profile shape
vector eta, every reflection.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_sbc.py --src DIR [DIR ...]
"""

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "sim"
ROWS = (("I", "$I$"), ("b", "$b$"), ("loglik", "$\\log p(c \\mid \\theta)$"), ("n_eff", "$n_\\mathrm{eff}$"),
        ("i_rb", "$I_\\mathrm{rb}$"))


@lru_cache(maxsize=None)
def band_c(n: int, n_sim: int = 4000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 1, 101)
    p = np.diff(z)
    cnt = rng.multinomial(n, p, size=n_sim)
    ecdf = np.concatenate([np.zeros((n_sim, 1)), np.cumsum(cnt, 1)], 1) / n
    inner = slice(1, -1)
    s = np.sqrt(z[inner] * (1 - z[inner]) / n)
    return float(np.quantile((np.abs(ecdf[:, inner] - z[inner]) / s).max(1), 0.95))


def band(n: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(0, 1, 101)
    return z, band_c(int(n)) * np.sqrt(z * (1 - z) / max(n, 1))


def panel(ax, d: pl.DataFrame, colour: str, frame: str, summary: dict | None) -> None:
    n = int(d["n"][0])
    if n == 0:
        ax.set_axis_off()
        return
    z, w = band(n)
    ax.fill_between(z, -w, w, color=frame, alpha=0.25, lw=0)
    ax.axhline(0, color=frame, lw=0.5)
    ax.plot(d["z"], d["ecdf_minus_z"], color=colour, lw=1.0)
    txt = f"n {n:,}"
    if summary is not None:
        txt += f"\n$12\\,\\mathrm{{var}}\\,u$ {summary['var12']:.2f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize="xx-small")


def bin_label(lo: float, hi: float, binning: str) -> str:
    if binning == "all":
        return "all"
    return f"{lo:g}--{hi:g}" if np.isfinite(hi) else f"$\\geq${lo:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, nargs="+", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    figstyle.use()
    roles = figstyle.roles()
    e = pl.concat([pl.read_csv(s / "sbc_ecdf.csv") for s in a.src])
    sm = pl.concat([pl.read_csv(s / "sbc_summary.csv") for s in a.src])
    for (method, dataset), em in e.group_by(["method", "dataset"], maintain_order=True):
        smm = sm.filter(method=method, dataset=dataset)
        colour = figstyle.arm_colors([method])[method]

        def summ(q, binning, lo):
            r = smm.filter(quantity=q, binning=binning, lo=lo)
            return r.row(0, named=True) if r.height else None

        edges = em.filter(binning="i_true").select("lo", "hi").unique(maintain_order=True).rows()
        cols = [(-1.0, 1.0)] + edges
        rows = [(q, lab) for q, lab in ROWS if em.filter(quantity=q).height]
        fig, ax = plt.subplots(len(rows), len(cols), figsize=figstyle.figsize("full", aspect=0.16 * len(rows) + 0.05),
                               sharex=True, sharey="row", layout="constrained", squeeze=False)
        for i, (q, lab) in enumerate(rows):
            by = "i_true" if q == "i_rb" else "qi_mean"
            for j, (lo, hi) in enumerate(cols):
                binning = "all" if j == 0 else by
                d = em.filter(quantity=q, binning=binning, lo=lo).sort("z")
                panel(ax[i, j], d, colour, roles["frame"], summ(q, binning, lo))
                if i == 0:
                    ax[i, j].set_title(bin_label(lo, hi, binning), fontsize="x-small")
            ax[i, 0].set_ylabel(lab + ("\nby $I_\\mathrm{true}$" if q == "i_rb" else "") + "\nECDF $-$ $z$",
                                fontsize="x-small")
        for j in range(len(cols)):
            ax[-1, j].set_xlabel("$z$")
        fig.suptitle("posterior mean of $I$ (photons); $I_\\mathrm{true}$ for $I_\\mathrm{rb}$", fontsize="x-small")
        for p in figstyle.save(fig, OUT / f"sim_sbc_{dataset}_{method}{a.tag}"):
            print(p)

        eta = sorted({q for q in em["quantity"].unique().to_list() if q.startswith("eta_")}, key=lambda s: int(s[4:]))
        if not eta:
            continue
        nc = 6
        nr = -(-len(eta) // nc)
        fig, ax = plt.subplots(nr, nc, figsize=figstyle.figsize("full", aspect=0.17 * nr + 0.04), sharex=True,
                               sharey=True, layout="constrained", squeeze=False)
        for k, q in enumerate(eta):
            d = em.filter(quantity=q, binning="all").sort("z")
            panel(ax[k // nc, k % nc], d, colour, roles["frame"], summ(q, "all", -1.0))
            ax[k // nc, k % nc].set_title(f"$\\eta_{{{q[4:]}}}$", fontsize="x-small")
        for k in range(len(eta), nr * nc):
            ax[k // nc, k % nc].set_axis_off()
        for p in figstyle.save(fig, OUT / f"sim_sbc_eta_{dataset}_{method}{a.tag}"):
            print(p)


if __name__ == "__main__":
    main()
