"""Integrated intensity against the simulated truth intensity, one panel per prediction set.

x is the truth intensity $I_\\text{true} = \\sum_p B_p$ of the reflection on that
shot, from `truth_table.py` (photons, noiseless, before the PSF); y is the
estimate in photons, $\\hat I = I / g$ from the unmerged MTZ. Only rows inside the
scoring domain of `score_against_truth.py` are drawn (truth wavelength in
[1.035, 1.165] A, mixed_0p25 <= 0.05). Lines: the identity, $0.854\\,I_\\text{true}$
(the fraction of a point source nanoBragg's 5-px PSF keeps within 5 px,
`docs/sim_truth_protocol.md` section 9), and the median of $\\hat I$ in
logarithmic bins of $I_\\text{true}$. The y axis is symmetric-log, linear within
+-10 photons, so negative estimates are shown.

Data: `score_against_truth.py --joined-out`, copied to
`docs/figures/data/sim_truth_joined_<dataset>.csv`. With --src naming a file
of several methods (laue.integrate and trained arms, both estimators), every
method in it gets a panel, in the order of `TITLES` then the file, wrapped
at --ncols; one figure per dataset in the file. --psf-fraction 0 drops the
PSF line, which does not apply to the photon-counting images.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_truth_scatter.py
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
SRC = ROOT / "docs" / "figures" / "data" / "sim_truth_joined_mos010.csv"
OUT = ROOT / "docs" / "figures" / "sim"
PSF_LOCAL = 0.854
LINTHRESH = 10.0
TITLES = {"laue_integrated": "laue.integrate, default prediction",
          "laue_integrated_all": "laue.integrate, permissive prediction",
          "laue_integrate": "laue.integrate",
          "svae_rb": "svae, profile fit at export",
          "svae_posterior": "svae, posterior mean",
          "svae_bg_rb": "svae_bg, profile fit at export",
          "svae_bg_posterior": "svae_bg, posterior mean",
          "svae_conjugate_rb": "svae_conjugate, profile fit at export",
          "svae_conjugate_posterior": "svae_conjugate, posterior mean"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--psf-fraction", type=float, default=PSF_LOCAL,
                    help="draw I_hat = f I_true; 0 omits the line (default %(default)s)")
    ap.add_argument("--ncols", type=int, default=3)
    a = ap.parse_args()
    if not a.src.exists():
        raise SystemExit(f"{a.src} is missing; copy it from the sims/<grid>/<row>/joined_vs_truth.csv")
    full = pl.read_csv(a.src, infer_schema_length=None).filter(pl.col("in_domain"))
    full = full.with_columns((pl.col("I") / pl.col("gain")).alias("i_hat"))
    figstyle.use()
    roles = figstyle.roles()
    edges = np.logspace(-1, 4.5, 23)
    for dataset in full["dataset"].unique(maintain_order=True).to_list():
        df = full.filter(pl.col("dataset") == dataset)
        present = df["method"].unique(maintain_order=True).to_list()
        methods = [m for m in TITLES if m in present] + [m for m in present if m not in TITLES]
        ncols = min(a.ncols, len(methods))
        nrows = -(-len(methods) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figstyle.figsize("full", aspect=0.5 if nrows == 1 else 0.85 * nrows / ncols),
                                 layout="constrained", squeeze=False, sharex=True, sharey=True)
        for ax in axes.flat[len(methods):]:
            ax.set_visible(False)
        for ax, m in zip(axes.flat, methods):
            d = df.filter(pl.col("method") == m)
            x, y = d["i_true"].to_numpy(), d["i_hat"].to_numpy()
            ax.scatter(x, y, s=0.6, alpha=0.15, linewidths=0, rasterized=True, color=roles["accent"])
            lo, hi = 0.1, max(full["i_true"].max(), 1e4)
            grid = np.logspace(np.log10(lo), np.log10(hi), 100)
            ax.plot(grid, grid, color=roles["ink"], lw=0.7, label="$\\hat I = I_\\mathrm{true}$")
            if a.psf_fraction > 0:
                ax.plot(grid, a.psf_fraction * grid, color=roles["ink"], lw=0.7, ls="--",
                        label=f"$\\hat I = {a.psf_fraction}\\,I_\\mathrm{{true}}$")
            idx = np.digitize(x, edges)
            cx, cy = [], []
            for i in range(1, len(edges)):
                s = idx == i
                if s.sum() >= 20:
                    cx.append(np.sqrt(edges[i - 1] * edges[i]))
                    cy.append(np.median(y[s]))
            ax.plot(cx, cy, color=roles["ref"], lw=1.2, marker="o", ms=2.5,
                    label="median $\\hat I$ per bin")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=LINTHRESH)
            ax.set_xlim(lo, hi)
            ax.set_title(f"{TITLES.get(m, m)}\nn = {len(d):,}", fontsize=6)
        for ax in axes[-1]:
            ax.set_xlabel("$I_\\mathrm{true}$ (photons)")
        for ax in axes[:, 0]:
            ax.set_ylabel("$\\hat I = I_\\mathrm{MTZ} / g$ (photons)")
        axes[0, 0].legend(loc="upper left", fontsize=6)
        fig.suptitle(f"{dataset}: integrated against truth intensity, per observation", fontsize=7)
        OUT.mkdir(parents=True, exist_ok=True)
        for p in figstyle.save(fig, OUT / f"sim_truth_scatter_{dataset}"):
            print(p)
        plt.close(fig)


if __name__ == "__main__":
    main()
