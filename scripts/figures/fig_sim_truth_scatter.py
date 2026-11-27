"""Integrated intensity against the simulated truth intensity, one panel per prediction set.

With --quantity noise the same layout draws sigma / I_hat against I_hat (observations with
I_hat > 0, log axes) with the Poisson line sigma / I = 1 / sqrt(I) and the median per log bin;
with --quantity background, each reflection's estimated background per pixel against the true
one (from dump_profile_truth_examples.py --bg-rows-out), with the identity and the binned median.

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
          "svae_conjugate_posterior": "svae_conjugate, posterior mean",
          # the mos010 careless -> phenix arms (2026-11-25)
          "t0_rb": "T0, profile fit at export",
          "p2_rb": "P2, profile fit at export",
          "b1_rb": "B1, profile fit at export",
          "b1_laplace": "B1, Laplace + collapsed posterior mean",
          "geom_laplace": "geometric prior, collapsed posterior mean",
          "svae_rb": "svae (learned basis), profile fit at export",
          "svae_bg_rb": "svae_bg (learned basis), profile fit at export",
          "geom_both_rb": "geom_both (geometric basis), profile fit at export",
          "geom_both_posterior": "geom_both, posterior mean"}


def _intensity(ax, d, a, roles, edges, full):
    x, y = d["i_true"].to_numpy(), d["i_hat"].to_numpy()
    ax.scatter(x, y, s=0.6, alpha=0.15, linewidths=0, rasterized=True, color=roles["accent"])
    lo, hi = 0.1, max(full["i_true"].max(), 1e4)
    grid = np.logspace(np.log10(lo), np.log10(hi), 100)
    ax.plot(grid, grid, color=roles["ink"], lw=0.7, label="$\\hat I = I_\\mathrm{true}$")
    if a.psf_fraction > 0:
        ax.plot(grid, a.psf_fraction * grid, color=roles["ink"], lw=0.7, ls="--",
                label=f"$\\hat I = {a.psf_fraction}\\,I_\\mathrm{{true}}$")
    cx, cy = _binned_median(x, y, edges)
    ax.plot(cx, cy, color=roles["ref"], lw=1.2, marker="o", ms=2.5, label="median $\\hat I$ per bin")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_xlim(lo, hi)


def _noise(ax, d, roles):
    y, s = d["i_hat"].to_numpy(), (d["SIGI"] / d["gain"]).to_numpy()
    ok = (y > 0) & (s > 0)
    y, r = y[ok], s[ok] / y[ok]
    ax.scatter(y, r, s=0.6, alpha=0.15, linewidths=0, rasterized=True, color=roles["accent"])
    g = np.logspace(-1, 5, 50)
    ax.plot(g, 1 / np.sqrt(g), color=roles["ink"], lw=0.8, ls="--", label="Poisson, $\\sigma/I = 1/\\sqrt{I}$")
    cx, cy = _binned_median(y, r, np.logspace(-1, 5, 31))
    ax.plot(cx, cy, color=roles["ref"], lw=1.2, marker="o", ms=2.5, label="median $\\sigma/I$ per bin")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(1e-3, 1e3)


def _background(ax, d, roles):
    x, y = d["b_true"].to_numpy(), d["b_hat"].to_numpy()
    ax.scatter(x, y, s=0.6, alpha=0.15, linewidths=0, rasterized=True, color=roles["accent"])
    lo, hi = np.quantile(x, [0.001, 0.999])
    ax.plot([lo, hi], [lo, hi], color=roles["ink"], lw=0.7, label="$\\hat b = b_\\mathrm{true}$")
    cx, cy = _binned_median(x, y, np.linspace(lo, hi, 21), log=False)
    ax.plot(cx, cy, color=roles["ref"], lw=1.2, marker="o", ms=2.5, label="median $\\hat b$ per bin")


def _binned_median(x, y, edges, log=True):
    idx = np.digitize(x, edges)
    cx, cy = [], []
    for i in range(1, len(edges)):
        sel = idx == i
        if sel.sum() >= 20:
            cx.append(np.sqrt(edges[i - 1] * edges[i]) if log else 0.5 * (edges[i - 1] + edges[i]))
            cy.append(np.median(y[sel]))
    return cx, cy


AXES = {  # quantity -> (x label, y label, title, output stem)
    "intensity": ("$I_\\mathrm{true}$ (photons)", "$\\hat I = I_\\mathrm{MTZ} / g$ (photons)",
                  "integrated against truth intensity, per observation", "sim_truth_scatter"),
    "noise": ("$\\hat I = I_\\mathrm{MTZ} / g$ (photons)", "$\\sigma / \\hat I$ (noise / signal)",
              "noise / signal against signal, per observation with $\\hat I > 0$", "sim_noise_signal"),
    "background": ("$b_\\mathrm{true}$ (photons / pixel)", "$\\hat b$ (photons / pixel)",
                   "per-reflection background against truth", "sim_background_scatter"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--quantity", choices=tuple(AXES), default="intensity",
                    help="intensity (default, the original figure); noise: sigma / I against I with the Poisson "
                         "line; background: per-reflection estimate against the true background, --src then "
                         "the parquet of dump_profile_truth_examples.py --bg-rows-out")
    ap.add_argument("--psf-fraction", type=float, default=PSF_LOCAL,
                    help="draw I_hat = f I_true; 0 omits the line (default %(default)s)")
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--tag", default="",
                    help="appended to the output name, so a second set of methods does not replace the first")
    a = ap.parse_args()
    if not a.src.exists():
        raise SystemExit(f"{a.src} is missing; copy it from the sims/<grid>/<row>/joined_vs_truth.csv")
    if a.quantity == "background":
        full = pl.read_parquet(a.src).filter(pl.col("in_domain"))
    else:
        full = pl.read_csv(a.src, infer_schema_length=None).filter(pl.col("in_domain"))
        full = full.with_columns((pl.col("I") / pl.col("gain")).alias("i_hat"))
    figstyle.use()
    roles = figstyle.roles()
    edges = np.logspace(-1, 4.5, 23)
    xl, yl, title, stem = AXES[a.quantity]
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
            if a.quantity == "intensity":
                _intensity(ax, d, a, roles, edges, full)
            elif a.quantity == "noise":
                _noise(ax, d, roles)
            else:
                _background(ax, d, roles)
            ax.set_title(f"{TITLES.get(m, m)}\nn = {len(d):,}", fontsize=6)
        for ax in axes[-1]:
            ax.set_xlabel(xl)
        # one y label for the figure: per row, the long label of each row ran
        # into the next one's at three panels per row
        fig.supylabel(yl, fontsize=7)
        axes[0, 0].legend(loc="upper left" if a.quantity != "noise" else "upper right", fontsize=6)
        fig.suptitle(f"{dataset}: {title}", fontsize=7)
        OUT.mkdir(parents=True, exist_ok=True)
        for p in figstyle.save(fig, OUT / (f"{stem}_{dataset}" + (f"_{a.tag}" if a.tag else ""))):
            print(p)
        plt.close(fig)


if __name__ == "__main__":
    main()
