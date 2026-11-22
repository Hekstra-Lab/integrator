"""Integrated intensity scored against the simulated truth, per bin of truth intensity, every method on one set of axes.

Reads the long table of `scripts/sim_truth/score_against_truth.py --out`
(one row per method, dataset and bin of $I_\\text{true}$; the bin with
`i_true_hi = inf` is the whole domain and is not drawn). The x axis is the
median $I_\\text{true}$ of the bin, so every method is placed by the truth,
never by its own estimate. Four panels per dataset, two by two:

- `rel_dev_mean`, the mean of $(\\hat I - I_\\text{true}) / I_\\text{true}$,
  without the bin $[0, 1)$, where truths as small as $10^{-22}$ photons make
  the ratio meaningless;
- `dev_sd`, the standard deviation of $\\hat I - I_\\text{true}$, with
  $\\sqrt{I_\\text{true}}$, the Poisson spread of the signal photons alone,
  drawn for reference;
- `z_mean` and `z_sd` of $z = (\\hat I - I_\\text{true}) / \\hat\\sigma$, with
  0 and 1 drawn for reference.

Colour is the architecture, from `figstyle.arm_colors`, with laue.integrate in
the ink colour; the line style is the estimator (solid: profile fit at export,
`i_rb`/`var_rb`; dashed: posterior mean, `qi_mean`/`qi_var`).

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_truth_scores.py \\
            --src docs/figures/data/sim_truth_scores_mos010_pc.csv

With --by-d (the long table of `score_against_truth.py --by-d-out`) it draws
the same comparison per resolution shell instead, against $1/d^2$ at the
shell's median $d$. Figure `sim_truth_scores_by_d_<dataset>`: every truth
intensity together, six panels: `sum_ratio` $= \\sum \\hat I / \\sum
I_\\text{true}$, `rel_dev_mean` (rows with $I_\\text{true} \\ge 1$), `cc_log`,
the correlation of $\\log \\hat I$ with $\\log I_\\text{true}$, `z_mean`, `z_sd`
and the count. Figure `sim_truth_rel_dev_by_d_<dataset>`: `rel_dev_mean` per
shell, one panel per bin of $I_\\text{true}$, which separates resolution from
the fall of intensity with resolution.
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
SRC = ROOT / "docs" / "figures" / "data" / "sim_truth_scores_mos010_pc.csv"
OUT = ROOT / "docs" / "figures" / "sim"
ARCH = ("svae", "svae_bg", "svae_conjugate")
EST = {"rb": "-", "posterior": "--"}
PANELS = (("rel_dev_mean", "mean $(\\hat I - I_\\mathrm{true}) / I_\\mathrm{true}$"),
          ("dev_sd", "sd $(\\hat I - I_\\mathrm{true})$ (photons)"),
          ("z_mean", "mean $z$"),
          ("z_sd", "sd $z$"))


def split(method: str) -> tuple[str, str | None]:
    """(architecture, estimator) of a method name such as `svae_bg_rb`."""
    for est in EST:
        if method.endswith("_" + est):
            return method[: -len(est) - 1], est
    return method, None


D_PANELS = (("sum_ratio", "$\\sum \\hat I / \\sum I_\\mathrm{true}$", 1.0),
            ("rel_dev_mean", "mean $(\\hat I - I_\\mathrm{true}) / I_\\mathrm{true}$", 0.0),
            ("cc_log", "corr$(\\log \\hat I, \\log I_\\mathrm{true})$", None),
            ("z_mean", "mean $z$", 0.0),
            ("z_sd", "sd $z$", 1.0),
            ("n", "reflections", None))


def by_d(src: Path) -> None:
    full = pl.read_csv(src, infer_schema_length=None)
    colour = figstyle.arm_colors(ARCH)
    ink = figstyle.roles()["ink"]
    ref = figstyle.reference_style()

    def style(method):
        arch, est = split(method)
        return dict(color=colour.get(arch, ink), ls=EST.get(est, ":"), lw=1.0, marker="o", ms=2, label=method)

    for dataset in full["dataset"].unique(maintain_order=True).to_list():
        df = full.filter(pl.col("dataset") == dataset)
        allI = df.filter(pl.col("i_true_hi") == float("inf"))
        fig, axes = plt.subplots(2, 3, figsize=figstyle.figsize("full", aspect=0.62), layout="constrained",
                                 sharex=True)
        axes = axes.ravel()
        for method in allI["method"].unique(maintain_order=True).to_list():
            d = allI.filter(pl.col("method") == method).sort("d_median", descending=True)
            x = 1.0 / d["d_median"].to_numpy() ** 2
            for ax, (col, _, _) in zip(axes, D_PANELS):
                ax.plot(x, d[col].to_numpy(), **style(method))
        for ax, (col, label, r) in zip(axes, D_PANELS):
            if r is not None:
                ax.axhline(r, **ref)
            ax.set_ylabel(label)
        for ax in axes[3:]:
            ax.set_xlabel("$1/d^2$ (\\AA$^{-2}$)")
        axes[0].legend(fontsize=5, loc="best")
        fig.suptitle(f"{dataset}: integrated against truth, by resolution shell, all intensities", fontsize=7)
        for p in figstyle.save(fig, OUT / f"sim_truth_scores_by_d_{dataset}"):
            print(p)
        plt.close(fig)

        bins = (df.filter(pl.col("i_true_hi") != float("inf")).select("i_true_lo", "i_true_hi").unique()
                .sort("i_true_lo").rows())
        fig, axes = plt.subplots(1, len(bins), figsize=figstyle.figsize("full", aspect=0.3), layout="constrained",
                                 sharex=True, squeeze=False)
        for ax, (lo, hi) in zip(axes[0], bins):
            g = df.filter((pl.col("i_true_lo") == lo) & (pl.col("i_true_hi") == hi))
            for method in g["method"].unique(maintain_order=True).to_list():
                d = g.filter(pl.col("method") == method).sort("d_median", descending=True)
                ax.plot(1.0 / d["d_median"].to_numpy() ** 2, d["rel_dev_mean"].to_numpy(), **style(method))
            ax.axhline(0, **ref)
            ax.set_title(f"$I_\\mathrm{{true}} \\in [{lo:g}, {hi:g})$" if hi < 1e8 else f"$I_\\mathrm{{true}} \\ge {lo:g}$",
                         fontsize=6)
            ax.set_xlabel("$1/d^2$ (\\AA$^{-2}$)")
        axes[0][0].set_ylabel("mean $(\\hat I - I_\\mathrm{true}) / I_\\mathrm{true}$")
        axes[0][0].legend(fontsize=4.5, loc="best")
        fig.suptitle(f"{dataset}: relative deviation by resolution shell, per bin of truth intensity", fontsize=7)
        for p in figstyle.save(fig, OUT / f"sim_truth_rel_dev_by_d_{dataset}"):
            print(p)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--by-d", type=Path, default=None,
                    help="score_against_truth.py --by-d-out table; draws the per-shell figures instead")
    a = ap.parse_args()
    if a.by_d is not None:
        figstyle.use()
        OUT.mkdir(parents=True, exist_ok=True)
        by_d(a.by_d)
        return
    if not a.src.exists():
        raise SystemExit(f"{a.src} is missing; it is score_against_truth.py --out, pulled from the cluster")
    full = pl.read_csv(a.src, infer_schema_length=None).filter(pl.col("i_true_hi") != float("inf"))
    figstyle.use()
    ink = figstyle.roles()["ink"]
    colour = figstyle.arm_colors(ARCH)
    for dataset in full["dataset"].unique(maintain_order=True).to_list():
        df = full.filter(pl.col("dataset") == dataset)
        fig, axes = plt.subplots(2, 2, figsize=figstyle.figsize("full", aspect=0.62), layout="constrained",
                                 sharex=True)
        axes = axes.ravel()
        for method in df["method"].unique(maintain_order=True).to_list():
            arch, est = split(method)
            d = df.filter(pl.col("method") == method).sort("i_true_median")
            x = d["i_true_median"].to_numpy()
            style = dict(color=colour.get(arch, ink), ls=EST.get(est, ":"), lw=1.0, marker="o", ms=2,
                         label=method)
            for ax, (col, _) in zip(axes, PANELS):
                y = d[col].to_numpy().astype(float)
                if col == "rel_dev_mean":
                    y = np.where(d["i_true_lo"].to_numpy() < 1, np.nan, y)
                ax.plot(x, y, **style)
        xs = np.logspace(np.log10(max(full["i_true_median"].min(), 0.1)), np.log10(full["i_true_median"].max()), 50)
        ref = figstyle.reference_style()
        axes[0].axhline(0, **ref)
        axes[1].plot(xs, np.sqrt(xs), **ref)
        axes[2].axhline(0, **ref)
        axes[3].axhline(1, **ref)
        axes[1].set_yscale("log")
        for ax, (_, label) in zip(axes, PANELS):
            ax.set_xscale("log")
            if ax in axes[2:]:
                ax.set_xlabel("median $I_\\mathrm{true}$ of the bin (photons)")
            ax.set_ylabel(label)
        axes[0].legend(fontsize=5, loc="best")
        fig.suptitle(f"{dataset}: integrated against truth, by bin of truth intensity", fontsize=7)
        OUT.mkdir(parents=True, exist_ok=True)
        for p in figstyle.save(fig, OUT / f"sim_truth_scores_{dataset}"):
            print(p)
        plt.close(fig)


if __name__ == "__main__":
    main()
