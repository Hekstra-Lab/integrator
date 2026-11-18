"""The Wilson prior's learned scale and B-factor over training.

The prior the model shrinks every intensity toward is not fixed: $\\Sigma_h =
G \\exp(-2 B s^2) \\epsilon_h$ with $G$ and $B$ learned alongside everything
else. So an arm's shrinkage behaviour is only interpretable next to the prior
it was shrinking toward, and these two numbers are that prior.

Two panels, because a B-factor in square angstroms and a scale in counts do
not share an axis. Where a run has more than one scale group the band around
$G$ is `G_min` to `G_max`: the spread across groups says how much the
crystals differ, which the mean alone hides.

Read this against the known failure rather than as a curve.
$G$ and $B$ **collapse together** when $G$ is parameterized in linear space:
the scale runs away, $B$ follows to compensate, and the ELBO barely moves
because the product is what enters the likelihood. The fix is a log-space
$G$ (`stabilize_scale`), so a run with a settled $G$ and a drifting $B$, or
either still moving at the last epoch, is a run whose prior never
identified. The loss also uses $s^2 = 1/(4d^2)$; using $1/d^2$ inflates
$\\exp(2Bs^2)$ and collapses $\\Sigma$ to zero.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_wilson_params.py \\
            --data docs/figures/data/wilson_params.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def figstyle():
    """figTeX, imported at the call; not a dependency of this project."""
    import figstyle as _figstyle

    return _figstyle


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data" / "wilson_params.csv"
OUT = ROOT / "docs" / "figures" / "m9"
# The panels, in reported order: what the prior's width is, then its scale.
# `G_min`/`G_max` are not panels of their own -- they are the band on `G`.
PANELS = (
    ("B", r"$B$ (\AA$^2$)", "Wilson B-factor"),
    ("G_mean", r"$G$", "Wilson scale"),
    ("G", r"$G$", "Wilson scale"),
    ("alpha_mean", r"$\alpha$", "prior concentration"),
)
BAND = {"G_mean": ("G_min", "G_max")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--prefix", default="wilson")
    return p.parse_args()


def panels_of(df: pl.DataFrame) -> list[tuple[str, str, str]]:
    """The listed panels that the table actually has, in reported order.

    `G` and `G_mean` are the same quantity under two names: the logger
    writes `G` for a single scale group and `G_mean` for several, so only
    one of them is ever present and both are listed.
    """
    have = set(df["parameter"].unique().to_list())
    out = [(name, label, title) for name, label, title in PANELS
           if name in have]
    # a parameter the logger adds later should appear rather than vanish
    listed = {name for name, _, _ in PANELS}
    extra = sorted(have - listed - {b for pair in BAND.values() for b in pair})
    return out + [(name, name, name) for name in extra]


def draw_panel(ax, df: pl.DataFrame, parameter: str, arms: list[str],
               palette: dict) -> bool:
    drawn = False
    for arm in arms:
        part = (df.filter((pl.col("method") == arm)
                          & (pl.col("parameter") == parameter))
                .sort("epoch"))
        if not len(part):
            continue
        colour = palette.get(arm)
        lo_name, hi_name = BAND.get(parameter, (None, None))
        if lo_name:
            lo = df.filter((pl.col("method") == arm)
                           & (pl.col("parameter") == lo_name)).sort("epoch")
            hi = df.filter((pl.col("method") == arm)
                           & (pl.col("parameter") == hi_name)).sort("epoch")
            if len(lo) == len(part) and len(hi) == len(part):
                ax.fill_between(part["epoch"], lo["value"], hi["value"],
                                color=colour, alpha=0.15, linewidth=0)
        ax.plot(part["epoch"], part["value"], lw=1.0, color=colour, label=arm)
        drawn = True
    if drawn:
        ax.set_xlabel("epoch")
    return drawn


def draw(df: pl.DataFrame, out_dir: Path, dataset: str, prefix: str) -> int:
    panels = panels_of(df)
    arms = sorted(a for a in df["method"].unique().to_list() if a)
    if not panels or not arms:
        logger.warning("nothing to draw for %s", dataset)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = figstyle().arm_colors(arms)
    slug = str(dataset).replace(" ", "_")
    written = 0

    for name, label, title in panels:
        fig, ax = plt.subplots(
            figsize=figstyle().figsize("half", aspect=0.8),
            layout="constrained")
        if draw_panel(ax, df, name, arms, palette):
            ax.set_ylabel(label)
            ax.set_title(title)
            if len(arms) > 1:
                ax.legend(frameon=False)
            path = out_dir / f"{prefix}_{name.lower()}_{slug}"
            figstyle().save(fig, path)
            logger.info("Saved %s", path)
            written += 1
        plt.close(fig)

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=figstyle().figsize("full", aspect=0.38),
        layout="constrained", squeeze=False)
    for ax, (name, label, title) in zip(axes.ravel(), panels, strict=True):
        if draw_panel(ax, df, name, arms, palette):
            ax.set_ylabel(label)
            ax.set_title(title)
    if len(arms) > 1:
        axes.ravel()[0].legend(frameon=False)
    path = out_dir / f"{prefix}_params_{slug}"
    figstyle().save(fig, path)
    logger.info("Saved %s (%d panels)", path, n)
    plt.close(fig)
    return written + 1


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"no {args.data}; it comes from each run's "
            "plots/wilson_params.csv through scripts/figures/tidy_mono.py, "
            "pulled by `bash scripts/figures/pull_figdata.sh mono`"
        )
    figstyle().use()
    df = pl.read_csv(args.data, infer_schema_length=None)
    datasets = ([args.dataset] if args.dataset
                else sorted(df["dataset"].unique().to_list()))
    total = 0
    for dataset in datasets:
        total += draw(df.filter(pl.col("dataset") == dataset),
                      args.out_dir, dataset, args.prefix)
    if not total:
        raise SystemExit("no figure written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
