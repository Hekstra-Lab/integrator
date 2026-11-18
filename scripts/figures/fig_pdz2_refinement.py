"""R-work and R-free per PDZ2 arm, against `laue.integrate`.

The endpoint comparison 3 is ranked on. CCsym is a half-dataset correlation
and `laue.integrate` borrows profiles from strong neighbours, which feeds
both halves of it; R-free against the deposited model at fixed geometry
cannot be inflated that way, so it is the number the dataset is judged on.

**Every arm is drawn as its replicates, not as a mean alone.** On Laue data
the careless merging seed moves R-free by about 0.0012 and dominates the
phenix seed about ninefold, and training itself is unseeded and worth about
0.0050 in R-free. A single refinement is therefore not a measurement, and a
bar chart of one number per arm would imply a precision the data does not
have. The mean is marked, the replicates are shown around it, and the
reference arm's spread is carried across the panel as a band so a gap can be
read against it.

Reads `docs/figures/data/pdz2_refinement.csv`, written by
`scripts/figures/tidy_pdz2.py` and pulled by
`bash scripts/figures/pull_figdata.sh pdz2`: one row per
(method, dataset, replicate) with `r_work` and `r_free`.

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_pdz2_refinement.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arms import label as arm_label  # noqa: E402


def figstyle():
    """figTeX, imported at the call.

    figTeX is deliberately not a dependency of this project, so importing it
    at module scope would make the helpers below untestable anywhere it is
    not installed.
    """
    import figstyle as _figstyle

    return _figstyle


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data" / "pdz2_refinement.csv"
OUT = ROOT / "docs" / "figures" / "pdz2"
REFERENCE = "laue"
PANELS = (("r_free", r"$R_\mathrm{free}$"), ("r_work", r"$R_\mathrm{work}$"))
# What a gap has to clear to be worth reading, from the seed studies: the
# unseeded-training spread in R-free. Drawn as a band on the reference.
NOISE_R_FREE = 0.0050


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--dataset", default=None,
                   help="draw only this dataset (default: each one present)")
    p.add_argument("--prefix", default="refinement")
    return p.parse_args()


def summarise(df: pl.DataFrame, metric: str,
              order: list[str] | None = None) -> pl.DataFrame:
    """Mean and spread per arm, with the replicates kept alongside.

    `n` is the number of refinements behind the mean, which the caller
    reports: a mean over one replicate and a mean over three look identical
    on an axis and are not the same claim.

    `order` fixes the row order. The panels share a y axis, so each sorting
    itself by its own metric puts the arms in two different orders on one
    set of tick labels -- whichever panel is drawn last wins the labels and
    the other panel's points then sit against the wrong names. R-work and
    R-free do not rank the arms the same way here, so this was not
    hypothetical.
    """
    stats = (
        df.drop_nulls(metric)
        .group_by("method")
        .agg(
            pl.col(metric).mean().alias("mean"),
            pl.col(metric).min().alias("lo"),
            pl.col(metric).max().alias("hi"),
            pl.col(metric).len().alias("n"),
            pl.col(metric).alias("values"),
        )
    )
    if order is None:
        return stats.sort("mean")
    rank = {m: i for i, m in enumerate(order)}
    return (stats.with_columns(
        pl.col("method").replace_strict(rank, default=len(rank))
        .alias("_rank"))
        .sort("_rank").drop("_rank"))


def ranking(df: pl.DataFrame, metric: str) -> list[str]:
    """Arms best-first on `metric`, which is the order both panels use."""
    return (df.drop_nulls(metric).group_by("method")
            .agg(pl.col(metric).mean().alias("mean"))
            .sort("mean")["method"].to_list())


def draw_panel(ax, stats: pl.DataFrame, metric: str, label: str,
               palette: dict) -> None:
    arms = stats["method"].to_list()
    y = np.arange(len(arms))

    ref = stats.filter(pl.col("method") == REFERENCE)
    if ref.height and metric == "r_free":
        # the reference's own mean, with the run-to-run spread around it, so
        # an arm inside the band is not distinguishable from the reference
        centre = float(ref["mean"][0])
        ax.axvspan(centre - NOISE_R_FREE, centre + NOISE_R_FREE,
                   color=figstyle().roles().get("grid", "0.85"),
                   alpha=0.45, lw=0, zorder=0)
        # the reference line takes the reference ARM's own colour, so the
        # line and the row it comes from agree; `reference_style`'s fixed
        # colour collides with whichever arm happens to be given it
        ref_style = figstyle().reference_style()
        ref_style["color"] = palette.get(REFERENCE, ref_style["color"])
        ax.axvline(centre, **ref_style)

    for i, row in enumerate(stats.iter_rows(named=True)):
        colour = palette.get(row["method"])
        values = list(row["values"])
        # the replicates, jittered off the row so coincident points are
        # visible rather than overprinted
        offs = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(values, [i + o for o in offs], s=11, color=colour,
                   zorder=3, alpha=0.85, edgecolors="none")
        ax.plot([row["mean"]], [i], marker="|", ms=13, mew=1.8,
                color=colour, zorder=4)
        if len(values) > 1:
            ax.plot([row["lo"], row["hi"]], [i, i], lw=0.9, color=colour,
                    alpha=0.6, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels([arm_label(a) for a in arms])
    ax.set_xlabel(label)
    ax.margins(x=0.12)


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"no {args.data}. It is written by "
            "scripts/figures/tidy_pdz2.py, pulled by "
            "`bash scripts/figures/pull_figdata.sh pdz2`"
        )
    df = pl.read_csv(args.data, infer_schema_length=None)
    missing = {"method", "dataset", "r_work", "r_free"} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.data} is missing {sorted(missing)}")

    figstyle().use()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = ([args.dataset] if args.dataset
                else df["dataset"].unique().sort().to_list())

    n = 0
    for dataset in datasets:
        sub = df.filter(pl.col("dataset") == dataset)
        if not sub.height:
            logger.warning("%s: no rows", dataset)
            continue
        # sorted, with the reference first: `arm_colors` assigns by position,
        # so an unstable input order would repaint the arms between renders
        # and break "an arm keeps its colour across figures"
        arms = sorted(sub["method"].unique().to_list(),
                      key=lambda m: (m != REFERENCE, m))
        palette = dict(figstyle().arm_colors(arms))

        fig, axes = plt.subplots(
            1, len(PANELS), sharey=True,
            figsize=figstyle().figsize("full", aspect=0.34),
        )
        flat = np.atleast_1d(axes)
        # once, not per panel: the axes share y, so inverting in the loop
        # would invert twice and put the lowest R-free at the bottom
        flat[0].invert_yaxis()
        # one order for both panels, from the metric the arms are ranked on
        order = ranking(sub, PANELS[0][0])
        for ax, (metric, label) in zip(flat, PANELS, strict=True):
            stats = summarise(sub, metric, order)
            draw_panel(ax, stats, metric, label, palette)
            if metric == "r_free":
                for row in stats.iter_rows(named=True):
                    logger.info("  %-14s %s = %.4f over %d refinement(s)",
                                row["method"], metric, row["mean"], row["n"])
        path = args.out_dir / f"{args.prefix}_{dataset}"
        figstyle().save(fig, path)
        plt.close(fig)
        logger.info("Saved %s", path)
        n += 1

    if not n:
        raise SystemExit("nothing drawn")


if __name__ == "__main__":
    main()
