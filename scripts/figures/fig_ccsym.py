"""CCsym by resolution shell, one panel per pump delay, arms on one axis.

PDZ2 / 1116 is the only dataset whose endpoint is the half-dataset
correlation of $F(h) - F(2\\text{-fold}\\cdot h)$, which is zero without an
electric field. So the **off panel is the control and is drawn**: it is not a
baseline to subtract silently but the measurement of how much of any raw
CCsym is reproducible non-field asymmetry, and `laue.integrate` reads
0.10--0.20 there where the true answer is exactly zero.

Each line is the mean over the three cross-validation repeats and the band
is their full spread, because a single repeat is not a measurement: the
careless merging seed moves the mean by about 0.012, which is the floor any
difference between arms has to clear.

The statistic is **Spearman**. `custom_ccsym.py` defaults to it and the
deposited `stats_for_production.sh` passes no `-m`, so every published PDZ2
CCsym including the paper's is a rank correlation, and a Pearson value is
not comparable with it.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_ccsym.py \\
            --data docs/figures/data/pdz2_ccsym.csv
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from integrator.utils import metrics  # noqa: E402


def figstyle():
    """figTeX, imported at the call.

    figTeX is deliberately not a dependency of this project, so importing it
    at module scope would make the axis and ordering helpers below
    untestable anywhere it is not installed.
    """
    import figstyle as _figstyle

    return _figstyle


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data" / "pdz2_ccsym.csv"
OUT = ROOT / "docs" / "figures" / "pdz2"
# the four conditions in time order; alphabetical would put 100ns first
DELAYS = ("off", "50ns", "100ns", "200ns")
# the arm that is the reference rather than a model arm
REFERENCE = "laue"
TITLES = {
    "off": "field off (the control; true value 0)",
    "50ns": r"50\,ns",
    "100ns": r"100\,ns",
    "200ns": r"200\,ns",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--dataset", default=None,
                   help="draw only this dataset (default: each one)")
    p.add_argument("--prefix", default="ccsym")
    p.add_argument("--template", default=None,
                   help="figTeX template to size for (article, paper, book, "
                        "slides, poster); default is figstyle's")
    return p.parse_args()


def delay_order(values) -> list[str]:
    """The delays present, in time order, with any unknown one appended."""
    seen = [v for v in values if v not in (None, "")]
    known = [d for d in DELAYS if d in seen]
    return known + sorted(set(seen) - set(known))


def arm_order(values) -> list[str]:
    """Arms with the reference last, so it draws on top of the model arms."""
    arms = sorted(v for v in values if v not in (None, ""))
    return [a for a in arms if a != REFERENCE] + (
        [REFERENCE] if REFERENCE in arms else []
    )


def shell_axis(frame: pl.DataFrame) -> list[float]:
    """Shell centres in $1/d^2$, from the edges rather than the bin index.

    A bin index is not a resolution: two arms could in principle bin
    differently, and a reader cannot read a resolution off an index.
    `shell_midpoint` cuts on $1/d^2$, which is where the shells were cut.
    """
    out = []
    for d_max, d_min in zip(frame["d_max"], frame["d_min"], strict=True):
        try:
            d = metrics.shell_midpoint(float(d_max), float(d_min))
            out.append(1.0 / d**2)
        except (TypeError, ValueError, ZeroDivisionError):
            out.append(float("nan"))
    return out


def resolution_ticks(ax, frame: pl.DataFrame, P) -> None:
    """Shell labels in d, horizontal, thinned to the panel's width.

    `parts.shell_axis` keeps a minor tick at every shell and labels only as
    many as the size model says fit side by side, so nothing is rotated.
    Rotated labels were costing about a line and a half of panel height and
    are unreadable at slide size; a thinned horizontal axis reads at every
    template.
    """
    seen, s2, ds = set(), [], []
    for d_max, d_min in zip(frame["d_max"], frame["d_min"], strict=True):
        key = (d_max, d_min)
        if key in seen:
            continue
        try:
            d = metrics.shell_midpoint(float(d_max), float(d_min))
        except (TypeError, ValueError):
            continue
        seen.add(key)
        s2.append(1.0 / d**2)
        ds.append(d)
    if not s2:
        return
    order = sorted(range(len(s2)), key=lambda i: s2[i])
    figstyle().parts.shell_axis(ax, P, [s2[i] for i in order],
                               [ds[i] for i in order])
    ax.set_xlabel(r"resolution (\AA)")
    ax.grid(axis="x", visible=False)


def draw_panel(ax, panel: pl.DataFrame, arms: list[str], palette: dict,
               P) -> bool:
    """One delay: the mean over repeats per arm, with the repeat spread."""
    drawn = False
    for arm in arms:
        part = panel.filter(pl.col("method") == arm)
        if not len(part):
            continue
        by_bin = (
            part.group_by("bin")
            .agg(
                pl.col("ccsym").mean().alias("mean"),
                pl.col("ccsym").min().alias("lo"),
                pl.col("ccsym").max().alias("hi"),
                pl.col("d_max").first(),
                pl.col("d_min").first(),
            )
            .sort("bin")
        )
        x = shell_axis(by_bin)
        # the reference is drawn by `parts.reference`, which carries the
        # vermilion, the dash and the z-order that mark a reference in
        # every other figure; a model arm takes its own colour
        if arm == REFERENCE:
            band = figstyle().reference_style()["color"]
            ax.fill_between(x, by_bin["lo"], by_bin["hi"],
                            color=band, alpha=0.15, linewidth=0)
            figstyle().parts.reference(ax, x, by_bin["mean"], label=arm)
        else:
            color = palette.get(arm)
            ax.fill_between(x, by_bin["lo"], by_bin["hi"],
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(x, by_bin["mean"], label=arm, lw=1.0, color=color)
        drawn = True
    if not drawn:
        return False
    ax.axhline(0.0, lw=0.5, color="0.6", zorder=0)
    resolution_ticks(ax, panel.sort("bin"), P)
    ax.set_ylabel(r"$\mathrm{CC}_\mathrm{sym}$ (Spearman)")
    return True


def draw(df: pl.DataFrame, out_dir: Path, dataset: str, prefix: str) -> int:
    delays = delay_order(df["delay"].unique().to_list())
    arms = arm_order(df["method"].unique().to_list())
    if not delays or not arms:
        logger.warning("nothing to draw for %s", dataset)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = figstyle().arm_colors(arms)
    slug = str(dataset).replace(" ", "_")
    written = 0

    # one panel per delay, sized to a single figure in the template: a
    # column where the template has columns, the text block where it does
    # not. `panels` reads the template's own geometry, which `figsize` on
    # figTeX's journal path does not.
    for delay in delays:
        panel = df.filter(pl.col("delay") == delay)
        P = figstyle().panels("1", column=True)
        ax = P.axs
        if draw_panel(ax, panel, arms, dict(palette), P):
            P.head(ax, 0, TITLES.get(delay, delay))
            top_legend(P, ax)
            P.finish()
            path = out_dir / f"{prefix}_{delay}_{slug}"
            figstyle().save(P.fig, path)
            logger.info("Saved %s", path)
            written += 1
        plt.close(P.fig)

    # the reported figure: every delay on one grid, a shared y-axis so the
    # off panel can be read against the others rather than rescaled to fill
    n = len(delays)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    P = figstyle().panels(f"{rows}x{cols}", shared="y")
    flat = list(np_ravel(P.axs))
    for i, delay in enumerate(delays):
        if draw_panel(flat[i], df.filter(pl.col("delay") == delay), arms,
                      dict(palette), P):
            P.head(flat[i], i, TITLES.get(delay, delay))
        # every panel is the same binning against the same axis, so the x
        # label goes on the bottom row only; the shared y suppresses the
        # inner y tick numbers, and the y label belongs on the left column
        if i < n - cols:
            flat[i].set_xlabel("")
        if i % cols:
            flat[i].set_ylabel("")
    for ax in flat[n:]:
        ax.set_visible(False)
    # the legend as a strip above the panels rather than inside the first
    # one: with four arms and a band each it covered the weak end of the
    # off panel, which is the control and the thing being read
    top_legend(P, flat[0])
    P.finish()
    path = out_dir / f"{prefix}_{slug}"
    figstyle().save(P.fig, path)
    logger.info("Saved %s (%d panels)", path, n)
    plt.close(P.fig)
    return written + 1


def np_ravel(axs):
    """`P.axs` is an array for a grid and a single axes for one panel."""
    import numpy as np

    return np.ravel(np.asarray(axs, dtype=object))


def top_legend(P, ax) -> None:
    """The arm legend as a boxed strip above the panels.

    Boxed, because the ink theme draws it on the page rather than on a
    panel and an unframed strip over a title reads as part of the title.
    `Panels.finish` sets the frame to the spine weight.

    The number of columns is MEASURED rather than set to the number of
    entries: four arms in one row is 47 pt wider than the article text
    block and was cut off in the output. The strip is drawn, measured, and
    given a row more until it fits, so the same call works at 243 pt and
    at 1015 pt.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n = len(labels)
    for ncols in [n] + [c for c in (3, 2, 1) if c < n]:
        leg = P.top_legend(handles, labels, frameon=True, ncols=ncols)
        r = P.fig.canvas.get_renderer()
        w_pt = leg.get_window_extent(r).width * 72 / P.fig.dpi
        if w_pt <= P.info["W"]:
            return
        leg.remove()
        P._top_legend = None
    P.top_legend(handles, labels, frameon=True, ncols=1)


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"no {args.data}; the cluster side is "
            "scripts/figures/tidy_pdz2.py, pulled by "
            "`bash scripts/figures/pull_figdata.sh pdz2`"
        )
    if args.template:
        figstyle().use(args.template)
    else:
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
