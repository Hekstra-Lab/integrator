"""What careless reports and DIALS cannot be asked for, by resolution shell.

The polychromatic arm scales and merges with careless, so the per-shell
numbers come from `careless.stats` rather than from `dials.merge`:
$CC_{1/2}$ and $CC_\\mathrm{anom}$ from the cross-validation MTZ, $CC_{pred}$
from the predictions, $R_\\mathrm{split}$, $I/\\sigma_I$ and completeness from
the merged file. `scripts/poly/run_pipeline.py --steps stats` runs every one
of them and concatenates the result into `careless_stats.csv`.

Nothing here knows the names careless gives its columns. The long table
carries `tool`, `statistic` and `value`, and a panel is drawn for each
`(tool, statistic)` that appears, so a statistic careless adds later shows
up without a change here.

**Drawn at one checkpoint, not over training** (decided 2026-11-17): one
careless scale is tens of GPU-minutes. When the table does hold several
epochs they go on the sequential ramp, as the monochromatic figures do.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_careless_stats.py \\
            --data docs/figures/data/careless_stats.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from integrator.utils import metrics  # noqa: E402


def figstyle():
    """figTeX, imported at the call.

    figTeX is deliberately not a dependency of this project, so importing
    it at module scope would make the shell-axis and panel-selection
    functions below untestable anywhere it is not installed -- and a test
    that skips itself is how the `merged.html` reader stayed broken.
    """
    import figstyle as _figstyle

    return _figstyle

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data" / "careless_stats.csv"
OUT = ROOT / "docs" / "figures" / "m9"

# How each statistic is drawn and what to call it. A statistic that is not
# listed is still drawn, with its careless column name as the label: the
# figure must not silently omit a number the pipeline produced.
LABELS = {
    "cchalf": r"$\mathrm{CC}_{1/2}$",
    "ccanom": r"$\mathrm{CC}_\mathrm{anom}$",
    "ccpred": r"$\mathrm{CC}_\mathrm{pred}$",
    "rsplit": r"$R_\mathrm{split}$",
    "isigi": r"$I/\sigma_I$",
    "i/sigi": r"$I/\sigma_I$",
    "completeness": "completeness",
    "spearman": r"$\rho_\mathrm{Spearman}$",
    "pearson": r"$r_\mathrm{Pearson}$",
}
# The order the panels are reported in; anything else follows, sorted.
TOOL_ORDER = ("cchalf", "ccanom", "ccpred", "rsplit", "isigi", "completeness")


def label_for(tool: str, statistic: str) -> str:
    key = str(statistic).strip().lower().replace(" ", "")
    if key in LABELS:
        return LABELS[key]
    if tool.lower() in LABELS:
        return LABELS[tool.lower()]
    return str(statistic)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA,
                   help="careless_stats.csv written by the stats step")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--dataset", default=None,
                   help="draw only this dataset (default: each one)")
    p.add_argument("--prefix", default="careless",
                   help="stem of the written figures")
    return p.parse_args()


def shell_axis(labels) -> list[float]:
    """Shell centres in `1/d^2`, so two binnings can share an axis.

    careless picks its own bins and DIALS picks others; on a bin-index axis
    position 5 would mean a different resolution in each.
    """
    out = []
    for text in labels:
        span = metrics.d_range_from_label(text)
        out.append(1.0 / metrics.shell_midpoint(*span) ** 2 if span
                   else float("nan"))
    return out


def resolution_ticks(ax, labels) -> None:
    seen, ticks, names = set(), [], []
    for text in labels:
        span = metrics.d_range_from_label(text)
        if not span or text in seen:
            continue
        seen.add(text)
        d = metrics.shell_midpoint(*span)
        ticks.append(1.0 / d**2)
        names.append(f"{d:.2f}")
    if not ticks:
        return
    order = sorted(range(len(ticks)), key=lambda i: ticks[i])
    ticks = [ticks[i] for i in order]
    names = [names[i] for i in order]
    step = max(1, len(ticks) // 8)  # a dense binning would crowd the axis
    ax.set_xticks(ticks[::step])
    ax.set_xticklabels(names[::step], rotation=55, ha="right")
    ax.set_xlabel(r"resolution (\AA)")
    ax.grid(axis="x", visible=False)


# What a `group` value means, for the legend. careless splits cchalf,
# ccanom and rsplit by cross-validation repeat and ccpred by Train/Test.
GROUP_PREFIX = {"cchalf": "repeat", "ccanom": "repeat", "rsplit": "repeat"}


def series_of(panel: pl.DataFrame) -> tuple[str, list[tuple[str, pl.DataFrame]]]:
    """The splitting column, and the lines in one panel.

    The split is whichever column actually varies: several methods on one
    axis is the comparison worth drawing, then epochs, then the grouping
    careless itself made -- the cross-validation repeat, or Train/Test for
    `ccpred`. The column is returned because epochs are drawn on the
    sequential ramp and a category is not: a repeat is not an ordering and
    must not be coloured as though it were.
    """
    for column in ("method", "epoch", "group"):
        if column not in panel.columns:
            continue
        values = [v for v in panel[column].unique().to_list()
                  if v not in (None, "")]
        if len(values) > 1:
            values.sort(key=lambda v: (0, float(v)) if str(v).replace(
                ".", "", 1).lstrip("-").isdigit() else (1, str(v)))
            return column, [
                (str(v), panel.filter(pl.col(column) == v)) for v in values
            ]
    return "", [("", panel)]


def _one_point_per_bin(part: pl.DataFrame) -> pl.DataFrame:
    """Collapse whatever still varies within a bin to a mean and a spread.

    A tool that reports per cross-validation repeat gives several rows per
    resolution shell. Plotted raw and sorted by bin they connect as a
    sawtooth, which is an artefact of the line order and not a property of
    the measurement.
    """
    if "bin" not in part.columns or part["bin"].n_unique() == len(part):
        return part.with_columns(lo=pl.col("value"), hi=pl.col("value"))
    return (
        part.group_by("bin")
        .agg(
            pl.col("value").mean().alias("value"),
            pl.col("value").min().alias("lo"),
            pl.col("value").max().alias("hi"),
            pl.col("shell").first(),
        )
        .sort("bin")
    )


def draw_panel(ax, panel: pl.DataFrame, tool: str, statistic: str,
               legend: bool = True) -> bool:
    # `careless.completeness` reports an `overall` row beside its shells. A
    # number over every reflection has no position on a resolution axis, so
    # it is dropped here rather than left to `shell_axis`, which returns NaN
    # for it and so dropped it by accident.
    panel = panel.filter(pl.col("shell").str.contains("-"))
    if not len(panel):
        return False
    column, lines = series_of(panel)
    ramp = None
    if column == "epoch" and len(lines) > 1:
        _, ramp = figstyle().epoch_ramp(len(lines))
    prefix = GROUP_PREFIX.get(tool.lower(), "") if column == "group" else ""
    names = [f"{prefix} {n}".strip() for n, _ in lines]
    palette = figstyle().arm_colors(names) if ramp is None else {}

    drawn = False
    for i, (name, part) in enumerate(lines):
        part = part.sort("bin") if "bin" in part.columns else part
        # Where the split is `method` and careless ALSO split the tool's
        # output -- cchalf reports one value per cross-validation repeat --
        # a line through the raw rows visits each bin once per repeat and
        # comes out as a sawtooth that reads as noise in the data. Collapse
        # to one point per bin and show the repeats as the band they are.
        part = _one_point_per_bin(part)
        y = part["value"].to_list()
        if not y:
            continue
        x = shell_axis(part["shell"].to_list())
        colour = ramp[i] if ramp is not None else palette.get(names[i])
        if "lo" in part.columns and part["lo"].ne(part["hi"]).any():
            ax.fill_between(x, part["lo"], part["hi"], color=colour,
                            alpha=0.15, linewidth=0)
        ax.plot(x, y, lw=0.9, color=colour, label=names[i] or None)
        drawn = True
    if not drawn:
        return False
    resolution_ticks(ax, panel["shell"].to_list())
    ax.set_ylabel(label_for(tool, statistic))
    if legend and ramp is None and any(names):
        ax.legend(frameon=False)
    return True


def panels_of(df: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    keys = (
        df.select("tool", "statistic")
        .unique()
        .sort("tool", "statistic")
        .rows()
    )

    def rank(key):
        tool = key[0].lower()
        return (TOOL_ORDER.index(tool) if tool in TOOL_ORDER else len(TOOL_ORDER),
                key[0], key[1])

    out = []
    for tool, statistic in sorted(keys, key=rank):
        panel = df.filter(
            (pl.col("tool") == tool) & (pl.col("statistic") == statistic)
        )
        if len(panel):
            out.append((tool, statistic, panel))
    return out


def _panel_slug(tool: str, statistic: str) -> str:
    """`cchalf`, but `completeness_anomalous` where a tool has several.

    A tool whose statistic repeats its own name adds nothing -- `cchalf`
    reports `CChalf` -- so the statistic is appended only where it differs.
    """
    t = str(tool).strip().lower()
    st = re.sub(r"[^0-9a-z]+", "_", str(statistic).strip().lower()).strip("_")
    return t if not st or st.replace("_", "") == t.replace("_", "") else f"{t}_{st}"


def draw(df: pl.DataFrame, out_dir: Path, dataset: str, prefix: str) -> int:
    panels = panels_of(df)
    if not panels:
        logger.warning("nothing to draw for %s", dataset)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = str(dataset).replace(" ", "_")
    written = 0

    # one panel per statistic, each its own figure
    for tool, statistic, panel in panels:
        fig, ax = plt.subplots(
            figsize=figstyle().figsize("half", aspect=0.85),
            layout="constrained")
        if draw_panel(ax, panel, tool, statistic):
            # `careless.completeness` reports three statistics -- all,
            # anomalous and non-anomalous -- from one tool, so naming the
            # file after the tool alone wrote three panels to one path and
            # kept whichever was drawn last
            path = out_dir / f"{prefix}_{_panel_slug(tool, statistic)}_{slug}"
            figstyle().save(fig, path)
            logger.info("Saved %s", path)
            written += 1
        plt.close(fig)

    # and the grid, which is the reported figure
    n = len(panels)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=figstyle().figsize("full", aspect=0.42 * rows),
        layout="constrained", squeeze=False)
    flat = axes.ravel()
    handles = None
    for i, (tool, statistic, panel) in enumerate(panels):
        # the axis furniture is written once, not once per panel: every
        # panel here is against resolution, so the x label belongs on the
        # bottom row only, and one legend serves the whole grid. The tick
        # LABELS stay on every panel, because careless bins each tool
        # separately -- isigi defaults to 20 shells and the rest to 10 --
        # so position 5 is a different resolution in different panels.
        if draw_panel(flat[i], panel, tool, statistic, legend=False):
            if handles is None:
                handles = flat[i].get_legend_handles_labels()
        if i < n - cols:
            flat[i].set_xlabel("")
    for ax in flat[n:]:
        ax.set_visible(False)
    if handles and handles[0]:
        flat[0].legend(*handles, frameon=False)
    path = out_dir / f"{prefix}_stats_{slug}"
    figstyle().save(fig, path)
    logger.info("Saved %s (%d panels)", path, n)
    plt.close(fig)
    return written + 1


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"no {args.data}; the polychromatic `stats` step writes it "
            "(scripts/poly/run_pipeline.py --steps stats)"
        )
    figstyle().use()
    df = pl.read_csv(args.data, infer_schema_length=None)
    missing = {"tool", "statistic", "value", "shell"} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.data} has no {sorted(missing)} column")
    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))

    datasets = (
        [args.dataset] if args.dataset
        else sorted(df["dataset"].unique().to_list())
        if "dataset" in df.columns else [""]
    )
    total = 0
    for dataset in datasets:
        part = df.filter(pl.col("dataset") == dataset) if dataset else df
        total += draw(part, args.out_dir, dataset or "all", args.prefix)
    logger.info("%d figures", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
