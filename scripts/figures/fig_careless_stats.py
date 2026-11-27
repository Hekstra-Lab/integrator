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
    "r-work": r"$R_\mathrm{work}$",
    "r-free": r"$R_\mathrm{free}$",
    "spearman": r"$\rho_\mathrm{Spearman}$",
    "pearson": r"$r_\mathrm{Pearson}$",
}
# The order the panels are reported in; anything else follows, sorted.
TOOL_ORDER = ("cchalf", "ccanom", "ccpred", "rsplit", "isigi", "completeness", "phenix")


QUALIFIERS = {"spearman": "Spearman", "pearson": "Pearson"}

# Which direction is better, shown as an arrow on the y label. Correlations,
# I/sigma and completeness: higher; R factors and Rsplit: lower.
HIGHER_BETTER = {"cchalf", "ccanom", "ccpred", "isigi", "i/sigi", "completeness"}
LOWER_BETTER = {"rsplit", "r-work", "r-free"}


def better_arrow(tool: str, statistic: str) -> str:
    """` $\\uparrow$` where higher is better, ` $\\downarrow$` where lower is, else ''."""
    key = re.sub(r"\s*\(.*$", "", str(statistic)).split(",")[0].strip().lower().replace(" ", "")
    for k in (key, str(tool).lower()):
        if k in LOWER_BETTER:
            return r" $\downarrow$"
        if k in HIGHER_BETTER:
            return r" $\uparrow$"
    return ""


def label_parts(tool: str, statistic: str) -> tuple[str, str]:
    """(typeset name, qualifier) of `label_for`; the y axis takes the name alone."""
    full = label_for(tool, statistic)
    base = label_for(tool, re.sub(r"\s*\(\w+\)$", "", str(statistic)))
    if tool.lower() in LABELS and full.startswith(LABELS[tool.lower()]):
        base = LABELS[tool.lower()] if str(statistic).strip().lower().replace(" ", "") not in LABELS else base
    return base, full[len(base):].lstrip(", ")


def label_for(tool: str, statistic: str) -> str:
    """The typeset name, with what distinguishes two panels of one tool.

    `panels_of` appends the correlation, `CChalf (spearman)`, and
    careless.completeness names its three statistics `all`, `anomalous` and
    `non-anomalous`; both used to be dropped, so two CC-half panels and three
    completeness panels carried identical titles.
    """
    s = str(statistic).strip()
    qual = ""
    m = re.match(r"^(.*?)\s*\((\w+)\)$", s)
    if m:
        s, qual = m.group(1).strip(), m.group(2)
    key = s.lower().replace(" ", "")
    if key in LABELS:
        base = LABELS[key]
    elif tool.lower() in LABELS:
        base = LABELS[tool.lower()]
        if key and key.replace("_", "") != tool.lower().replace("_", ""):
            qual = s + (f", {qual}" if qual else "")
    else:
        base = s
    qual = QUALIFIERS.get(qual, qual)
    return f"{base}, {qual}" if qual else base


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA,
                   help="careless_stats.csv written by the stats step")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--method", default=None,
                   help="draw only these arms, comma separated; the default "
                        "draws every arm in the table, which is the "
                        "comparison. One arm gives the single-model figure.")
    p.add_argument("--correlation", default=None,
                   help="draw only this correlation (pearson, spearman); "
                        "the default draws a panel for each present")
    p.add_argument("--dataset", default=None,
                   help="draw only this dataset (default: each one)")
    p.add_argument("--template", default=None,
                   help="figTeX template to size for (article, paper, book, "
                        "slides, poster); default is figstyle's")
    p.add_argument("--prefix", default="careless",
                   help="stem of the written figures")
    p.add_argument("--extra-data", type=Path, nargs="*", default=[],
                   help="more careless_stats tables appended to --data (another dataset's)")
    p.add_argument("--seed", type=int, default=None,
                   help="draw only this careless seed (tables with a seed column)")
    p.add_argument("--facet", choices=("method", "dataset", "overlay"), default=None,
                   help="method: per statistic, one panel per arm and one line per dataset; "
                        "dataset: per statistic, one panel per dataset and one line per arm, "
                        "on a shared y range; overlay: per statistic, every arm in one panel (colour = arm, "
                        "line style = dataset) beside the change from the first dataset to the second "
                        "(default: one figure per dataset, one line per arm)")
    p.add_argument("--tool", default="cchalf",
                   help="with --facet: the careless tools to draw, comma separated")
    p.add_argument("--refine-shells", type=Path, nargs="*", default=[],
                   help="collect_downstream.py downstream_refine_shells.csv tables: phenix's final "
                        "per-shell R-work and R-free, added as tool `phenix` (with --tool phenix)")
    p.add_argument("--dataset-label", action="append", default=[],
                   help="with --facet: DATASET=LABEL for the legend, repeatable")
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


def resolution_ticks(ax, labels, P) -> None:
    """One tick per DISPLAYED resolution, not per shell label.

    The arms do not share a binning: careless rounds each run's own bin
    edges, so `28.96 - 3.76` and `28.96 - 3.77` are the same shell written
    two ways. Deduplicating on the label text kept both, which put two
    ticks a hair apart and drew their identical names on top of each
    other -- the "bold" tick labels in the CC-half and R-split panels were
    two labels, not one.

    Deduplicating on the formatted midpoint instead collapses them, which
    is also the honest thing to show: the axis can only claim the
    precision it prints.
    """
    ds = []
    for text in labels:
        span = metrics.d_range_from_label(text)
        if not span:
            continue
        ds.append(metrics.shell_midpoint(*span))
    ds = _merge_close(sorted(ds, reverse=True))
    if not ds:
        return
    figstyle().shell_axis(ax, P, [1.0 / d**2 for d in ds], ds)
    ax.set_xlabel(r"resolution (\AA)")
    ax.grid(axis="x", visible=False)


def _merge_close(ds: list[float], tol: float = 0.02) -> list[float]:
    """Collapse shells the arms binned a hair apart into one tick.

    Deduplicating on the PRINTED midpoint was not enough: the arms' first
    shell comes back as 5.29 and 5.27 A, 0.38% apart, which are two ticks
    with two labels drawn almost on top of each other -- the doubled, bold
    "5.29" in the CC-half and R-split panels. They are the same shell;
    careless rounds each run's own bin edges and the runs differ in the
    second decimal.

    Two midpoints within `tol` of each other (relative) are averaged, so
    one tick is drawn where there is one shell. The next real gap in this
    data is 5.27 to 3.32, a factor of 1.6, so nothing distinct is merged.
    """
    out: list[list[float]] = []
    for d in ds:
        if out and abs(d - out[-1][-1]) <= tol * max(d, out[-1][-1]):
            out[-1].append(d)
        else:
            out.append([d])
    return [sum(g) / len(g) for g in out]


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


# the arm that is the reference integrator rather than a model arm. It is
# drawn as the reference -- vermilion, dashed, on top -- as in `fig_ccsym`,
# instead of taking the next categorical colour, which made it green and
# indistinguishable from a model arm.
REFERENCE = "laue"


def short(name: str) -> str:
    """The arm without its dataset prefix.

    `tidy_poly` writes `method` as the run directory name, which carries
    the dataset -- `hewl1118_svae_bg_rb` -- and the dataset is already a
    column of its own. Ten characters per entry is the difference between
    a legend that fits the text block and one cut off in the output.
    """
    return (str(name).replace("hewl1118_", "").replace("hewl9b7c_", "")
            .replace("pdz2_", ""))


def draw_panel(ax, panel: pl.DataFrame, tool: str, statistic: str, P,
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
    names = [short(f"{prefix} {n}".strip()) for n, _ in lines]
    from arms import colours, label as arm_label

    palette = colours(names) if ramp is None else {}

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
        is_ref = ramp is None and str(names[i]).strip() in (REFERENCE, "laue_integrate")
        colour = (figstyle().reference_style()["color"] if is_ref else
                  ramp[i] if ramp is not None else palette.get(names[i]))
        if "lo" in part.columns and part["lo"].ne(part["hi"]).any():
            ax.fill_between(x, part["lo"], part["hi"], color=colour,
                            alpha=0.15, linewidth=0)
        shown = arm_label(names[i]) if names[i] else None
        if is_ref:
            figstyle().parts.reference(ax, x, y, label=shown)
        else:
            ax.plot(x, y, lw=0.9, color=colour, label=shown)
        drawn = True
    if not drawn:
        return False
    resolution_ticks(ax, panel["shell"].to_list(), P)
    # the name alone: the qualifier (Pearson, Spearman, anomalous, ...) is in
    # the panel title, and a full label on every y axis overprinted the
    # neighbouring panel in the two-column grid
    ax.set_ylabel(label_parts(tool, statistic)[0] + better_arrow(tool, statistic))
    if legend and ramp is None and any(names):
        # boxed, like the grid's: the ink theme draws a legend on the page
        # rather than on a panel, and an unframed one over the data reads
        # as an annotation of it
        ax.legend(frameon=True, loc="best")
    return True


def panels_of(df: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    """One panel per (tool, statistic, correlation) present.

    `correlation` is part of the key, not a line style: Pearson and
    Spearman CChalf are different numbers on the same axis label, and
    overlaying them would read as six arms where there are three. A table
    written before the column existed has none, and then this behaves
    exactly as it did.
    """
    has_corr = "correlation" in df.columns
    if has_corr:
        # an empty correlation is written as an empty field and read back as
        # null, so the tools that take no `-m` -- rsplit, isigi,
        # completeness -- matched nothing and vanished from the figure
        df = df.with_columns(pl.col("correlation").fill_null(""))
    cols = ["tool", "statistic"] + (["correlation"] if has_corr else [])
    keys = df.select(cols).unique().sort(cols).rows()

    def rank(key):
        tool = key[0].lower()
        return (TOOL_ORDER.index(tool) if tool in TOOL_ORDER else len(TOOL_ORDER),
                key[0], key[1], key[2] if len(key) > 2 else "")

    out = []
    for key in sorted(keys, key=rank):
        tool, statistic = key[0], key[1]
        corr = key[2] if has_corr else ""
        where = (pl.col("tool") == tool) & (pl.col("statistic") == statistic)
        if has_corr:
            where = where & (pl.col("correlation") == corr)
        panel = df.filter(where)
        if len(panel):
            # the correlation rides on the statistic so every caller --
            # label, slug, title -- carries it without further plumbing
            name = f"{statistic} ({corr})" if corr else statistic
            out.append((tool, name, panel))
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
        P = figstyle().panels("1", column=True)
        ax, fig = P.axs, None
        if draw_panel(ax, panel, tool, statistic, P, legend=False):
            # the legend beside the panel, on every template: inside it,
            # `loc="best"` put the box over the lower half of the CC-half
            # dip band in the book render and over the dip itself on a
            # slide, and the dip is what that panel is there to show
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                P.side_legend(handles, labels, frameon=True)
            # `careless.completeness` reports three statistics -- all,
            # anomalous and non-anomalous -- from one tool, so naming the
            # file after the tool alone wrote three panels to one path and
            # kept whichever was drawn last
            P.finish()
            path = out_dir / f"{prefix}_{_panel_slug(tool, statistic)}_{slug}"
            figstyle().save(P.fig, path)
            logger.info("Saved %s", path)
            written += 1
        plt.close(P.fig)

    # and the grid, which is the reported figure
    n = len(panels)
    # three columns once there are more than six panels, and an explicit
    # height: at two columns figTeX's default height gave each of six rows a
    # data area too short to read (11 panels on mos010, 2026-11-25)
    cols = 1 if n == 1 else (2 if n <= 6 else 3)
    rows = (n + 1 + cols - 1) // cols  # +1: the legend cell
    # Each panel is a DIFFERENT statistic, so neither axis is shared:
    # `panels` shares both by default once there is more than one row, and
    # that put CC-half, R-split and completeness on I/sigma's 0-110 scale,
    # where all three are flat lines along the bottom. careless also bins
    # each tool separately -- isigi defaults to 20 shells, the rest to
    # 10 -- so position 5 is a different resolution in different panels and
    # x cannot be shared either.
    from figtex import sizing as _sizing

    tall = 0.9 * _sizing.TEMPLATES[figstyle().active()].text_height
    P = figstyle().panels(f"{rows}x{cols}", shared=False,
                          height=min(tall, rows * 0.8 * _sizing.TEMPLATES[figstyle().active()].text_width / cols))
    import numpy as np

    flat = list(np.ravel(np.asarray(P.axs, dtype=object)))
    handles = None
    for i, (tool, statistic, panel) in enumerate(panels):
        # the axis furniture is written once, not once per panel: every
        # panel here is against resolution, so the x label belongs on the
        # bottom row only, and one legend serves the whole grid. The tick
        # LABELS stay on every panel, because careless bins each tool
        # separately -- isigi defaults to 20 shells and the rest to 10 --
        # so position 5 is a different resolution in different panels.
        if draw_panel(flat[i], panel, tool, statistic, P, legend=False):
            if handles is None:
                handles = flat[i].get_legend_handles_labels()
            P.head(flat[i], i, label_for(tool, statistic))
        if i < n - cols:
            flat[i].set_xlabel("")
    # The legend goes in the leftover cell rather than on top of panel 1.
    # With five statistics in a 3x2 grid the sixth cell is empty, and a
    # legend drawn there is both off the data and larger than one squeezed
    # into a panel corner. The cell keeps its axes object so the grid
    # stays rectangular; only its frame and ticks come off.
    spare = flat[n:]
    if handles and handles[0] and spare:
        cell = spare[0]
        # no ticks at all, not only hidden ones: figTeX's clip check
        # measures the default 0-1 tick labels of an axes that is off
        cell.set_xticks([])
        cell.set_yticks([])
        cell.set_axis_off()
        cell.legend(*handles, loc="center", frameon=True,
                    ncols=1 if len(handles[1]) <= 6 else 2)
        for ax in spare[1:]:
            ax.set_visible(False)
    else:
        for ax in spare:
            ax.set_visible(False)
        if handles and handles[0]:
            figstyle().top_legend(P, *handles)
    P.finish()
    path = out_dir / f"{prefix}_stats_{slug}"
    figstyle().save(P.fig, path)
    logger.info("Saved %s (%d panels)", path, n)
    plt.close(P.fig)
    return written + 1


def draw_by_method(df: pl.DataFrame, args) -> int:
    """Per (tool, statistic, correlation): one panel per arm, one line per dataset.

    For one model across datasets that differ in one simulation parameter
    (mos010 and mos025, mosaic spread 0.10 and 0.25). Each line is the mean
    over careless's cross-validation repeats per shell, with their range as
    the band; panels share the y range so arms compare across panels.
    """
    import numpy as np

    from arms import label as arm_label

    tools = [t.strip().lower() for t in args.tool.split(",") if t.strip()]
    df = df.filter(pl.col("tool").str.to_lowercase().is_in(tools))
    names = dict(x.split("=", 1) for x in args.dataset_label)
    datasets = [d for d in df["dataset"].unique(maintain_order=True).to_list()]
    labels = [names.get(d, d) for d in datasets]
    colour = figstyle().arm_colors(labels)
    methods = ([m.strip() for m in args.method.split(",") if m.strip()] if args.method
               else sorted(df["method"].unique().to_list()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for tool, statistic, panel in panels_of(df):
        panel = panel.filter(pl.col("shell").str.contains("-"))
        if tool.lower() == "ccpred":  # Train and Test are different quantities: the test set
            panel = panel.filter(pl.col("group").cast(pl.Utf8) == "Test")
            statistic = f"{statistic}, test"
        ms = [m for m in methods if panel.filter(method=m).height]
        if not ms:
            continue
        cols = min(3, len(ms))
        rows = (len(ms) + cols - 1) // cols
        P = figstyle().panels(f"{rows}x{cols}", shared=False)
        flat = list(np.ravel(np.asarray(P.axs, dtype=object)))
        lo, hi = np.inf, -np.inf
        for i, m in enumerate(ms):
            ax = flat[i]
            for d, lab in zip(datasets, labels):
                part = panel.filter((pl.col("method") == m) & (pl.col("dataset") == d))
                if not part.height:
                    continue
                part = _one_point_per_bin(part.sort("bin"))
                x = shell_axis(part["shell"].to_list())
                ax.fill_between(x, part["lo"], part["hi"], color=colour[lab], alpha=0.15, linewidth=0)
                ax.plot(x, part["value"], lw=0.9, marker="o", ms=2, color=colour[lab], label=lab)
                lo, hi = min(lo, float(part["lo"].min())), max(hi, float(part["hi"].max()))
            resolution_ticks(ax, panel.filter(method=m)["shell"].to_list(), P)
            ax.set_ylabel(label_parts(tool, statistic)[0] + better_arrow(tool, statistic) if i % cols == 0 else "")
            P.head(ax, i, arm_label(short(m)))
            if i < len(ms) - cols:
                ax.set_xlabel("")
        pad = 0.04 * (hi - lo)
        for ax in flat[:len(ms)]:
            ax.set_ylim(lo - pad, hi + pad)
        for ax in flat[len(ms):]:
            ax.set_visible(False)
        handles = flat[0].get_legend_handles_labels()
        if handles[0]:
            figstyle().top_legend(P, *handles)
        P.finish()
        path = args.out_dir / f"{args.prefix}_by_method_{_panel_slug(tool, statistic)}"
        figstyle().save(P.fig, path)
        logger.info("Saved %s (%d arms)", path, len(ms))
        plt.close(P.fig)
        written += 1
    return written


def draw_by_dataset(df: pl.DataFrame, args) -> int:
    """Per (tool, statistic, correlation): one panel per dataset, one line per arm.

    The arms keep their colours from `arms.py` and laue.integrate is drawn as
    the reference; panels share the y range so the datasets compare.
    """
    import numpy as np

    from arms import colours, label as arm_label

    tools = [t.strip().lower() for t in args.tool.split(",") if t.strip()]
    df = df.filter(pl.col("tool").str.to_lowercase().is_in(tools))
    names = dict(x.split("=", 1) for x in args.dataset_label)
    datasets = df["dataset"].unique(maintain_order=True).to_list()
    methods = ([m.strip() for m in args.method.split(",") if m.strip()] if args.method
               else sorted(df["method"].unique().to_list()))
    palette = colours([short(m) for m in methods])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for tool, statistic, panel in panels_of(df):
        panel = panel.filter(pl.col("shell").str.contains("-"))
        if tool.lower() == "ccpred":  # Train and Test are different quantities: the test set
            panel = panel.filter(pl.col("group").cast(pl.Utf8) == "Test")
            statistic = f"{statistic}, test"
        ds = [d for d in datasets if panel.filter(dataset=d).height]
        if not ds:
            continue
        P = figstyle().panels(f"1x{len(ds)}", shared=False)
        flat = list(np.ravel(np.asarray(P.axs, dtype=object)))
        lo, hi = np.inf, -np.inf
        for i, d in enumerate(ds):
            ax = flat[i]
            for m in methods:
                part = panel.filter((pl.col("method") == m) & (pl.col("dataset") == d))
                if not part.height:
                    continue
                part = _one_point_per_bin(part.sort("bin"))
                x = shell_axis(part["shell"].to_list())
                lab = arm_label(short(m))
                if short(m) in (REFERENCE, "laue_integrate"):
                    figstyle().parts.reference(ax, x, part["value"].to_list(), label=lab)
                else:
                    c = palette.get(short(m))
                    ax.fill_between(x, part["lo"], part["hi"], color=c, alpha=0.15, linewidth=0)
                    ax.plot(x, part["value"], lw=0.9, marker="o", ms=2, color=c, label=lab)
                lo, hi = min(lo, float(part["lo"].min())), max(hi, float(part["hi"].max()))
            resolution_ticks(ax, panel.filter(dataset=d)["shell"].to_list(), P)
            ax.set_ylabel(label_parts(tool, statistic)[0] + better_arrow(tool, statistic) if i == 0 else "")
            P.head(ax, i, names.get(d, d))
        pad = 0.04 * (hi - lo)
        for ax in flat:
            ax.set_ylim(lo - pad, hi + pad)
        handles = flat[0].get_legend_handles_labels()
        if handles[0]:
            figstyle().top_legend(P, *handles)
        P.finish()
        path = args.out_dir / f"{args.prefix}_by_dataset_{_panel_slug(tool, statistic)}"
        figstyle().save(P.fig, path)
        logger.info("Saved %s (%d datasets)", path, len(ds))
        plt.close(P.fig)
        written += 1
    return written


def draw_overlay(df: pl.DataFrame, args) -> int:
    """Per (tool, statistic, correlation): all arms and both datasets on one axis, and the change.

    Left, colour is the arm (laue.integrate as the reference colour) and the line
    style the dataset, solid for the first and dashed for the second. Right, the
    second dataset's value minus the first's, per arm. careless and phenix bin each
    run separately, so the first dataset's curve is interpolated linearly in 1/d^2
    onto the second's shell centres, inside the range both cover only.
    """
    import numpy as np

    from arms import colours, label as arm_label

    tools = [t.strip().lower() for t in args.tool.split(",") if t.strip()]
    df = df.filter(pl.col("tool").str.to_lowercase().is_in(tools))
    names = dict(x.split("=", 1) for x in args.dataset_label)
    datasets = df["dataset"].unique(maintain_order=True).to_list()
    if len(datasets) != 2:
        raise SystemExit(f"--facet overlay needs exactly two datasets, got {datasets}")
    d0, d1 = datasets
    methods = ([m.strip() for m in args.method.split(",") if m.strip()] if args.method
               else sorted(df["method"].unique().to_list()))
    palette = colours([short(m) for m in methods])
    ref_colour = figstyle().reference_style()["color"]
    styles = {d0: "-", d1: "--"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for tool, statistic, panel in panels_of(df):
        panel = panel.filter(pl.col("shell").str.contains("-"))
        if tool.lower() == "ccpred":
            panel = panel.filter(pl.col("group").cast(pl.Utf8) == "Test")
            statistic = f"{statistic}, test"
        ms = [m for m in methods if panel.filter(method=m).height]
        if not ms:
            continue
        P = figstyle().panels("1x2", shared=False)
        ax, dx = list(np.ravel(np.asarray(P.axs, dtype=object)))[:2]
        curves = {}
        for m in ms:
            c = ref_colour if short(m) in (REFERENCE, "laue_integrate") else palette.get(short(m))
            for d in (d0, d1):
                part = panel.filter((pl.col("method") == m) & (pl.col("dataset") == d))
                if not part.height:
                    continue
                part = _one_point_per_bin(part.sort("bin"))
                x = np.asarray(shell_axis(part["shell"].to_list()), float)
                y = part["value"].to_numpy()
                curves[m, d] = (x, y)
                ax.plot(x, y, ls=styles[d], lw=1.0 if d == d0 else 1.2, color=c,
                        label=f"{arm_label(short(m))}, {names.get(d, d)}")
            if (m, d0) in curves and (m, d1) in curves:
                (x0, y0), (x1, y1) = curves[m, d0], curves[m, d1]
                o = np.argsort(x0)
                inside = (x1 >= x0.min()) & (x1 <= x0.max())
                dx.plot(x1[inside], y1[inside] - np.interp(x1[inside], x0[o], y0[o]), lw=1.0, marker="o", ms=2,
                        color=c, label=arm_label(short(m)))
        dx.axhline(0, color=figstyle().roles()["frame"], lw=0.6)
        for a in (ax, dx):
            resolution_ticks(a, panel["shell"].to_list(), P)
        base = label_parts(tool, statistic)[0]
        ax.set_ylabel(base + better_arrow(tool, statistic))
        dx.set_ylabel(rf"{names.get(d1, d1)} $-$ {names.get(d0, d0)}")
        P.head(ax, 0, label_for(tool, statistic))
        P.head(dx, 1, "change")
        handles = dx.get_legend_handles_labels()
        if handles[0]:
            from matplotlib.lines import Line2D

            h, l = list(handles[0]), list(handles[1])
            h += [Line2D([], [], color="0.3", ls=styles[d]) for d in (d0, d1)]
            l += [names.get(d, d) for d in (d0, d1)]
            figstyle().top_legend(P, h, l)
        P.finish()
        path = args.out_dir / f"{args.prefix}_overlay_{_panel_slug(tool, statistic)}"
        figstyle().save(P.fig, path)
        logger.info("Saved %s", path)
        plt.close(P.fig)
        written += 1
    return written


def main() -> int:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"no {args.data}; the polychromatic `stats` step writes it "
            "(scripts/poly/run_pipeline.py --steps stats)"
        )
    if args.template:
        figstyle().use(args.template)
    else:
        figstyle().use()
    df = pl.read_csv(args.data, infer_schema_length=None)
    missing = {"tool", "statistic", "value", "shell"} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.data} has no {sorted(missing)} column")
    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
    for extra in args.extra_data:
        more = pl.read_csv(extra, infer_schema_length=None).with_columns(
            pl.col("value").cast(pl.Float64, strict=False))
        df = pl.concat([df, more], how="diagonal_relaxed")
    for rs in args.refine_shells:
        sh = pl.read_csv(rs)
        sh = sh.with_columns(
            shell=pl.format("{} - {}", pl.col("d_max").round(2), pl.col("d_min").round(2)),
            bin=pl.col("shell").cast(pl.Int64))
        long = (sh.unpivot(index=["method", "dataset", "seed", "shell", "bin"],
                           on=["r_work", "r_free"], variable_name="statistic", value_name="value")
                .with_columns(tool=pl.lit("phenix"), correlation=pl.lit(""), group=pl.lit(None, pl.Utf8),
                              statistic=pl.col("statistic").replace({"r_work": "R-work", "r_free": "R-free"})))
        df = pl.concat([df, long], how="diagonal_relaxed")
    if args.seed is not None and "seed" in df.columns:
        df = df.filter(pl.col("seed") == args.seed)

    if args.method:
        want = [m.strip() for m in args.method.split(",") if m.strip()]
        have = df["method"].unique().to_list()
        unknown = [m for m in want if m not in have]
        if unknown:
            raise SystemExit(
                f"no rows for {unknown}; the table has {sorted(have)}"
            )
        df = df.filter(pl.col("method").is_in(want))
    if args.correlation and "correlation" in df.columns:
        have = [c for c in df["correlation"].unique().to_list() if c]
        if args.correlation not in have:
            raise SystemExit(
                f"no rows with correlation {args.correlation!r}; the table "
                f"has {sorted(have) or 'none -- it predates the column'}"
            )
        # the tools with no `-m` carry "" and are kept, since they are not
        # a correlation and dropping them would silently shorten the figure
        df = df.filter(pl.col("correlation").fill_null("")
                       .is_in([args.correlation, ""]))

    if args.facet in ("method", "dataset", "overlay"):
        n = {"method": draw_by_method, "dataset": draw_by_dataset, "overlay": draw_overlay}[args.facet](df, args)
        logger.info("%d figures", n)
        return 0

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
