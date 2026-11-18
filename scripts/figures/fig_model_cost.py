"""What the svae and hierarchical families cost: parameters, and time per epoch.

Two panels, and they are not equally trustworthy, which the figure says
rather than leaves to a caption.

**Parameters** are counted from the last checkpoint's `state_dict`, grouped by
top-level module. They do not depend on where the run happened, so all three
datasets are drawn.

**Seconds per epoch** is the median gap between checkpoint timestamps. That is
only a property of the model when the allocation is the same, and on this
cluster it usually is not: data loading scales with `-c` and the partitions
mix A100, H100 and H200. So a pair is drawn solid only when both arms record
the same node and CPU count; an unmatched or unrecorded pair is drawn hollow
and labelled, because the PDZ2 pair read as "svae 38% slower" when it was
8 CPUs against 16, two months apart, pointing opposite to the 1118 pair.

Reads `docs/figures/data/cmp_*.csv` from `scripts/figures/compare_models.py`,
pulled by `bash scripts/figures/pull_figdata.sh models`. Long format, one row
per (method, dataset, quantity).

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_model_cost.py
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


def figstyle():
    """figTeX, imported at the call: it is not a dependency of this project."""
    import figstyle as _figstyle

    return _figstyle


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "cmp"
# left to right, smallest dataset first; a dataset absent from the data is
# dropped rather than drawn empty
DATASETS = ("hewl_9b7c", "hewl_1118", "pdz2_1116", "pdz2_1116_matched")
LABELS = {"hewl_9b7c": "9b7c", "hewl_1118": "1118", "pdz2_1116": "PDZ2",
          "pdz2_1116_matched": "PDZ2\n(matched)"}
FAMILY = {"svae": "svae", "hierarchical": "hierarchical"}
# the two families' colours, fixed here so they do not move between renders
FAMILY_SLOT = {"hierarchical": 0, "svae": 3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--prefix", default="model_cost")
    return p.parse_args()


def load(data_dir: Path) -> pl.DataFrame:
    """Every `cmp_*.csv`, with `value` read as text and cast per quantity.

    The column holds parameter counts, seconds, node names and CPU counts, so
    a single numeric schema cannot describe it and polars infers different
    types per file. Read as text, cast where a number is wanted.
    """
    frames = []
    for path in sorted(data_dir.glob("cmp_*.csv")):
        df = pl.read_csv(path, infer_schema_length=0)  # all columns as text
        df = df.with_columns(pl.col("value").cast(pl.Utf8))
        if path.stem == "cmp_timing":
            # The matched-allocation run is PDZ2 measured again, not a
            # different dataset, so it carries the same `dataset` value and
            # would overwrite the unmatched rows keyed on (dataset, family).
            # Given its own key it is drawn beside them instead, which is the
            # point: the two disagree and the reader should see both.
            df = df.with_columns(
                (pl.col("dataset") + "_matched").alias("dataset"))
        frames.append(df)
    if not frames:
        raise SystemExit(
            f"no cmp_*.csv in {data_dir}. They come from "
            "scripts/figures/compare_models.py, pulled by "
            "`bash scripts/figures/pull_figdata.sh models`"
        )
    return pl.concat(frames, how="diagonal")


def pick(df: pl.DataFrame, quantity: str) -> dict[tuple[str, str], str]:
    """{(dataset, integrator): value} for one quantity."""
    sub = df.filter(pl.col("quantity") == quantity)
    return {(r["dataset"], r["integrator"]): r["value"]
            for r in sub.iter_rows(named=True)}


def matched(df: pl.DataFrame, dataset: str) -> bool:
    """True when both arms of a dataset ran on one node with one CPU count.

    Unrecorded counts as unmatched. The claim being made is about the model,
    and without the allocation there is nothing to support it.
    """
    sub = df.filter((pl.col("dataset") == dataset)
                    & pl.col("quantity").is_in(["node", "cpus"]))
    if sub.height < 4:  # two quantities for each of two arms
        return False
    for quantity in ("node", "cpus"):
        vals = (sub.filter(pl.col("quantity") == quantity)["value"]
                .unique().to_list())
        if len(vals) != 1 or not vals[0]:
            return False
    return True


def grouped_bars(ax, datasets, values, families, colours, hollow=()):
    """One group per dataset, one bar per family."""
    x = np.arange(len(datasets))
    width = 0.36
    for i, fam in enumerate(families):
        offs = x + (i - 0.5) * width
        heights = [values.get((d, fam), np.nan) for d in datasets]
        for xi, h, d in zip(offs, heights, datasets, strict=True):
            if h != h:  # NaN
                continue
            if d in hollow:
                ax.bar(xi, h, width, facecolor="none", edgecolor=colours[fam],
                       lw=1.1, hatch="///", zorder=2)
            else:
                ax.bar(xi, h, width, color=colours[fam], zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(d, d) for d in datasets])


def main() -> None:
    args = parse_args()
    df = load(args.data)
    figstyle().use()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cat = figstyle().categorical(8)
    colours = {f: cat[FAMILY_SLOT[f] % len(cat)] for f in FAMILY}

    params = {k: float(v) for k, v in pick(df, "n_params_total").items()}
    rates = {k: float(v) for k, v in pick(df, "seconds_per_epoch").items()}

    # the matched run is the same two models measured again, so it adds a
    # third identical pair of parameter bars and belongs only on the right
    present = [d for d in DATASETS
               if any(k[0] == d for k in params) and d is not None
               and not d.endswith("_matched")]
    # the matched-allocation run is a separate measurement of PDZ2, drawn
    # beside it rather than replacing the unmatched one
    timing_sets, hollow = [], set()
    for d in [x for x in DATASETS if any(k[0] == x for k in rates)]:
        if any(k[0] == d for k in rates):
            timing_sets.append(d)
            if not matched(df, d):
                hollow.add(d)

    fig, axes = plt.subplots(
        1, 2, figsize=figstyle().figsize("full", aspect=0.46),
        # figTeX typesets through LuaLaTeX at a fixed box, so a label that
        # does not fit is clipped rather than growing the figure: the
        # two-line `PDZ2 (matched)` tick was being cut off the bottom.
        layout="constrained")

    grouped_bars(axes[0], present, params, list(FAMILY), colours)
    axes[0].set_ylabel("parameters")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    grouped_bars(axes[1], timing_sets, rates, list(FAMILY), colours,
                 hollow=hollow)
    axes[1].set_ylabel("seconds per epoch")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[f]) for f in FAMILY]
    labels = [FAMILY[f] for f in FAMILY]
    if hollow:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none",
                                     edgecolor=figstyle().roles().get(
                                         "text", "0.3"), hatch="///"))
        labels.append("allocation not matched")
    # across the top rather than `loc="best"`, which put the legend on the
    # 1118 bars; the headroom is made for it instead of overlapping data
    top = max(rates.values()) if rates else 1.0
    axes[1].set_ylim(0, top * 1.55)
    # two columns, not three: at three the legend was wider than its axes
    # and ran across into the parameters panel, over its y-axis label
    axes[1].legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   handlelength=1.2, columnspacing=0.9, borderaxespad=0.2,
                   fontsize="small")

    path = args.out_dir / f"{args.prefix}"
    figstyle().save(fig, path)
    plt.close(fig)
    logger.info("Saved %s", path)

    for d in present:
        p_s, p_h = params.get((d, "svae")), params.get((d, "hierarchical"))
        if p_s and p_h:
            logger.info("  %-12s svae is %.0f%% of hierarchical (%s vs %s)",
                        LABELS.get(d, d), 100 * p_s / p_h,
                        f"{int(p_s):,}", f"{int(p_h):,}")
    for d in timing_sets:
        r_s, r_h = rates.get((d, "svae")), rates.get((d, "hierarchical"))
        if r_s and r_h:
            logger.info("  %-12s %s: svae %.1f s/epoch, hierarchical %.1f",
                        LABELS.get(d, d),
                        "matched" if d not in hollow else "UNMATCHED",
                        r_s, r_h)


if __name__ == "__main__":
    main()
