import argparse
import logging
import re
from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import yaml
from cycler import cycler
from matplotlib.lines import Line2D

# logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Dark2 qualitative palette for every plot and the per-run color source.
_DARK2 = plt.get_cmap("Dark2").colors
plt.rcParams["axes.prop_cycle"] = cycler(color=list(_DARK2))


def _assign_colors(keys) -> dict:
    """Map each key to a fixed Dark2 color in the given order.

    Build this once (from the plot-config run order) and share it across every
    figure so a model keeps the same color in the peak and refinement plots.
    """
    ordered = list(dict.fromkeys(keys))  # dedupe, preserve order
    return {k: _DARK2[i % len(_DARK2)] for i, k in enumerate(ordered)}


def _group_key_col(frame: pl.DataFrame) -> str:
    """Column that identifies a run for coloring: `label` if present, else `run_id`."""
    if "label" in frame.columns and frame["label"].drop_nulls().len() > 0:
        return "label"
    return "run_id"


def load_config(resource: str | Path) -> dict:
    if isinstance(resource, str):
        resource = Path(resource)

    with as_file(resource) as p:
        with open(Path(p), encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    return raw


@dataclass
class ReferenceData:
    refl: Path | None = None
    merge: Path | None = None
    refinement: Path | None = None
    peaks: Path | None = None


@dataclass
class PlotConfig:
    runs: dict[str, dict]  # name -> {path, label}
    reference: ReferenceData | None = None


def parse_plot_cfg(raw: dict) -> PlotConfig:
    """Parse a raw plot-config dict into a typed PlotConfig.

    Expected schema:
        runs:
          run1: {path: ..., label: ...}
          run2: {path: ..., label: ...}
        reference_data:        # optional
          refl: ...
          merge: ...
          refinement: ...
          peaks: ...
    """
    ref = raw.get("reference_data")
    return PlotConfig(
        runs=raw.get("runs", {}),
        reference=ReferenceData(**{k: Path(v) for k, v in ref.items()})
        if ref
        else None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot anomalous peak heights")

    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="Path(s) to --run-dir containing a run_paths.yaml file",
    )
    parser.add_argument(
        "--plot-cfg",
        type=str,
        required=False,
        help="A plot config.yaml file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=False,
        help="directory to write outputs to",
    )
    return parser.parse_args()


def _get_peak_lf(
    peak_csvs,
    labels: dict[str, str] | None = None,
) -> pl.LazyFrame | None:
    """Build a LazyFrame of peak heights, or None when nothing was found.

    A run whose post-processing has not produced `peaks.csv` is skipped with
    a warning; `pl.scan_csv([])` would otherwise defer an opaque "empty
    input: paths: []" until collection.
    """
    lfs = []
    for run_dir, files in peak_csvs:
        if not files:
            logger.warning("no peaks.csv under %s: skipping", run_dir)
            continue
        lf = pl.scan_csv(
            files,
            include_file_paths="filenames",
            schema_overrides={
                "seqid": pl.Int64,
                "epoch": pl.Int64,
                "peakz": pl.Float32,
            },
        )
        lf = _add_run_epoch_cols(lf)
        if labels is not None:
            lf = lf.with_columns(
                pl.lit(labels.get(str(run_dir))).alias("label")
            )
        lfs.append(lf)
    if not lfs:
        return None
    return pl.concat(lfs, how="diagonal")


def _get_reference_lf(
    reference: ReferenceData,
) -> pl.LazyFrame | None:
    """Build a LazyFrame for the reference peaks CSV, labeled 'reference'.

    The reference has no per-epoch structure, so `epoch`/`run_id` are left
    null and the reference is drawn as a flat line across the epoch range.
    """
    if reference is None or reference.peaks is None:
        return None

    return pl.scan_csv(
        str(reference.peaks),
        include_file_paths="filenames",
        schema_overrides={
            "seqid": pl.Int64,
            "peakz": pl.Float32,
        },
    ).with_columns(
        pl.lit("reference").alias("label"),
        pl.lit("reference").alias("run_id"),
        pl.lit(None, dtype=pl.Int32).alias("epoch"),
    )


def _get_peak_csv(run_dir):
    _yaml = list(run_dir.glob("run_paths.yaml"))[0]
    paths = load_config(_yaml)
    pred_dir = Path(paths["predictions_dir"])
    paths = [p.as_posix() for p in list(pred_dir.glob("**/peaks.csv"))]
    return paths


def _get_peak_csvs(
    run_dirs: list[Path],
) -> list[tuple[Path, list[str]]]:
    return [(r, _get_peak_csv(r)) for r in run_dirs]


def _add_run_epoch_cols(
    lf: pl.LazyFrame,
    path_col: str = "filenames",
    epoch_pattern: str = r"/epoch_(\d+)/",
) -> pl.LazyFrame:
    """Add `run_id` and `epoch` columns extracted from a path column."""
    return lf.with_columns(
        pl.col(path_col)
        .str.extract(r"/run-[^/]+-([^/]+)/", 1)
        .alias("run_id"),
        pl.col(path_col)
        .str.extract(epoch_pattern, 1)
        .cast(pl.Int32)
        .alias("epoch"),
    )


# Matches a phenix.refine log line: "Final R-work = 0.1570, R-free = 0.1738".
_FINAL_R_RE = re.compile(
    r"Final R-work\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*R-free\s*=\s*([0-9]*\.?[0-9]+)"
)
# Fallback for a stats file: "Rwork:  0.1570" / "Rfree:  0.1738".
_STATS_RWORK_RE = re.compile(r"Rwork\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_STATS_RFREE_RE = re.compile(r"Rfree\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_refine_log(path) -> tuple[float | None, float | None]:
    """Extract (r_work, r_free) from a phenix refine log or stats file.

    Returns (None, None) when the file is missing or has no parseable
    R-factors, so an epoch whose refinement failed is skipped rather than
    aborting the whole plot.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return None, None

    matches = _FINAL_R_RE.findall(text)
    if matches:
        r_work, r_free = matches[-1]  # last refinement macro-cycle
        return float(r_work), float(r_free)

    mw = _STATS_RWORK_RE.search(text)
    mf = _STATS_RFREE_RE.search(text)
    if mw and mf:
        return float(mw.group(1)), float(mf.group(1))
    return None, None


def _get_refine_log(run_dir):
    _yaml = list(run_dir.glob("run_paths.yaml"))[0]
    paths = load_config(_yaml)
    pred_dir = Path(paths["predictions_dir"])
    return [p.as_posix() for p in list(pred_dir.glob("**/refine_*.log"))]


def _get_refine_logs(
    run_dirs: list[Path],
) -> list[tuple[Path, list[str]]]:
    return [(r, _get_refine_log(r)) for r in run_dirs]


def _get_refine_lf(
    refine_logs,
    labels: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """Build a LazyFrame of per-epoch refinement R-factors.

    Logs that are missing or have no parseable R-factors are dropped, so an
    epoch whose refinement did not complete simply leaves a gap in the curve.
    """
    rows: list[dict] = []
    for run_dir, logs in refine_logs:
        label = labels.get(str(run_dir)) if labels is not None else None
        for log in logs:
            r_work, r_free = _parse_refine_log(log)
            if r_work is None:
                continue
            rows.append(
                {
                    "filenames": log,
                    "label": label,
                    "r_work": r_work,
                    "r_free": r_free,
                }
            )

    schema = {
        "filenames": pl.Utf8,
        "label": pl.Utf8,
        "r_work": pl.Float64,
        "r_free": pl.Float64,
    }
    return _add_run_epoch_cols(pl.DataFrame(rows, schema=schema).lazy())


def _per_epoch(frame: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Collapse to one (epoch, value) point per epoch, sorted by epoch.

    Rows with a null epoch or value are dropped, so an epoch with no
    measurement leaves a gap rather than misaligning the x/y arrays.
    """
    return (
        frame.drop_nulls(["epoch", value_col])
        .group_by("epoch")
        .agg(pl.col(value_col).mean())
        .sort("epoch")
    )


def _plot_peakz(lf, out_dir, colors=None, ref_data=None):
    # get reference data if provided
    if ref_data is not None:
        ref_df = pl.read_csv(
            ref_data.peaks,
            schema_overrides={
                "seqid": pl.Int64,
                "epoch": pl.Int64,
                "peakz": pl.Float32,
            },
        )

    # get run data
    _lf = lf.collect()
    if colors is None:
        key = _group_key_col(_lf)
        colors = _assign_colors(_lf[key].drop_nulls().unique().to_list())

    for (seqid,), df in _lf.group_by("seqid"):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for label, color, _df in _runs_by_label(df, colors):
            residue = _df["residue"].unique().item()

            series = _per_epoch(_df, "peakz")
            if series.height == 0:
                continue

            ax.plot(
                series["epoch"],
                series["peakz"],
                color=color,
                marker="o",
                markersize=5,
                label=label,
            )

        #  handling reference data
        if ref_data is not None:
            if seqid in ref_df["seqid"]:
                peak = ref_df.filter(pl.col("seqid") == seqid)["peakz"].item()
                ax.axhline(
                    peak,
                    label="DIALS",
                    color="red",
                    linestyle="--",
                )

        # plot params
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("peakz")
        plt.legend()
        plt.title(f"Anomalous signal for {residue} {seqid}")
        plt.savefig(f"{out_dir}/{residue}_{seqid}.png")
        plt.close(fig)


# R-work is drawn solid, R-free dashed; a run's two curves share a color.
_R_STYLES = {"r_work": "-", "r_free": "--"}


def _runs_by_label(df, colors):
    """(label, color, sub_df) per run, sorted alphabetically by label.

    `group_by` yields groups in nondeterministic order, so sort here to keep
    the plot and legend order stable across runs and figures.
    """
    runs = []
    for (id,), _df in df.group_by("run_id"):
        if "label" in _df.columns:
            _labels = _df["label"].drop_nulls().unique()
            label = _labels.item() if _labels.len() == 1 else id
        else:
            label = id
        runs.append((label, colors.get(label), _df))
    return sorted(runs, key=lambda r: str(r[0]))


def _refine_reference(ref_data) -> dict:
    """Reference {r_work, r_free} from `ref_data.refinement`, or {} if absent."""
    if ref_data is None or ref_data.refinement is None:
        return {}
    r_work, r_free = _parse_refine_log(ref_data.refinement)
    return {"r_work": r_work, "r_free": r_free} if r_work is not None else {}


def _legend_right(ax, **kwargs):
    """Place the legend just outside the axes, to the right."""
    return ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, **kwargs
    )


def _plot_refinement(lf, out_dir, colors=None, ref_data=None):
    """Plot per-epoch Rwork/Rfree: one combined figure plus one per R-factor.

    Every run keeps its shared Dark2 color. The combined figure draws R-work
    solid and R-free dashed (a run's two curves share a color); the individual
    figures draw a single R-factor per run for easier reading. Missing epochs
    are skipped (one point per epoch, sorted). An optional DIALS reference is
    drawn as a flat black line.
    """
    df = lf.collect()
    if df.height == 0:
        logger.info("No refinement logs found; skipping refinement plots")
        return

    if colors is None:
        key = _group_key_col(df)
        colors = _assign_colors(df[key].drop_nulls().unique().to_list())

    runs = _runs_by_label(df, colors)
    ref = _refine_reference(ref_data)

    # combined figure: both R-factors, solid = R-work, dashed = R-free
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for _label, color, _df in runs:
        for metric, style in _R_STYLES.items():
            series = _per_epoch(_df, metric)
            if series.height == 0:
                continue
            ax.plot(
                series["epoch"],
                series[metric],
                color=color,
                linestyle=style,
                marker="o",
                markersize=5,
            )

    handles = [Line2D([0], [0], color=c, lw=2, label=lab) for lab, c, _ in runs]
    if ref:
        ax.axhline(ref["r_work"], color="black", linestyle="-", lw=1)
        ax.axhline(ref["r_free"], color="black", linestyle="--", lw=1)
        handles.append(Line2D([0], [0], color="black", lw=1, label="DIALS"))
    # fold the R-factor line-style key into the same legend
    handles += [
        Line2D([0], [0], color="gray", lw=2, linestyle=s, label=m.replace("_", "-"))
        for m, s in _R_STYLES.items()
    ]

    ax.grid(True)
    ax.set_xlabel("epoch")
    ax.set_ylabel("R-factor")
    ax.set_title("Refinement R-factors")
    _legend_right(ax, handles=handles)
    fig.savefig(f"{out_dir}/refinement.png", bbox_inches="tight")
    plt.close(fig)

    # individual figures: one R-factor each, easier to read
    for metric in _R_STYLES:
        name = metric.replace("_", "-")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for label, color, _df in runs:
            series = _per_epoch(_df, metric)
            if series.height == 0:
                continue
            ax.plot(
                series["epoch"],
                series[metric],
                color=color,
                marker="o",
                markersize=5,
                label=label,
            )
        if metric in ref:
            ax.axhline(
                ref[metric], color="black", linestyle="--", lw=1, label="DIALS"
            )
        ax.grid(True)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.set_title(f"Refinement {name}")
        _legend_right(ax)
        fig.savefig(f"{out_dir}/refinement_{metric}.png", bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    logger.info("Begin plotting anomalous peak")

    cfg = parse_plot_cfg(load_config(args.plot_cfg)) if args.plot_cfg else None

    if args.run_dirs:
        run_dirs = args.run_dirs
    elif cfg:
        run_dirs = [v["path"] for v in cfg.runs.values()]
    else:
        raise ValueError("Requires --run-dirs or --plot-cfg")

    run_dirs = [Path(r) for r in run_dirs]
    csv = _get_peak_csvs(run_dirs)
    logger.info(f"Number of runs: {len(csv)}")

    labels = (
        {v["path"]: v["label"] for v in cfg.runs.values()} if cfg else None
    )
    lf = _get_peak_lf(csv, labels)
    refine_lf = _get_refine_lf(_get_refine_logs(run_dirs), labels)

    # one fixed color per model, shared by every plot so a model keeps its
    # color across the peak and refinement figures. Assigned in alphabetical
    # label order (falling back to run_id, unioned across both frames).
    if cfg is not None:
        color_keys = sorted({v["label"] for v in cfg.runs.values()})
    else:
        keys = set(
            refine_lf.select("run_id").unique().collect()["run_id"].to_list()
        )
        if lf is not None:
            keys |= set(
                lf.select("run_id").unique().collect()["run_id"].to_list()
            )
        color_keys = sorted(k for k in keys if k is not None)
    colors = _assign_colors(color_keys)

    # make save directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    logger.info(f"Saving images to: {out_dir.as_posix()}")

    ref_cfg = cfg.reference if cfg and cfg.reference else None

    # plot anomalous peak heights
    if lf is None:
        logger.warning(
            "no peaks.csv under any run: skipping the peak figure. Run the "
            "post-processing that produces them first."
        )
    else:
        _plot_peakz(lf, out_dir=out_dir, colors=colors, ref_data=ref_cfg)

    # plot refinement R-factors
    if refine_lf.select(pl.len()).collect().item() == 0:
        logger.warning(
            "no parseable refine_*.log under any run: skipping the "
            "refinement figure."
        )
    else:
        _plot_refinement(
            refine_lf, out_dir=out_dir, colors=colors, ref_data=ref_cfg
        )


if __name__ == "__main__":
    main()
