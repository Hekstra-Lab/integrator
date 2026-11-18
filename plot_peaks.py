import argparse
import logging
import sys
from dataclasses import dataclass
from importlib.resources import as_file
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import yaml

from integrator.utils import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "figures"))
import figstyle  # noqa: E402
from matplotlib.lines import Line2D

# logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# The witch categorical palette is the per-run colour source; `figstyle.use`
# installs it as the cycle, along with the document's fonts and sizes.
_JOURNAL = figstyle.use()


def _palette(n: int):
    """Qualitative colors for `n` keys, without repeating one.

    The witch categorical palette holds 8, ordered most distinct first; past
    that `i % 8` would silently give the 9th run the 1st run's colour, so a
    longer ramp is sampled instead.
    """
    if n <= 8:
        return list(figstyle.arm_colors(range(8)).values())
    cmap = figstyle.epoch_ramp(n)[0]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _assign_colors(keys) -> dict:
    """Map each key to a fixed qualitative color in the given order.

    Build this once (from the plot-config run order) and share it across every
    figure so a model keeps the same color in the peak and refinement plots.
    """
    ordered = list(dict.fromkeys(keys))  # dedupe, preserve order
    pal = _palette(len(ordered))
    return {k: pal[i % len(pal)] for i, k in enumerate(ordered)}


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


# `rs.find_peaks` writes a `name` column for the nearest atom and the fork in
# use leaves it empty, so the atom is resolved against the refined model that
# the peak search itself ran on. The residue name does not settle it: a
# sulfur peak is SD on a methionine and SG on a cysteine.
def _atom_table(peak_csvs) -> pl.DataFrame | None:
    """(chain, seqid, residue, atom, n_heavy) per site, from the models.

    One row per site, not per epoch: an atom's identity does not change
    between checkpoints, so the mapping is built once and joined on.
    `metrics.peaks_from_find_peaks_csv` accepts the model's answer only
    where the chain, residue number and residue name agree with what the
    peak search reported, so a site it cannot confirm comes back with an
    empty atom and is drawn under its residue alone.

    `n_heavy` is how many atoms in that residue could carry an anomalous
    peak, which is what `metrics.panel_site_label` needs: it is 1 for a
    cysteine, a methionine, an iodide and a sulfate, so those titles do not
    have to name the atom at all.
    """
    rows: dict[tuple, dict] = {}
    for _, files in peak_csvs:
        for f in files:
            model = sorted(
                Path(f).parent.glob("*[0-9].pdb"),
                key=lambda q: q.stat().st_mtime,
            )
            if not model:
                continue
            candidates = metrics.heavy_atom_candidates(model[-1])
            for peak in metrics.peaks_from_find_peaks_csv(f, model=model[-1]):
                if not peak["atom"] or peak["seqid"] is None:
                    continue
                key = (peak["chain"], peak["seqid"], peak["residue"])
                rows.setdefault(key, {
                    "chain": peak["chain"],
                    "seqid": peak["seqid"],
                    "residue": peak["residue"],
                    "atom": peak["atom"],
                    "n_heavy": len(candidates.get(key, [])),
                })
    if not rows:
        return None
    return pl.DataFrame(
        list(rows.values()),
        schema={"chain": pl.Utf8, "seqid": pl.Int64,
                "residue": pl.Utf8, "atom": pl.Utf8, "n_heavy": pl.Int64},
    )


def _with_atom_names(lf: pl.LazyFrame, peak_csvs) -> pl.LazyFrame:
    """Add an `atom` column, empty where no model confirmed the site."""
    table = _atom_table(peak_csvs)
    if table is None:
        logger.info(
            "no refined PDB beside any peaks.csv: panel titles will name the "
            "residue without the atom"
        )
        return lf.with_columns(
            pl.lit("").alias("atom"), pl.lit(1, dtype=pl.Int64).alias("n_heavy")
        )
    return lf.join(
        table.lazy(), on=["chain", "seqid", "residue"], how="left"
    ).with_columns(
        pl.col("atom").fill_null(""), pl.col("n_heavy").fill_null(1)
    )


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
def _parse_refine_log(path) -> tuple[float | None, float | None]:
    """(r_work, r_free) from a phenix refine log or a stats file.

    The regular expressions live in `integrator.utils.metrics`, with the
    chain's own emitter, so this figure and the chain cannot disagree about
    what a refinement produced. Two details that were easy to get wrong and
    are pinned by its tests: the LAST `Final R-work` line wins, because
    phenix prints `Start R-work` first and a rerun appends; and a stats file
    writes the same numbers as `Rwork:`/`Rfree:`, which is the only form the
    9b7c reference comes in.

    Returns (None, None) when the file is missing or has no parseable
    R-factors, so an epoch whose refinement failed is skipped rather than
    aborting the whole plot.
    """
    got = metrics.refinement_from_phenix_log(path)
    if got is None:
        return None, None
    return got["r_work"], got["r_free"]


# phenix logs are named differently by the two pipelines: the Laue chain writes
# `refine_1.log`/`refine_2.log`, the monochromatic one `phenix_refine.log` and
# `refined_001.log`. Globbing only the first silently drops every mono R-factor,
# so all three are matched and `_parse_refine_log` decides what is parseable.
REFINE_LOG_GLOBS = ("**/refine_*.log", "**/refined_*.log", "**/phenix_refine.log")


def _get_refine_log(run_dir):
    _yaml = list(run_dir.glob("run_paths.yaml"))[0]
    paths = load_config(_yaml)
    pred_dir = Path(paths["predictions_dir"])
    found = []
    for pattern in REFINE_LOG_GLOBS:
        found.extend(p.as_posix() for p in pred_dir.glob(pattern))
    return sorted(set(found))


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
        fig, ax = plt.subplots(
            1, 1, figsize=figstyle.figsize("half", aspect=0.72),
            layout="constrained")
        site = ""
        for label, color, _df in _runs_by_label(df, colors):
            residue = _df["residue"].unique().item()
            atom, n_heavy = "", 1
            if "atom" in _df.columns:
                atoms = _df["atom"].drop_nulls().unique().to_list()
                atom = atoms[0] if len(atoms) == 1 else ""
            if "n_heavy" in _df.columns:
                counts = _df["n_heavy"].drop_nulls().unique().to_list()
                n_heavy = max(counts) if counts else 1
            site = metrics.panel_site_label(residue, seqid, atom, n_heavy)

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
                    linestyle=(0, (4, 2)),
                    **figstyle.reference_style(),
                )

        # plot params
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"peak height ($\sigma$)")
        ax.legend(frameon=False)
        # the filename stays `<RESIDUE>_<seqid>` whatever the title says:
        # the gallery sorts a per-site panel by that stem and a third field
        # would drop every one of them into the unsorted section
        plt.title(f"Anomalous signal for {site or f'{residue} {seqid}'}")
        figstyle.save(fig, f"{out_dir}/{residue}_{seqid}")
        plt.close(fig)


_R_LABELS = {"r_work": r"$R_\mathrm{work}$", "r_free": r"$R_\mathrm{free}$"}
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


# The residues that carry the anomalous signal. A water or a side chain that
# happens to sit near a peak is not an anomalous scatterer, and averaging it in
# drags the mean towards the noise floor.
# the residues carrying anomalous signal; defined once, beside the
# parsers, because `tidy_mono.py` and the chain filter on the same set
ANOMALOUS_RESIDUES = metrics.ANOMALOUS_RESIDUES


def _plot_peakz_mean(lf, out_dir, colors=None, ref_data=None):
    """Mean peak height over the anomalous scatterers, against epoch.

    One number per epoch where `_plot_peakz` gives one panel per site: it is
    the summary the per-site panels support, and it is what moves when the
    model gets better rather than when one site happens to be well ordered.
    """
    _lf = lf.collect()
    _lf = _lf.filter(pl.col("residue").is_in(ANOMALOUS_RESIDUES))
    if _lf.height == 0:
        logger.info("No anomalous scatterers in the peak table; skipping")
        return
    if colors is None:
        colors = _assign_colors(
            _lf[_group_key_col(_lf)].drop_nulls().unique().to_list()
        )

    fig, ax = plt.subplots(
        1, 1, figsize=figstyle.figsize("half", aspect=0.72),
        layout="constrained")
    for label, color, _df in _runs_by_label(_lf, colors):
        m = (
            _df.group_by("epoch")
            .agg(pl.col("peakz").mean().alias("mean"),
                 pl.col("peakz").count().alias("n"))
            .sort("epoch")
        )
        ax.plot(m["epoch"], m["mean"], color=color, marker="o", markersize=4,
                label=f"{label} ({int(m['n'][0])} sites)")
    if ref_data is not None and ref_data.peaks is not None:
        ref_df = pl.read_csv(
            ref_data.peaks,
            schema_overrides={"seqid": pl.Int64, "epoch": pl.Int64,
                              "peakz": pl.Float32},
        ).filter(pl.col("residue").is_in(ANOMALOUS_RESIDUES))
        if ref_df.height:
            ax.axhline(
                float(ref_df["peakz"].mean()), label="DIALS",
                linestyle=(0, (4, 2)), **figstyle.reference_style(),
            )
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"mean peak height ($\sigma$)")
    ax.legend(frameon=False)
    figstyle.save(fig, f"{out_dir}/peakz_mean")
    plt.close(fig)
    logger.info("Saved %s/peakz_mean", out_dir)


def _plot_refinement(lf, out_dir, colors=None, ref_data=None):
    """Plot per-epoch Rwork/Rfree: one combined figure plus one per R-factor.

    Every run keeps its shared witch colour. The combined figure draws R-work
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
    fig, ax = plt.subplots(
        1, 1, figsize=figstyle.figsize("half", aspect=0.72),
        layout="constrained")
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
    refkw = figstyle.reference_style()
    if ref:
        ax.axhline(ref["r_work"], linestyle="-", **refkw)
        ax.axhline(ref["r_free"], linestyle=(0, (4, 2)), **refkw)
        handles.append(
            Line2D([0], [0], color=refkw["color"], lw=refkw["lw"], label="DIALS")
        )
    # fold the R-factor line-style key into the same legend
    handles += [
        Line2D([0], [0], color=figstyle.roles()["muted"], lw=2, linestyle=s,
               label=_R_LABELS[m])
        for m, s in _R_STYLES.items()
    ]

    ax.grid(axis="x", visible=False)
    ax.set_xlabel("epoch")
    ax.set_ylabel("$R$-factor")
    _legend_right(ax, handles=handles)
    figstyle.save(fig, f"{out_dir}/refinement")
    plt.close(fig)

    # individual figures: one R-factor each, easier to read
    for metric in _R_STYLES:
        name = metric.replace("_", "-")
        fig, ax = plt.subplots(
            1, 1, figsize=figstyle.figsize("half", aspect=0.72),
            layout="constrained")
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
                ref[metric], linestyle=(0, (4, 2)), label="DIALS",
                **figstyle.reference_style(),
            )
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("epoch")
        ax.set_ylabel(_R_LABELS[metric])
        _legend_right(ax)
        figstyle.save(fig, f"{out_dir}/refinement_{metric}")
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
    if lf is not None:
        lf = _with_atom_names(lf, csv)
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
        _plot_peakz_mean(
            lf, out_dir=out_dir, colors=colors, ref_data=ref_cfg
        )

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
