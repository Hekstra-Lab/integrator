"""Compare predictions per reflection against the reference integrator.

Per quantity:

    intensity_prf    -> model qi_mean   vs  DIALS intensity.prf.value
    intensity_sum    -> model qi_mean   vs  DIALS intensity.sum.value
    background_mean  -> model qbg_mean  vs  DIALS background.mean

The model has one intensity and DIALS has two: the profile fit and the flat
sum over the mask are different integrators, so the same model column is
drawn against each.

Two figures come out of each: the per-reflection scatter at one epoch, and
the correlation within resolution bins, which is the cheap way to read the
comparison over training (one line per epoch rather than ten scatters).

Three modes. The first two split the work because rendering is local --
the documents are set in fonts installed on the laptop only -- while the
1.9 million reflections are on the cluster:

    --dump-dir DIR    reduce each run's prediction parquet to two
                      long-format CSVs and stop. Runs on the cluster;
                      needs no figTeX.
    --data DIR        draw from those CSVs. Runs here, with figTeX.
    (neither)         read the parquet and draw in one pass, for a quick
                      look where the data is.

The reference columns travel inside the prediction parquet, written beside
the model's own at export, so `reference_data.refl` is only read when a run
was exported without them.

Usage:
    # on the cluster
    python plot_compare.py --plot-cfg cfg.yaml --dump-dir figdata/
    # here
    uv run --with-editable /Users/luis/master/software/figtex \\
        python plot_compare.py --data docs/figures/data --out-dir fig/
"""

import argparse
import logging
import re
import sys
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# quantity label -> (model prediction column, DIALS .refl column)
#
# The model has one intensity; DIALS has two, and they are different
# integrators. `intensity.prf.value` is its profile fit and
# `intensity.sum.value` its flat sum over the mask, so the same model column
# is compared against both -- the profile fit is the reference to beat and
# the summation is the one that holds up on weak data.
QUANTITIES = {
    "intensity_prf": {"model": "qi_mean", "dials": "intensity.prf.value"},
    "intensity_sum": {"model": "qi_mean", "dials": "intensity.sum.value"},
    "background_mean": {"model": "qbg_mean", "dials": "background.mean"},
}

# What earlier dumps called these. `intensity` was written before there was a
# second DIALS intensity to tell it apart from, so a `scatter_points.csv` on
# disk still carries the old name and is read under the new one rather than
# being redumped.
QUANTITY_ALIASES = {
    "intensity": "intensity_prf",
    "background": "background_mean",
}

# the axis a quantity is drawn on, and how its two sides are named
AXES = {
    "intensity_prf": {
        "scale": "log",
        "model": r"model $\langle I\rangle$",
        "dials": r"DIALS $I_\mathrm{prf}$",
    },
    "intensity_sum": {
        "scale": "log",
        "model": r"model $\langle I\rangle$",
        "dials": r"DIALS $I_\mathrm{sum}$",
    },
    "background_mean": {
        "scale": "linear",
        "model": r"model $\langle b\rangle$",
        "dials": r"DIALS $\bar{b}$",
    },
}


def quantities_in(have) -> dict[str, dict]:
    """The quantities whose DIALS column is present in `have`.

    A run exported before `intensity.sum.value` travelled in the prediction
    parquet is missing one column, not all of them: skipping the whole run
    for it would delete the intensity and background comparisons that are
    there, which is how a figure comes to be silently empty.
    """
    return {
        name: q for name, q in QUANTITIES.items() if q["dials"] in have
    }


def canonical_quantity(name: str) -> str:
    """The current name for a quantity label read off a dumped CSV."""
    return QUANTITY_ALIASES.get(name, name)

POINTS_CSV = "scatter_points.csv"
CC_CSV = "scatter_cc.csv"

_ID = "refl_ids"
_STYLE = None


def figstyle():
    """figTeX and the witch scheme, imported on first use.

    The reduction runs on the cluster, where figTeX is deliberately absent,
    so importing it at module scope would break the half of this script that
    does not draw anything.
    """
    global _STYLE
    if _STYLE is None:
        here = Path(__file__).resolve().parent / "scripts" / "figures"
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        import figstyle as _fs

        _STYLE = _fs
    return _STYLE


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-model log-log comparison of predictions vs DIALS"
    )
    parser.add_argument(
        "--plot-cfg", default=None, help="plot_cfg.yaml (not needed with --data)"
    )
    parser.add_argument(
        "--out-dir", default=".", help="Directory to write the figures to"
    )
    parser.add_argument(
        "--dump-dir",
        default=None,
        help="Reduce the parquet to long-format CSVs here and stop",
    )
    parser.add_argument(
        "--data",
        default=None,
        help=f"Draw from {POINTS_CSV} and {CC_CSV} in this directory",
    )
    parser.add_argument(
        "--dataset",
        default="hewl_9b7c",
        help="Value of the `dataset` key column in the dumped tables",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "all"),
        help="Reflections to compare; `test` is the split that gets exported",
    )
    parser.add_argument(
        "--points-epoch",
        type=int,
        default=None,
        help="Epoch the scatter is dumped at (default: the last). The "
        "correlation is dumped at every epoch either way, because a scatter "
        "per epoch is ten figures and 12 MB where ten lines are one and 8 kB.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Save to <run-dir>/plots (overrides --out-dir)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch to compare (default: latest available per run)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Subsample points per scatter for speed/file size",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Resolution bins for the per-bin correlation plot",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _find_pred_parquet(pred_dir: Path, epoch: int | None) -> list[Path]:
    """Resolve the prediction parquet(s) for one run's chosen epoch."""
    epoch_dirs = sorted(d for d in pred_dir.glob("epoch_*") if d.is_dir())
    if epoch_dirs and epoch is not None:
        match = [
            d for d in epoch_dirs if re.search(rf"epoch_0*{epoch}$", d.name)
        ]
        chosen = match[-1] if match else epoch_dirs[-1]
    elif epoch_dirs:
        chosen = epoch_dirs[-1]
    else:
        chosen = pred_dir  # predictions written directly under pred_dir

    single = chosen / "pred.parquet"
    if single.exists():
        return [single]
    shards = list(chosen.glob("preds_epoch_*.parquet"))
    if shards:
        return shards
    raise FileNotFoundError(f"No prediction parquet under {chosen}")


def _load_model(run_dir: Path, epoch: int | None) -> pl.DataFrame | None:
    """Load (refl_ids, qi_mean, qbg_mean) for one run, or None if unavailable."""
    rp = run_dir / "run_paths.yaml"
    if not rp.is_file():
        logger.warning("No run_paths.yaml in %s; skipping", run_dir)
        return None
    pred_dir = Path(load_config(rp)["predictions_dir"])
    try:
        files = _find_pred_parquet(pred_dir, epoch)
    except FileNotFoundError as e:
        logger.warning("%s; skipping", e)
        return None

    # `dict.fromkeys` because two quantities share one model column:
    # `qi_mean` is compared against both DIALS intensities
    cols = [_ID, *dict.fromkeys(q["model"] for q in QUANTITIES.values())]
    lf = pl.scan_parquet(files)
    have = lf.collect_schema().names()
    missing = [c for c in cols if c not in have]
    if missing:
        logger.warning("%s missing columns %s; skipping", pred_dir, missing)
        return None
    return lf.select(cols).collect()


def _load_reference(refl_path: Path) -> pl.DataFrame | None:
    """Read DIALS (refl_ids, intensity.prf.value, background.mean) from a .refl."""
    import reciprocalspaceship.io as rs_io

    from integrator.io.dtypes import DEFAULT_REFL_COLS

    ds = rs_io.read_dials_stills(str(refl_path), extra_cols=DEFAULT_REFL_COLS)
    present = quantities_in(ds)
    missing = [
        q["dials"] for n, q in QUANTITIES.items() if n not in present
    ]
    if missing:
        logger.warning(
            "reference %s has no %s; those comparisons are skipped",
            refl_path, ", ".join(missing),
        )
    if not present:
        return None
    cols = [_ID, *dict.fromkeys(q["dials"] for q in present.values())]
    if "d" in ds:  # resolution, for the per-bin correlation plot
        cols.append("d")
    return pl.DataFrame({c: np.asarray(ds[c]).ravel() for c in cols})


# --------------------------------------------------------------------------- #
# reduction: parquet -> two long-format CSVs
# --------------------------------------------------------------------------- #
def _epoch_dirs(pred_dir: Path) -> list[tuple[int, Path]]:
    """(epoch, directory) for every checkpoint exported under `pred_dir`."""
    out = []
    for d in sorted(pred_dir.glob("epoch_*")):
        m = re.search(r"epoch_0*(\d+)$", d.name)
        if d.is_dir() and m:
            out.append((int(m.group(1)), d))
    return out


def _parquets(d: Path) -> list[Path]:
    """The prediction parquet(s) in one epoch directory."""
    single = d / "pred.parquet"
    if single.exists():
        return [single]
    return sorted(d.glob("preds_epoch_*.parquet"))


def _load_pairs(files: list[Path], split: str) -> pl.DataFrame | None:
    """Model and reference columns for every quantity, row for row.

    Both sides are written into the same parquet at export, so no join and no
    `.refl` is involved; `d` comes along for the resolution bins.
    """
    lf = pl.scan_parquet(files)
    have = lf.collect_schema().names()
    present = quantities_in(have)
    want = [_ID, "d", "is_test"]
    for q in present.values():
        want += [q["model"], q["dials"]]
    missing = [
        c for c in (_ID, "d") if c not in have
    ] + [q["dials"] for n, q in QUANTITIES.items() if n not in present]
    if missing:
        logger.warning("%s has no %s", files[0], ", ".join(missing))
    if not present or _ID not in have or "d" not in have:
        logger.warning("%s: nothing comparable; skipping", files[0])
        return None
    cols = [c for c in dict.fromkeys(want) if c in have]
    df = lf.select(cols).collect()
    if split == "test":
        if "is_test" not in df.columns:
            logger.warning("%s has no is_test column; using every row", files[0])
        else:
            df = df.filter(pl.col("is_test").cast(pl.Boolean))
    return df


def dump_tables(cfg, out_dir: Path, args) -> None:
    """Write the scatter points and the binned correlation, then stop.

    The resolution bin edges are taken once, from the first epoch read, and
    reused for every later one: `d` is a property of the reflection and does
    not move between checkpoints, so re-quantiling per epoch would only make
    the lines incomparable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pts: list[dict] = []
    ccs: list[dict] = []

    for v in sorted(cfg.get("runs", {}).values(), key=lambda v: str(v["label"])):
        label = str(v["label"])
        rp = Path(v["path"]) / "run_paths.yaml"
        if not rp.is_file():
            logger.warning("No run_paths.yaml in %s; skipping", v["path"])
            continue
        pred_dir = Path(load_config(rp)["predictions_dir"])
        epochs = _epoch_dirs(pred_dir)
        if not epochs:
            logger.warning("No epoch directories under %s; skipping", pred_dir)
            continue
        want_pts = (
            args.points_epoch if args.points_epoch is not None else epochs[-1][0]
        )
        edges = None

        for epoch, d in epochs:
            files = _parquets(d)
            if not files:
                logger.warning("No parquet in %s; skipping", d)
                continue
            df = _load_pairs(files, args.split)
            if df is None:
                continue
            dv = df["d"].cast(pl.Float64).to_numpy()
            if edges is None:
                fin = dv[np.isfinite(dv)]
                edges = np.quantile(fin, np.linspace(0.0, 1.0, args.n_bins + 1))

            for quantity, q in quantities_in(df.columns).items():
                log = AXES[quantity]["scale"] == "log"
                x = df[q["model"]].cast(pl.Float64).to_numpy()
                y = df[q["dials"]].cast(pl.Float64).to_numpy()
                key = {
                    "method": label,
                    "dataset": args.dataset,
                    "split": args.split,
                    "epoch": epoch,
                    "quantity": quantity,
                }

                for i, cc in enumerate(_binned_cc(x, y, dv, edges, log)):
                    lo, hi = edges[i], edges[i + 1]
                    in_bin = (dv >= lo) & (
                        dv <= hi if i == args.n_bins - 1 else dv < hi
                    )
                    ccs.append({
                        **key,
                        "bin": i + 1,
                        "d_max": round(float(hi), 4),
                        "d_min": round(float(lo), 4),
                        "n": int(in_bin.sum()),
                        "cc": cc,
                    })

                if epoch != want_pts:
                    continue
                keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(dv)
                if log:
                    keep &= (x > 0) & (y > 0)
                idx = np.flatnonzero(keep)
                if idx.size > args.max_points:
                    idx = np.random.default_rng(0).choice(
                        idx, args.max_points, replace=False
                    )
                for j in idx:
                    pts.append({
                        **key,
                        "model": float(x[j]),
                        "dials": float(y[j]),
                        "d": float(dv[j]),
                    })
            logger.info("%s epoch %d: %d reflections", label, epoch, df.height)

    for rows, name in ((pts, POINTS_CSV), (ccs, CC_CSV)):
        if not rows:
            logger.warning("%s: nothing to write", name)
            continue
        pl.DataFrame(rows).write_csv(out_dir / name)
        logger.info("Wrote %s (%d rows)", out_dir / name, len(rows))


# --------------------------------------------------------------------------- #
# matching + plotting
# --------------------------------------------------------------------------- #
def _matched(df_x: pl.DataFrame, xcol: str, df_y: pl.DataFrame, ycol: str):
    """Inner-join two sources on refl_ids; return aligned (x, y) arrays.

    refl_ids can arrive as int (pred parquet) or float (DIALS via rs), so both
    keys are normalized to Int64 before joining.
    """

    def prep(df: pl.DataFrame, col: str, name: str) -> pl.DataFrame:
        return df.select(
            pl.col(_ID).cast(pl.Float64).round().cast(pl.Int64),
            pl.col(col).cast(pl.Float64).alias(name),
        )

    j = prep(df_x, xcol, "x").join(prep(df_y, ycol, "y"), on=_ID, how="inner")
    return j["x"].to_numpy(), j["y"].to_numpy()


def _scatter(x, y, color, xlabel, ylabel, title, out_path, max_points, scale):
    """Per-reflection scatter of model (x) against the reference (y)."""
    fs = figstyle()
    log = scale == "log"
    keep = np.isfinite(x) & np.isfinite(y)
    if log:
        keep &= (x > 0) & (y > 0)
    n_keep, n_all = int(keep.sum()), x.size
    x, y = x[keep], y[keep]
    if x.size == 0:
        logger.warning("No points left for %s; skipping", title)
        return
    if n_keep < n_all:
        logger.info(
            "%s: dropped %d of %d (non-finite%s)",
            title, n_all - n_keep, n_all, ", non-positive" if log else "",
        )

    a, b = (np.log(x), np.log(y)) if log else (x, y)
    r = float(np.corrcoef(a, b)[0, 1]) if x.size > 1 else float("nan")
    logger.info("%s: r = %.4f over %d reflections", title, r, n_keep)

    xp, yp = x, y
    if xp.size > max_points:
        sel = np.random.default_rng(0).choice(
            xp.size, max_points, replace=False
        )
        xp, yp = xp[sel], yp[sel]

    roles = fs.roles()
    fig, ax = plt.subplots(1, 1, figsize=fs.figsize("half", aspect=1.0))
    ax.scatter(
        xp, yp, s=2, alpha=0.25, color=color, linewidths=0, rasterized=True
    )

    # the identity line is the whole point of the figure, so both axes get the
    # same range and the same scale -- a scatter against a reference read on
    # two different ranges hides exactly the disagreement it is drawn to show
    if log:
        lo = max(min(float(x.min()), float(y.min())), 1e-2)
    else:
        lo = min(float(x.min()), float(y.min()), 0.0)
    hi = max(float(x.max()), float(y.max()), lo + 1.0)
    ref = fs.reference_style() | {"lw": 0.9}  # thin: it crosses the dense core
    ax.plot([lo, hi], [lo, hi], linestyle=(0, (4, 2)), label=r"$y = x$", **ref)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="both", color=roles["grid"], lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    fs.save(fig, out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# A comparison source: DIALS or a model. `kind` selects the quantity column.
Source = namedtuple("Source", ["name", "df", "kind", "color"])


def _qcol(kind: str, quantity: str) -> str:
    """Column for a quantity on a given source ('dials' vs 'model')."""
    return QUANTITIES[quantity]["dials" if kind == "dials" else "model"]


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _compare(
    xs: Source,
    ys: Source,
    quantity: str,
    out_dir: Path,
    max_points: int,
    scale: str = "log",
):
    """One log-log scatter of source `xs` (x) against source `ys` (y)."""
    xcol, ycol = _qcol(xs.kind, quantity), _qcol(ys.kind, quantity)
    x, y = _matched(xs.df, xcol, ys.df, ycol)
    axes = AXES[quantity]
    _scatter(
        x,
        y,
        xs.color,
        xlabel=f"{xs.name}, {axes['model']}",
        ylabel=axes["dials"] if ys.kind == "dials" else f"{ys.name}, {axes['model']}",
        title=f"{xs.name} against {ys.name}",
        out_path=out_dir
        / f"compare_{quantity}_{_slug(xs.name)}_vs_{_slug(ys.name)}.png",
        max_points=max_points,
        scale=scale,
    )


# --------------------------------------------------------------------------- #
# correlation vs resolution
# --------------------------------------------------------------------------- #
def _matched_d(model_df, model_col, dials_df, dials_col):
    """Join model vs DIALS on refl_ids; return (model, DIALS, resolution d)."""
    a = model_df.select(
        pl.col(_ID).cast(pl.Float64).round().cast(pl.Int64),
        pl.col(model_col).cast(pl.Float64).alias("x"),
    )
    b = dials_df.select(
        pl.col(_ID).cast(pl.Float64).round().cast(pl.Int64),
        pl.col(dials_col).cast(pl.Float64).alias("y"),
        pl.col("d").cast(pl.Float64).alias("d"),
    )
    j = a.join(b, on=_ID, how="inner")
    return j["x"].to_numpy(), j["y"].to_numpy(), j["d"].to_numpy()


def _binned_cc(x, y, d, edges, log):
    """Pearson correlation of x vs y within each resolution (d) bin."""
    ccs = []
    n = len(edges) - 1
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (d >= lo) & (d <= hi) if i == n - 1 else (d >= lo) & (d < hi)
        xi, yi = x[in_bin], y[in_bin]
        if log:
            pos = (xi > 0) & (yi > 0)
            xi, yi = xi[pos], yi[pos]
        if xi.size > 2:
            a = np.log(xi) if log else xi
            b = np.log(yi) if log else yi
            ccs.append(float(np.corrcoef(a, b)[0, 1]))
        else:
            ccs.append(np.nan)
    return ccs


def _resolution_edges(dials, n_bins):
    """Quantile edges of DIALS resolution d, plus per-bin range labels."""
    d = dials.df["d"].to_numpy()
    d = d[np.isfinite(d)]
    edges = np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1))
    labels = [f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(n_bins)]
    return edges, labels


def _draw_correlation(ax, model, dials, quantity, edges, log):
    """Plot one model's CC-vs-resolution curve on `ax`."""
    xv, yv, dv = _matched_d(
        model.df, _qcol("model", quantity), dials.df, _qcol("dials", quantity)
    )
    ccs = _binned_cc(xv, yv, dv, edges, log)
    # reverse so low resolution (large d) is on the left (DIALS convention)
    ax.plot(
        range(len(edges) - 1), ccs[::-1], marker="o",
        color=model.color, label=model.name,
    )


def _finish_correlation(ax, labels, quantity, title, legend=True):
    fs = figstyle()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels[::-1], rotation=55, ha="right")
    ax.set_xlabel(r"resolution (\AA)")
    ax.set_ylabel(_cc_label(quantity))
    ax.set_title(title)
    ax.grid(True, color=fs.roles()["grid"], lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(frameon=False)


def _cc_label(quantity: str) -> str:
    """Name what the correlation is between, not just 'CC'."""
    a = AXES[quantity]
    inner = f"{a['model']}, {a['dials']}"
    if AXES[quantity]["scale"] == "log":
        inner = f"$\\log$ {a['model']}, $\\log$ {a['dials']}"
    return f"$r$({inner})"


def _save(fig, out_path):
    figstyle().save(fig, out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_correlation(quantity, models, dials, out_dir, n_bins, log):
    """CC(model, DIALS) per resolution bin, all models overlaid on one figure."""
    edges, labels = _resolution_edges(dials, n_bins)
    fig, ax = plt.subplots(figsize=figstyle().figsize("full", aspect=0.6))
    for m in models:
        _draw_correlation(ax, m, dials, quantity, edges, log)
    _finish_correlation(ax, labels, quantity, f"{quantity}: correlation vs resolution")
    _save(fig, out_dir / f"correlation_{quantity}.png")


def _plot_correlation_individual(quantity, model, dials, out_dir, n_bins, log):
    """CC(model, DIALS) per resolution bin for a single model."""
    edges, labels = _resolution_edges(dials, n_bins)
    fig, ax = plt.subplots(figsize=figstyle().figsize("full", aspect=0.6))
    _draw_correlation(ax, model, dials, quantity, edges, log)
    _finish_correlation(ax, labels, quantity, f"{quantity}: {model.name} vs DIALS")
    _save(fig, out_dir / f"correlation_{quantity}_{_slug(model.name)}.png")


# --------------------------------------------------------------------------- #
# drawing from the dumped tables (this is the path the paper figures take)
# --------------------------------------------------------------------------- #
def _read_table(data_dir: Path, name: str) -> pl.DataFrame | None:
    """Read a dumped table, with the old quantity labels mapped forward.

    A CSV dumped before the second DIALS intensity existed calls its two
    quantities `intensity` and `background`; they are read as
    `intensity_prf` and `background_mean` so an existing file still draws.
    """
    p = data_dir / name
    if not p.is_file():
        logger.warning("%s not in %s", name, data_dir)
        return None
    df = pl.read_csv(p)
    if "quantity" in df.columns:
        df = df.with_columns(
            pl.col("quantity").replace(QUANTITY_ALIASES)
        )
    return df


def _draw_scatter_table(df: pl.DataFrame, out_dir: Path, max_points: int):
    """One scatter per (method, quantity) from the dumped points."""
    fs = figstyle()
    for (method, quantity), g in df.group_by(["method", "quantity"]):
        if quantity not in AXES:
            logger.warning("Unknown quantity %s; skipping", quantity)
            continue
        axes = AXES[quantity]
        epochs = sorted(set(g["epoch"].to_list()))
        for epoch in epochs:
            h = g.filter(pl.col("epoch") == epoch)
            _scatter(
                h["model"].to_numpy(),
                h["dials"].to_numpy(),
                fs.arm_colors([method])[method],
                xlabel=axes["model"],
                ylabel=axes["dials"],
                title=f"{method}, epoch {epoch}",
                out_path=out_dir
                / f"scatter_{quantity}_{_slug(str(method))}_ep{epoch:04d}.png",
                max_points=max_points,
                scale=axes["scale"],
            )


def _draw_cc_table(df: pl.DataFrame, out_dir: Path):
    """Correlation within resolution bins, one line per epoch.

    This is how the comparison is read over training: the epoch sits on the
    sequential ramp, so the direction of travel is legible without a legend
    entry per checkpoint.
    """
    fs = figstyle()
    for (method, quantity), g in df.group_by(["method", "quantity"]):
        if quantity not in AXES:
            continue
        epochs = sorted(set(g["epoch"].to_list()))
        cmap, colors = fs.epoch_ramp(len(epochs))
        bins = g.filter(pl.col("epoch") == epochs[0]).sort("bin")
        labels = [
            f"{lo:.2f}--{hi:.2f}"
            for lo, hi in zip(bins["d_min"], bins["d_max"], strict=True)
        ]
        fig, ax = plt.subplots(figsize=fs.figsize("full", aspect=0.6))
        for i, epoch in enumerate(epochs):
            h = g.filter(pl.col("epoch") == epoch).sort("bin")
            cc = h["cc"].to_numpy()
            # reverse so low resolution (large d) is on the left, as the
            # DIALS merging tables are printed
            ax.plot(
                range(len(cc)), cc[::-1], marker="o", ms=3,
                color=colors[i], lw=1.2, label=f"epoch {epoch}",
            )
        _finish_correlation(
            ax, labels, quantity,
            f"{method}: {AXES[quantity]['dials']} against the model",
            legend=False,
        )
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=min(epochs), vmax=max(epochs)),
        )
        fig.colorbar(sm, ax=ax, label="epoch", pad=0.02)
        _save(
            fig,
            out_dir / f"cc_resolution_{quantity}_{_slug(str(method))}.png",
        )


def draw_from_tables(data_dir: Path, out_dir: Path, max_points: int) -> None:
    figstyle().use()
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = _read_table(data_dir, POINTS_CSV)
    if pts is not None:
        _draw_scatter_table(pts, out_dir, max_points)
    cc = _read_table(data_dir, CC_CSV)
    if cc is not None:
        _draw_cc_table(cc, out_dir)
    if pts is None and cc is None:
        raise SystemExit(f"Neither {POINTS_CSV} nor {CC_CSV} is in {data_dir}")


def main():
    args = parse_args()

    out_dir = (
        Path(args.run_dir) / "plots" if args.run_dir else Path(args.out_dir)
    )

    # drawing from the dumped tables needs no run, no parquet and no cluster
    if args.data:
        draw_from_tables(Path(args.data), out_dir, args.max_points)
        return

    if not args.plot_cfg:
        raise SystemExit("--plot-cfg is required unless --data is given")
    cfg = load_config(args.plot_cfg)

    # the reduction writes tables and draws nothing, so it runs where the
    # reflections are
    if args.dump_dir:
        dump_tables(cfg, Path(args.dump_dir), args)
        return

    figstyle().use()
    ref = (cfg.get("reference_data") or {}).get("refl")
    dials_df = _load_reference(Path(ref)) if ref else None
    if ref and dials_df is None:
        logger.warning(
            "Could not read DIALS reference; model-vs-DIALS skipped"
        )
    dials = (
        Source("DIALS", dials_df, "dials", "black")
        if dials_df is not None
        else None
    )

    # models in alphabetical label order, each keeping one witch colour
    runs = sorted(cfg.get("runs", {}).values(), key=lambda v: str(v["label"]))
    loaded = [(str(v["label"]), _load_model(Path(v["path"]), args.epoch)) for v in runs]
    palette = figstyle().arm_colors([n for n, df in loaded if df is not None])
    models: list[Source] = [
        Source(name, df, "model", palette[name])
        for name, df in loaded
        if df is not None
    ]

    if not models or (dials is None and len(models) < 2):
        raise SystemExit(
            "Nothing to compare (need DIALS + a model, or 2 models)"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    for quantity in QUANTITIES:
        scale = AXES[quantity]["scale"]

        # each model vs DIALS (model on x, DIALS on y)
        if dials is not None:
            for m in models:
                _compare(m, dials, quantity, out_dir, args.max_points, scale)
        # each model vs each other model (alphabetical-first on x)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                _compare(
                    models[i],
                    models[j],
                    quantity,
                    out_dir,
                    args.max_points,
                    scale,
                )

        # correlation coefficient per resolution bin, model(s) vs DIALS
        if dials is not None and "d" in dials.df.columns:
            _plot_correlation(
                quantity, models, dials, out_dir, args.n_bins,
                log=(scale == "log"),
            )
            for m in models:
                _plot_correlation_individual(
                    quantity, m, dials, out_dir, args.n_bins,
                    log=(scale == "log"),
                )


if __name__ == "__main__":
    main()
