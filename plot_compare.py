"""Compare predictions per reflection, one log-log scatter per comparison.

Reads a plot-config (same schema as plot_peaks.py), finds each run's
prediction parquet, and draws an individual log-log scatter for:
  * each model vs DIALS   (model on x, DIALS on y)
  * each model vs each other model

plus, per quantity, one `correlation_<quantity>.png` giving the model-vs-DIALS
correlation coefficient per resolution bin (a line per model).

per quantity:

    intensity  -> model qi_mean   vs  DIALS intensity.prf.value
    background -> model qbg_mean   vs  DIALS background.mean

(model-vs-model uses the model column, e.g. qi_mean, on both axes.)

The DIALS reference is read from `reference_data.refl`. Sources are matched by
their shared `refl_ids` column.

Usage:
    uv run python plot_compare.py --plot-cfg plot_cfg.yaml --out-dir compare/
"""

import argparse
import logging
import re
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# quantity label -> (model prediction column, DIALS .refl column)
QUANTITIES = {
    "intensity": {"model": "qi_mean", "dials": "intensity.prf.value"},
    "background": {"model": "qbg_mean", "dials": "background.mean"},
}

_ID = "refl_ids"
_DARK2 = plt.get_cmap("Dark2").colors


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-model log-log comparison of predictions vs DIALS"
    )
    parser.add_argument("--plot-cfg", required=True, help="plot_cfg.yaml")
    parser.add_argument(
        "--out-dir", default=".", help="Directory to write the figures to"
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

    cols = [_ID, *(q["model"] for q in QUANTITIES.values())]
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
    cols = [_ID, *(q["dials"] for q in QUANTITIES.values())]
    missing = [c for c in cols if c not in ds]
    if missing:
        logger.warning("reference %s missing columns %s", refl_path, missing)
        return None
    if "d" in ds:  # resolution, for the per-bin correlation plot
        cols.append("d")
    return pl.DataFrame({c: np.asarray(ds[c]).ravel() for c in cols})


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
    """Individual log-log scatter of model (x) vs DIALS (y)."""
    pos = (x > 0) & (y > 0)
    n_pos, n_all = int(pos.sum()), x.size
    x, y = x[pos], y[pos]
    if x.size == 0:
        logger.warning("No positive points for %s; skipping", title)
        return

    r = np.corrcoef(np.log(x), np.log(y))[0, 1] if x.size > 1 else np.nan

    xp, yp = x, y
    if xp.size > max_points:
        sel = np.random.default_rng(0).choice(
            xp.size, max_points, replace=False
        )
        xp, yp = xp[sel], yp[sel]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(xp, yp, s=4, alpha=0.2, color=color, rasterized=True)

    lo = 1e-1
    hi = max(float(x.max()), float(y.max()), lo * 10)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.text(
    #     0.05,
    #     0.95,
    #     f"r={r:.3f}\nn={n_pos}",
    #     transform=ax.transAxes,
    #     va="top",
    #     fontsize=9,
    # )
    if n_pos < n_all:
        logger.info(
            "%s: dropped %d non-positive of %d", title, n_all - n_pos, n_all
        )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
    _scatter(
        x,
        y,
        xs.color,
        xlabel=f"{xs.name}  ({quantity})",
        ylabel=f"DIALS ({quantity})",
        title=f"{quantity}: {xs.name} vs {ys.name}",
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


def _finish_correlation(ax, labels, quantity, title):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels[::-1], rotation=55, fontsize=10, ha="right")
    ax.set_xlabel("resolution (Å)")
    ax.set_ylabel(f"CC ({quantity} vs DIALS)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_correlation(quantity, models, dials, out_dir, n_bins, log):
    """CC(model, DIALS) per resolution bin, all models overlaid on one figure."""
    edges, labels = _resolution_edges(dials, n_bins)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        _draw_correlation(ax, m, dials, quantity, edges, log)
    _finish_correlation(ax, labels, quantity, f"{quantity}: correlation vs resolution")
    _save(fig, out_dir / f"correlation_{quantity}.png")


def _plot_correlation_individual(quantity, model, dials, out_dir, n_bins, log):
    """CC(model, DIALS) per resolution bin for a single model."""
    edges, labels = _resolution_edges(dials, n_bins)
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_correlation(ax, model, dials, quantity, edges, log)
    _finish_correlation(ax, labels, quantity, f"{quantity}: {model.name} vs DIALS")
    _save(fig, out_dir / f"correlation_{quantity}_{_slug(model.name)}.png")


def main():
    args = parse_args()
    cfg = load_config(args.plot_cfg)

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

    # models in alphabetical label order, contiguous Dark2 colors
    runs = sorted(cfg.get("runs", {}).values(), key=lambda v: str(v["label"]))
    models: list[Source] = []
    for v in runs:
        df = _load_model(Path(v["path"]), args.epoch)
        if df is not None:
            color = _DARK2[len(models) % len(_DARK2)]
            models.append(Source(str(v["label"]), df, "model", color))

    if not models or (dials is None and len(models) < 2):
        raise SystemExit(
            "Nothing to compare (need DIALS + a model, or 2 models)"
        )

    out_dir = (
        Path(args.run_dir) / "plots" if args.run_dir else Path(args.out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for quantity in QUANTITIES:
        if quantity == "background":
            scale = "linear"
        else:
            scale = "log"

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
