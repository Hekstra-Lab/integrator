"""Plot training loss curves: ELBO, NLL, and the individual KL components.

Reads a plot-config (same schema as plot_peaks.py), finds each run's
`plots/loss_history.csv` (per-epoch, columns `epoch` and `{split}_{term}`),
and draws:

  * per-model     -> one 2x3 grid of every component (train solid, val dashed)
  * inter-model   -> one figure per component overlaying all models

Components: loss (ELBO), nll, kl (total), kl_prf, kl_i, kl_bg.

Usage:
    uv run python plot_loss.py --plot-cfg plot_cfg.yaml --out-dir loss/
"""

import argparse
import logging
import math
import re
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# loss-trace column -> display label, in grid order
METRICS = {
    "loss": "ELBO (loss)",
    "nll": "NLL",
    "kl": "KL (total)",
    "kl_prf": "KL profile",
    "kl_i": "KL intensity",
    "kl_bg": "KL background",
}

_DARK2 = plt.get_cmap("Dark2").colors
_TRAIN_KW = {"linestyle": "-"}
_VAL_KW = {"linestyle": "--", "alpha": 0.7}

# per-model aggregates: epoch-mean frames for each split
Model = namedtuple("Model", ["label", "train", "val", "color"])


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training loss / ELBO / KL-component curves"
    )
    parser.add_argument("--plot-cfg", required=True, help="plot_cfg.yaml")
    parser.add_argument("--out-dir", default=".", help="Directory for figures")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_history(run_dir: Path) -> pl.DataFrame | None:
    """Read a run's plots/loss_history.csv (one row per epoch, wide columns)."""
    rp = run_dir / "run_paths.yaml"
    if not rp.is_file():
        logger.warning("No run_paths.yaml in %s; skipping", run_dir)
        return None
    meta = load_config(rp)

    plots_dir = meta.get("plots_dir")
    if not plots_dir:  # derive from output_root / log_dir when absent
        root = meta.get("output_root")
        log_dir = meta.get("log_dir") or meta.get("wandb", {}).get("log_dir")
        if root:
            plots_dir = str(Path(root) / "plots")
        elif log_dir:
            plots_dir = str(Path(log_dir).parent / "plots")

    csv = Path(plots_dir) / "loss_history.csv" if plots_dir else None
    if csv is None or not csv.is_file():  # fallback: search under the run dir
        found = sorted(run_dir.glob("**/loss_history.csv"))
        csv = found[-1] if found else None
    if csv is None:
        logger.warning("No loss_history.csv for %s; skipping", run_dir)
        return None
    return pl.read_csv(csv)


def _split_frame(hist: pl.DataFrame, split: str) -> pl.DataFrame | None:
    """Per-epoch frame for one split, renaming `{split}_{term}` -> term."""
    present = [m for m in METRICS if f"{split}_{m}" in hist.columns]
    if not present:
        return None
    return hist.select(
        pl.col("epoch"),
        *[pl.col(f"{split}_{m}").alias(m) for m in present],
    ).sort("epoch")


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _line(ax, frame, metric, **kwargs):
    """Plot metric vs epoch from a per-epoch frame, dropping null rows."""
    if frame is not None and metric in frame.columns:
        sub = frame.select("epoch", metric).drop_nulls()
        if sub.height:
            ax.plot(sub["epoch"], sub[metric], **kwargs)
            return True
    return False


def _plot_per_model(model: Model, out_path: Path):
    """2x3 grid of every component for one model: train solid, val dashed."""
    present = [
        m for m in METRICS
        if (model.train is not None and m in model.train.columns)
        or (model.val is not None and m in model.val.columns)
    ]
    if not present:
        return
    ncols = 3
    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    axs = axes.ravel()
    for ax, m in zip(axs, present):
        _line(ax, model.train, m, color="tab:blue", label="train", **_TRAIN_KW)
        _line(ax, model.val, m, color="tab:orange", label="val", **_VAL_KW)
        ax.set_title(METRICS[m])
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axs[len(present):]:
        ax.axis("off")

    fig.suptitle(f"Loss components: {model.label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_compare(metric: str, models: list[Model], out_path: Path):
    """One component across all models: color per model, train solid/val dashed."""
    fig, ax = plt.subplots(figsize=(7, 5))
    drawn = False
    for m in models:
        got = _line(ax, m.train, metric, color=m.color, label=m.label, **_TRAIN_KW)
        _line(ax, m.val, metric, color=m.color, **_VAL_KW)
        drawn = drawn or got
    if not drawn:
        plt.close(fig)
        return

    ax.set_title(f"{METRICS[metric]}  (solid=train, dashed=val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    args = parse_args()
    cfg = load_config(args.plot_cfg)

    runs = sorted(cfg.get("runs", {}).values(), key=lambda v: str(v["label"]))
    models: list[Model] = []
    for v in runs:
        hist = _load_history(Path(v["path"]))
        if hist is None:
            continue
        color = _DARK2[len(models) % len(_DARK2)]
        models.append(
            Model(str(v["label"]), _split_frame(hist, "train"), _split_frame(hist, "val"), color)
        )

    if not models:
        raise SystemExit("No loss_history.csv found for any run")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-model grids
    for m in models:
        _plot_per_model(m, out_dir / f"loss_{re.sub(r'[^0-9A-Za-z._-]+', '_', m.label)}.png")

    # inter-model comparison, one figure per component
    for metric in METRICS:
        _plot_compare(metric, models, out_dir / f"loss_compare_{metric}.png")


if __name__ == "__main__":
    main()
