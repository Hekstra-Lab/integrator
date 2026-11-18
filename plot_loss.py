"""Plot training loss curves: ELBO, NLL, and the individual KL components.

Reads a plot-config (same schema as plot_peaks.py), finds each run's
`plots/loss_history.csv` (per-epoch, columns `epoch` and `{split}_{term}`),
and draws:

  * per-model     -> one 2x3 grid of every component (train solid, val dashed)
  * inter-model   -> one figure per component overlaying all models

Components: loss (ELBO), nll, kl (total), kl_prf, kl_i, kl_bg.

Usage:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python plot_loss.py --plot-cfg plot_cfg.yaml --out-dir loss/
"""

import argparse
import logging
import math
import re
import sys
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import yaml
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "figures"))
import figstyle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# loss-trace column -> display label, in grid order
# The ELBO and its parts, named by what each one is. `kl` is the sum of the
# other three, not a separate term.
METRICS = {
    "loss": r"$-\mathrm{ELBO}$",
    "nll": "Poisson NLL",
    "kl": r"$D_\mathrm{KL}$, total",
    "kl_prf": r"$D_\mathrm{KL}$, profile",
    "kl_i": r"$D_\mathrm{KL}$, intensity",
    "kl_bg": r"$D_\mathrm{KL}$, background",
}

_TRAIN_KW = {"linestyle": "-"}
_VAL_KW = {"linestyle": (0, (3, 2)), "alpha": 0.85}

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
        nrows,
        ncols,
        figsize=figstyle.figsize("full", aspect=0.26 * nrows),
        squeeze=False,
        layout="constrained",
    )
    axs = axes.ravel()
    for ax, m in zip(axs, present):
        split = figstyle.arm_colors(["train", "val"])
        _line(ax, model.train, m, color=split["train"], label="train",
              **_TRAIN_KW)
        _line(ax, model.val, m, color=split["val"], label="validation",
              **_VAL_KW)
        ax.set_ylabel(METRICS[m])
        ax.set_xlabel("epoch")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=7)
    for ax in axs[len(present):]:
        ax.axis("off")

    fig.suptitle(f"Loss components: {model.label}")
    # `layout="constrained"` already handles this
    figstyle.save(fig, Path(out_path).with_suffix(""))
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_compare(metric: str, models: list[Model], out_path: Path):
    """One component across all models: color per model, train solid/val dashed."""
    fig, ax = plt.subplots(
        figsize=figstyle.figsize("half", aspect=0.8), layout="constrained")
    drawn = False
    for m in models:
        got = _line(ax, m.train, metric, color=m.color, label=m.label, **_TRAIN_KW)
        _line(ax, m.val, metric, color=m.color, **_VAL_KW)
        drawn = drawn or got
    if not drawn:
        plt.close(fig)
        return

    ax.set_xlabel("epoch")
    # the quantity, not the column name it is stored under
    ax.set_ylabel(METRICS[metric])
    ax.grid(axis="x", visible=False)
    # Which line is train and which is validation belongs in the legend; it
    # was written into the title as `(solid=train, dashed=val)`, which is a
    # legend set as prose.
    handles, labels = ax.get_legend_handles_labels()
    ink = figstyle.roles()["ink"]
    handles += [
        Line2D([], [], color=ink, **_TRAIN_KW),
        Line2D([], [], color=ink, **_VAL_KW),
    ]
    labels += ["train", "validation"]
    ax.legend(handles, labels, frameon=False, fontsize=8)
    # `layout="constrained"` already handles this
    figstyle.save(fig, Path(out_path).with_suffix(""))
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main():
    args = parse_args()
    figstyle.use()
    cfg = load_config(args.plot_cfg)

    runs = sorted(cfg.get("runs", {}).values(), key=lambda v: str(v["label"]))
    # the witch categorical palette holds 8, ordered most distinct first
    palette = list(figstyle.arm_colors(
        [str(v["label"]) for v in runs]).values())
    models: list[Model] = []
    for v in runs:
        hist = _load_history(Path(v["path"]))
        if hist is None:
            continue
        color = palette[len(models) % len(palette)]
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
