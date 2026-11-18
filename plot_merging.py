"""Merging-statistic curves vs resolution: each model against DIALS.

Reads a plot-config (same schema as plot_peaks.py), finds each run's DIALS
`merged.html`, and draws a 2x2 grid of merging statistics vs resolution shell
for that model, overlaying the reference integrator:

    CChalf (CC1/2) | Rpim
    CCanom         | I/sigI

One figure per model (model vs DIALS only). Each model's epochs are drawn as
one line each, on the witch sequential ramp with an epoch colorbar, and the
reference takes a categorical colour so it cannot be read as a checkpoint. The DIALS reference is read from `reference_data.merge`; stats
are parsed from the DIALS `merged.html` resolution-bin table.

Usage:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python plot_merging.py --plot-cfg plot_cfg.yaml --out-dir merging/
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

from integrator.utils import metrics

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "figures"))
import figstyle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# stat key -> y-axis label, in 2x2 order (matches refltorch)
# Multi-letter names upright, single-letter quantities italic, so the
# figure labels and the table headers in docs/tables/ agree.
STATS = [
    ("cchalf", r"$\mathrm{CC}_{1/2}$"),
    ("rpim", r"$R_\mathrm{pim}$"),
    ("ccanom", r"$\mathrm{CC}_\mathrm{anom}$"),
    ("isigi", r"$I/\sigma_I$"),
]


def _epoch_palette(n: int):
    """`(cmap, n colors)` over training, from `figstyle`."""
    return figstyle.epoch_ramp(n)


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merging statistics vs resolution: each model vs DIALS"
    )
    parser.add_argument("--plot-cfg", required=True, help="plot_cfg.yaml")
    parser.add_argument("--out-dir", default=".", help="Directory for figures")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Plot only this epoch per run (default: all epochs, epoch-colored)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# reading the per-shell statistics
# --------------------------------------------------------------------------- #
# The three forms a run can be scored through -- the canonical
# `merging_stats.csv`, a DIALS `merged.html`, and the text table
# `dials.merge` prints -- are parsed by `integrator.utils.metrics`, which is
# also what `tidy_mono.py` and the chain's own readers use. They used to be
# parsed here with a private set of column-matching rules, and a second set
# in `tidy_mono.py`, so the same run could be described two ways.
#
# `metrics` returns one row per shell with canonical names; the plotting
# below wants a column per statistic, so this is the one place that
# transposes, and the stat keys below are this figure's own axis names.
_STAT_KEYS = {
    "cchalf": "cc_half",
    "ccanom": "cc_anom",
    "rpim": "r_pim",
    "isigi": "i_over_sigma",
}


def _parse_stats(path: Path) -> dict | None:
    """Per-shell statistics from any of the three forms, as column lists."""
    rows = metrics.shells_from_any(path)
    if not rows:
        logger.warning("No per-shell statistics in %s", path)
        return None
    stats: dict[str, list] = {
        "resolution": [r.get("resolution", "") for r in rows],
        "d": [float(r.get("d", float("nan"))) for r in rows],
    }
    for key, source in _STAT_KEYS.items():
        stats[key] = [float(r.get(source, float("nan"))) for r in rows]
    return stats


def _model_merged_htmls(run_dir: Path, epoch: int | None) -> list[tuple[int, Path]]:
    """All (epoch, merged.html) for a run, sorted by epoch (or just `epoch`)."""
    rp = run_dir / "run_paths.yaml"
    if not rp.is_file():
        logger.warning("No run_paths.yaml in %s; skipping", run_dir)
        return []
    pred_dir = Path(load_config(rp)["predictions_dir"])
    # the canonical CSV wins when present; merged.html keeps older runs working
    found = []
    for pattern in ("**/merging_stats.csv", "**/merged.html"):
        for h in pred_dir.glob(pattern):
            m = re.search(r"epoch_(\d+)", str(h))
            found.append((int(m.group(1)) if m else 0, h))
        if found:
            break
    if not found:
        logger.warning(
            "No merging_stats.csv or merged.html under %s; skipping", pred_dir
        )
        return []
    found.sort(key=lambda t: t[0])
    if epoch is not None:
        match = [t for t in found if t[0] == epoch]
        return match or found[-1:]
    return found


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _shell_axis(stats: dict, key: str):
    """x values for one series: shell centres in 1/d^2, else the shell index.

    Plotting against real resolution rather than shell number is what lets
    two runs with different binning share an axis -- DIALS picks its own
    shells and careless picks others, and on an index axis position 5 would
    mean a different resolution in each.
    """
    d = stats.get("d")
    y = stats[key]
    if d and len(d) == len(y) and not all(np.isnan(v) for v in d):
        return [1.0 / v**2 if v and not np.isnan(v) else np.nan for v in d]
    return list(range(len(y)))


def _draw_stat(ax, key, ylabel, res, epoch_stats, colors, ref_stats):
    """Draw one statistic on `ax`: a line per epoch, plus the reference."""
    for (_epoch, st), c in zip(epoch_stats, colors, strict=False):
        ax.plot(_shell_axis(st, key), st[key], color=c, lw=0.9)
    if ref_stats is not None:
        ax.plot(_shell_axis(ref_stats, key), ref_stats[key],
                label="DIALS", **figstyle.reference_style())
        ax.legend(frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"resolution (\AA)")

    newest = epoch_stats[-1][1]
    d = newest.get("d")
    if d and len(d) == len(res) and not all(np.isnan(v) for v in d):
        ticks = [1.0 / v**2 for v in d if v and not np.isnan(v)]
        labels = [f"{v:.2f}" for v in d if v and not np.isnan(v)]
        step = max(1, len(ticks) // 8)  # a dense binning would crowd the axis
        ax.set_xticks(ticks[::step])
        ax.set_xticklabels(labels[::step], rotation=55, ha="right")
    else:
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(res, rotation=55, ha="right")
    ax.grid(axis="x", visible=False)


def _epoch_colorbar(fig, ax, cmap, epochs):
    """Attach an epoch colorbar to `ax` (a single axis or an axis array)."""
    if len(epochs) <= 1:
        return
    norm = mpl.colors.Normalize(vmin=min(epochs), vmax=max(epochs))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label("epoch")


def _plot_model(label, epoch_stats, ref_stats, out_path):
    """2x2 grid of merging stats vs resolution, one shared epoch colorbar."""
    epochs = [e for e, _ in epoch_stats]
    cmap, colors = _epoch_palette(len(epochs))
    res = epoch_stats[-1][1]["resolution"]  # newest epoch's shell labels

    fig, axes = plt.subplots(
        2, 2, figsize=figstyle.figsize("full", aspect=0.82),
        layout="constrained")
    for ax, (key, ylabel) in zip(axes.ravel(), STATS, strict=False):
        _draw_stat(ax, key, ylabel, res, epoch_stats, colors, ref_stats)
    _epoch_colorbar(fig, axes, cmap, epochs)

    figstyle.save(fig, out_path.with_suffix(""))
    plt.close(fig)
    logger.info("Saved %s (%d epochs)", out_path.with_suffix(""), len(epochs))


def _plot_model_individual(label, epoch_stats, ref_stats, out_dir, slug):
    """Save each statistic as its own figure, each with its own colorbar."""
    epochs = [e for e, _ in epoch_stats]
    cmap, colors = _epoch_palette(len(epochs))
    res = epoch_stats[-1][1]["resolution"]

    for key, ylabel in STATS:
        fig, ax = plt.subplots(
            figsize=figstyle.figsize("half", aspect=0.85),
            layout="constrained")
        _draw_stat(ax, key, ylabel, res, epoch_stats, colors, ref_stats)
        _epoch_colorbar(fig, ax, cmap, epochs)
        out_path = out_dir / f"merging_{slug}_{key}"
        figstyle.save(fig, out_path)
        plt.close(fig)
        logger.info("Saved %s", out_path)


def main():
    args = parse_args()
    figstyle.use()
    cfg = load_config(args.plot_cfg)

    ref = (cfg.get("reference_data") or {}).get("merge")
    ref_stats = _parse_stats(Path(ref)) if ref else None
    if ref and ref_stats is None:
        logger.warning("Could not parse DIALS reference merged.html")

    runs = sorted(cfg.get("runs", {}).values(), key=lambda v: str(v["label"]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in runs:
        label = str(v["label"])
        htmls = _model_merged_htmls(Path(v["path"]), args.epoch)
        epoch_stats = []
        for epoch, html in htmls:
            st = _parse_stats(html)
            if st is not None:
                epoch_stats.append((epoch, st))
        if not epoch_stats:
            continue
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", label)
        _plot_model(label, epoch_stats, ref_stats, out_dir / f"merging_{slug}.png")
        _plot_model_individual(label, epoch_stats, ref_stats, out_dir, slug)


if __name__ == "__main__":
    main()
