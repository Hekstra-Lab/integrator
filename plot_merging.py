"""Merging-statistic curves vs resolution: each model against DIALS.

Reads a plot-config (same schema as plot_peaks.py), finds each run's DIALS
`merged.html`, and draws a 2x2 grid of merging statistics vs resolution shell
for that model, overlaying the DIALS reference in red:

    CChalf (CC1/2) | Rpim
    CCanom         | I/sigI

One figure per model (model vs DIALS only). Each model's epochs are drawn as
one line each, colored by a cubehelix epoch ramp (matching refltorch) with an
epoch colorbar. The DIALS reference is read from `reference_data.merge`; stats
are parsed from the DIALS `merged.html` resolution-bin table.

Usage:
    uv run python plot_merging.py --plot-cfg plot_cfg.yaml --out-dir merging/
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# stat key -> y-axis label, in 2x2 order (matches refltorch)
STATS = [
    ("cchalf", "CChalf"),
    ("rpim", "Rpim"),
    ("ccanom", "CCanom"),
    ("isigi", "I/sigI"),
]


def _cubehelix(n: int):
    """refltorch's epoch palette: (cmap, n discrete colors) via seaborn."""
    import seaborn as sns

    cmap = sns.cubehelix_palette(
        start=0.5, rot=-0.55, dark=0, light=0.8, as_cmap=True
    )
    return cmap, cmap(np.linspace(0.0, 1.0, max(n, 1)))


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
# merged.html parsing
# --------------------------------------------------------------------------- #
def _pick_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """The resolution-bin table (has a Resolution column and a CC column)."""
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("resolution" in c for c in cols) and any("cc" in c for c in cols):
            return t
    return tables[1] if len(tables) > 1 else tables[0]


def _find_col(df: pd.DataFrame, *subs: str):
    """First column whose name contains all `subs` (case-insensitive)."""
    for c in df.columns:
        s = str(c).lower()
        if all(sub.lower() in s for sub in subs):
            return c
    return None


def _floats(series) -> list[float]:
    """DIALS marks significant bins with '*'; strip it and coerce to float."""
    out = []
    for v in series.tolist():
        if isinstance(v, str):
            try:
                out.append(float(v.strip().strip("*")))
            except ValueError:
                out.append(np.nan)
        else:
            out.append(float(v))
    return out


def _parse_merged_html(path: Path) -> dict | None:
    """Parse a DIALS merged.html into per-shell stat lists, or None on failure."""
    try:
        tables = pd.read_html(path)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    return _stats_from_table(_pick_table(tables))


def _stats_from_table(df: pd.DataFrame) -> dict | None:
    """Extract per-shell (resolution, cchalf, ccanom, rpim, isigi) from a table."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]

    cols = {
        "resolution": _find_col(df, "resolution"),
        "cchalf": _find_col(df, "cc", "½") or _find_col(df, "cc", "half"),
        "ccanom": _find_col(df, "ccano") or _find_col(df, "ccanom"),
        "rpim": _find_col(df, "rpim"),
        "isigi": _find_col(df, "i/σ") or _find_col(df, "mean i") or _find_col(df, "i/sig"),
    }
    if cols["resolution"] is None:
        logger.warning("merged.html table has no resolution column")
        return None

    resolution = [str(x) for x in df[cols["resolution"]].tolist()]
    keep = [not r.strip().lower().startswith("overall") for r in resolution]
    stats: dict[str, list] = {
        "resolution": [r for r, k in zip(resolution, keep, strict=False) if k]
    }
    for key in ("cchalf", "ccanom", "rpim", "isigi"):
        c = cols[key]
        vals = _floats(df[c]) if c is not None else [np.nan] * len(df)
        stats[key] = [v for v, k in zip(vals, keep, strict=False) if k]
    return stats


def _parse_merging_stats_csv(path: Path) -> dict | None:
    """Read the canonical merging_stats.csv both arms emit.

    Columns are locked across the monochromatic and polychromatic pipelines:
    bin, d_max, d_min, n_obs, n_unique, cc_half, cc_anom, r_pim,
    i_over_sigma. They are mapped onto the same keys the merged.html parser
    produces, so everything downstream is unchanged.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
        logger.warning("Could not read %s: %s", path, e)
        return None

    rename = {
        "cc_half": "cchalf",
        "cc_anom": "ccanom",
        "r_pim": "rpim",
        "i_over_sigma": "isigi",
    }
    missing = [c for c in ("d_max", "d_min") if c not in df.columns]
    if missing:
        logger.warning("%s has no %s column", path.name, ", ".join(missing))
        return None

    stats: dict[str, list] = {
        "resolution": [
            f"{hi:.2f} - {lo:.2f}"
            for hi, lo in zip(df["d_max"], df["d_min"], strict=False)
        ],
        "d": [
            _shell_midpoint(hi, lo)
            for hi, lo in zip(df["d_max"], df["d_min"], strict=False)
        ],
    }
    for src, key in rename.items():
        stats[key] = (
            [float(v) if pd.notna(v) else np.nan for v in df[src]]
            if src in df.columns
            else [np.nan] * len(df)
        )
    return stats


def _shell_midpoint(d_max: float, d_min: float) -> float:
    """Resolution at the middle of a shell, in 1/d^2 (how shells are cut)."""
    try:
        s_hi, s_lo = 1.0 / float(d_max) ** 2, 1.0 / float(d_min) ** 2
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")
    return float(1.0 / np.sqrt((s_hi + s_lo) / 2.0))


def _shell_d_from_labels(stats: dict) -> list[float]:
    """Numeric shell centres parsed out of "79.63 - 2.97" style labels."""
    out = []
    for label in stats.get("resolution", []):
        nums = re.findall(r"[0-9]*\.?[0-9]+", str(label))
        out.append(
            _shell_midpoint(float(nums[0]), float(nums[1]))
            if len(nums) >= 2
            else float("nan")
        )
    return out


def _parse_stats(path: Path) -> dict | None:
    """Read either the canonical CSV or a DIALS merged.html."""
    stats = (
        _parse_merging_stats_csv(path)
        if path.suffix == ".csv"
        else _parse_merged_html(path)
    )
    if stats is not None and "d" not in stats:
        stats["d"] = _shell_d_from_labels(stats)
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
    """Draw one statistic on `ax`: a line per epoch, plus a red DIALS line."""
    for (_epoch, st), color in zip(epoch_stats, colors, strict=False):
        ax.plot(_shell_axis(st, key), st[key], color=color, lw=1)
    if ref_stats is not None:
        ax.plot(_shell_axis(ref_stats, key), ref_stats[key],
                color="red", lw=2, label="DIALS")
        ax.legend(fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("resolution (Å)")
    ax.grid(alpha=0.5)

    newest = epoch_stats[-1][1]
    d = newest.get("d")
    if d and len(d) == len(res) and not all(np.isnan(v) for v in d):
        ticks = [1.0 / v**2 for v in d if v and not np.isnan(v)]
        labels = [f"{v:.2f}" for v in d if v and not np.isnan(v)]
        step = max(1, len(ticks) // 8)  # a dense binning would crowd the axis
        ax.set_xticks(ticks[::step])
        ax.set_xticklabels(labels[::step], rotation=55, fontsize=6, ha="right")
    else:
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(res, rotation=55, fontsize=6, ha="right")


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
    cmap, colors = _cubehelix(len(epochs))
    res = epoch_stats[-1][1]["resolution"]  # newest epoch's shell labels

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, (key, ylabel) in zip(axes.ravel(), STATS, strict=False):
        _draw_stat(ax, key, ylabel, res, epoch_stats, colors, ref_stats)
    _epoch_colorbar(fig, axes, cmap, epochs)

    fig.suptitle(f"Merging statistics: {label} vs DIALS")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (%d epochs)", out_path, len(epochs))


def _plot_model_individual(label, epoch_stats, ref_stats, out_dir, slug):
    """Save each statistic as its own figure, each with its own colorbar."""
    epochs = [e for e, _ in epoch_stats]
    cmap, colors = _cubehelix(len(epochs))
    res = epoch_stats[-1][1]["resolution"]

    for key, ylabel in STATS:
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        _draw_stat(ax, key, ylabel, res, epoch_stats, colors, ref_stats)
        _epoch_colorbar(fig, ax, cmap, epochs)
        ax.set_title(f"{ylabel}: {label} vs DIALS")
        out_path = out_dir / f"merging_{slug}_{key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out_path)


def main():
    args = parse_args()
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
