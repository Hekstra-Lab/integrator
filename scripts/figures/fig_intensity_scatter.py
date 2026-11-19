"""Predicted intensities, model against model.

`plot_compare.py` draws a model against the reference integrator, which
answers "is this port faithful". This answers a different question: two
models trained on the same reflections, what do they disagree about, and
does the disagreement depend on resolution.

Three panels per pair:

- **scatter**, `qi_mean` against `qi_mean`, log-log, with the identity
  line. Log because the intensities span four decades and a linear axis
  shows only the brightest hundred reflections.
- **ratio against resolution**, which is where a gate defect should show.
  A gate that collapses at high resolution takes `sum_i r_i c_i` down with
  it, so the ratio drifts in the outer shells while the bright end looks
  fine. Drawn as the median and the interquartile band per shell, because
  the per-reflection ratio is heavy-tailed at the weak end and a mean
  would be carried by a handful of points.
- **correlation per shell**, the cheap summary of the same thing.

**What this cannot separate.** The arms are independently trained and
training here is unseeded, so a difference is architecture AND run-to-run
variation together. A systematic trend with resolution is attributable to
the model; scatter width alone is not. Nothing here is a downstream
result: two models can disagree about every intensity and merge to the
same structure factors.

Reads the long table from `scripts/figures/tidy_intensities.py`, one row
per (method, refl_id) with `qi_mean`, `qi_var` and `d`.

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_intensity_scatter.py \\
            --data docs/figures/data/intensities_1118.csv
"""

from __future__ import annotations

import argparse
import itertools
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
N_SHELLS = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path,
                   default=DATA / "intensities_1118.csv")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--prefix", default="intensity")
    p.add_argument("--max-points", type=int, default=20000,
                   help="points drawn in the scatter; the statistics use "
                        "every reflection regardless")
    return p.parse_args()


def shells(d: np.ndarray, n: int = N_SHELLS) -> np.ndarray:
    """Equal-count bins in 1/d^2, which is how resolution shells are cut."""
    s = 1.0 / np.maximum(d, 1e-9) ** 2
    edges = np.quantile(s, np.linspace(0, 1, n + 1))
    edges[-1] += 1e-9
    return np.clip(np.digitize(s, edges) - 1, 0, n - 1)


def paired(df: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """The reflections both arms predicted, joined on the reflection id."""
    left = df.filter(pl.col("method") == a).select(
        "refl_id", "d", pl.col("qi_mean").alias("x"))
    right = df.filter(pl.col("method") == b).select(
        "refl_id", pl.col("qi_mean").alias("y"))
    return left.join(right, on="refl_id", how="inner")


def draw_pair(df, a, b, out_dir, prefix, max_points) -> Path | None:
    j = paired(df, a, b).drop_nulls(["x", "y", "d"])
    # both positive, so the log axes and the ratio are defined
    j = j.filter((pl.col("x") > 0) & (pl.col("y") > 0))
    if j.height < 100:
        logger.warning("  %s vs %s: only %d shared reflections", a, b,
                       j.height)
        return None
    x = j["x"].to_numpy()
    y = j["y"].to_numpy()
    d = j["d"].to_numpy()
    idx = shells(d)

    fig, axes = plt.subplots(
        1, 3, figsize=figstyle().figsize("full", aspect=0.34),
        layout="constrained")
    pal = figstyle().arm_colors([a, b])

    # the scatter is thinned for drawing only
    take = (np.random.default_rng(0).choice(len(x), max_points, replace=False)
            if len(x) > max_points else np.arange(len(x)))
    ax = axes[0]
    ax.scatter(x[take], y[take], s=2, alpha=0.25, lw=0,
               color=pal[a], rasterized=True)
    lo = float(max(np.percentile(x, 0.1), np.percentile(y, 0.1), 1e-3))
    hi = float(max(np.percentile(x, 99.9), np.percentile(y, 99.9)))
    ax.plot([lo, hi], [lo, hi], **figstyle().reference_style())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(short(a))
    ax.set_ylabel(short(b))

    # ratio by shell: median and interquartile band
    ratio = y / x
    mid, q1, q3, dd = [], [], [], []
    for k in range(N_SHELLS):
        sel = idx == k
        if sel.sum() < 20:
            continue
        mid.append(float(np.median(ratio[sel])))
        q1.append(float(np.percentile(ratio[sel], 25)))
        q3.append(float(np.percentile(ratio[sel], 75)))
        dd.append(float(np.median(1.0 / np.maximum(d[sel], 1e-9) ** 2)))
    ax = axes[1]
    ax.fill_between(dd, q1, q3, color=pal[b], alpha=0.25, lw=0)
    ax.plot(dd, mid, lw=1.2, color=pal[b])
    ax.axhline(1.0, **figstyle().reference_style())
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/d^2$ (\AA$^{-2}$)")
    ax.set_ylabel(f"{short(b)} / {short(a)}")

    # correlation of the logs per shell: the intensities are log-spread, so
    # a correlation on the raw values is set by the brightest few
    rho, dd2 = [], []
    for k in range(N_SHELLS):
        sel = idx == k
        if sel.sum() < 20:
            continue
        rho.append(float(np.corrcoef(np.log(x[sel]), np.log(y[sel]))[0, 1]))
        dd2.append(float(np.median(1.0 / np.maximum(d[sel], 1e-9) ** 2)))
    ax = axes[2]
    ax.plot(dd2, rho, lw=1.2, color=pal[b], marker="o", ms=3)
    ax.set_xlabel(r"$1/d^2$ (\AA$^{-2}$)")
    ax.set_ylabel(r"corr$(\log I_a, \log I_b)$")
    ax.set_ylim(min(0.0, min(rho) - 0.05), 1.02)

    path = out_dir / f"{prefix}_{short(a)}_vs_{short(b)}"
    figstyle().save(fig, path)
    plt.close(fig)
    logger.info("  %-22s n=%d, median ratio %.3f, outer-shell ratio %.3f, "
                "corr %.3f -> %.3f", f"{short(a)}/{short(b)}", j.height,
                float(np.median(ratio)), mid[-1] if mid else float("nan"),
                rho[0] if rho else float("nan"),
                rho[-1] if rho else float("nan"))
    return path


def short(name: str) -> str:
    return name.replace("hewl1118_", "").replace("hewl9b7c_", "")


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"no {args.data}. It comes from "
            "scripts/figures/tidy_intensities.py"
        )
    df = pl.read_csv(args.data, infer_schema_length=None)
    missing = {"method", "refl_id", "qi_mean", "d"} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.data} has no {sorted(missing)}")
    figstyle().use()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms = sorted(df["method"].unique().to_list())
    if len(arms) < 2:
        raise SystemExit(f"need two arms to compare, found {arms}")
    n = 0
    for a, b in itertools.combinations(arms, 2):
        if draw_pair(df, a, b, args.out_dir, args.prefix,
                     args.max_points) is not None:
            n += 1
    logger.info("%d figures from %d arms", n, len(arms))


if __name__ == "__main__":
    main()
