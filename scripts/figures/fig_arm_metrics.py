"""What each arm is worth after scaling, merging and refinement.

One panel per metric, from `docs/figures/data/arm_metrics.csv`. Both
estimators are drawn for every arm: `rb` is the Rao-Blackwellized profile fit
computed at export and `model` is the integrator's own posterior. One
training gives both, `dials.merge` weights by 1/sigma^2, and which one wins is
not monotone across arms, so showing one can invert the conclusion.

The anomalous correlation and R-free panels share no axis with the others and
are drawn separately rather than on a second y-axis.

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_arm_metrics.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import figstyle  # noqa: E402
from figtex import read  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import label, ordered, palette  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "m9"
DX = 0.17

PANELS = (
    ("i_over_sigma", r"$I/\sigma$"),
    ("cc_anom", "anomalous correlation"),
    ("r_free_f", r"$R_\mathrm{free}$ (from $F$)"),
    ("max_peak", r"tallest anomalous peak ($\sigma$)"),
)


def competitive(df: pl.DataFrame) -> list[str]:
    """Arms whose profile has not collapsed, by R-free under 0.25.

    The collapsed arms sit at R-free 0.39 and 13 sigma of peak, far enough out
    that including them in the limits would squash the 0.0028 that separates
    the three arms the figure is actually comparing. They are drawn at the
    axis edge with their value printed instead.
    """
    r = df.filter((pl.col("metric") == "r_free_f") & (pl.col("value") < 0.25))
    return r["method"].unique().to_list()


def panel(ax, df: pl.DataFrame, metric: str, ylabel: str, methods, pal,
          inscale, xlabels: bool) -> None:
    roles = figstyle.roles()
    rows = {(r["method"], r["estimator"]): r["value"]
            for r in df.filter(pl.col("metric") == metric).to_dicts()}
    x = np.arange(len(methods), dtype=float)

    vals = [v for (m, _), v in rows.items() if m in inscale]
    lo, hi = min(vals), max(vals)
    pad = 0.18 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
    lo, hi = lo - pad, hi + pad

    for i, m in enumerate(methods):
        for est, off, fill in (("rb", -DX, True), ("model", DX, False)):
            v = rows.get((m, est))
            if v is None:
                continue
            off_scale = v < lo or v > hi
            ax.plot(
                x[i] + off,
                min(max(v, lo), hi) if off_scale else v,
                marker=("v" if v < lo else "^") if off_scale
                else ("o" if fill else "s"),
                markersize=4,
                color=pal[m],
                markerfacecolor=pal[m] if (fill and not off_scale) else "none",
                ls="",
            )
            if off_scale:
                ax.annotate(
                    f"{v:.3g}",
                    (x[i] + off, min(max(v, lo), hi)),
                    xytext=(0, 7 if v < lo else -11),
                    textcoords="offset points",
                    ha="center",
                    color=roles["muted"],
                )
        pair = [rows.get((m, "rb")), rows.get((m, "model"))]
        if all(v is not None for v in pair) and m in inscale:
            ax.plot([x[i] - DX, x[i] + DX], pair, color=pal[m], lw=0.7, zorder=0)
    if metric == "cc_anom" and lo < 0 < hi:
        ax.axhline(0.0, color=roles["rule"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [label(m) for m in methods] if xlabels else [""] * len(methods),
        rotation=35,
        ha="right",
    )
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def main() -> None:
    df = read(DATA / "arm_metrics.csv")
    methods = ordered(df["method"].unique().to_list())
    pal = palette(methods)
    roles = figstyle.roles()

    figstyle.use()
    fig, axes = plt.subplots(
        2, 2, figsize=figstyle.figsize("full", aspect=0.78), layout="constrained"
    )
    inscale = competitive(df)
    for k, (ax, (metric, ylabel)) in enumerate(
        zip(axes.ravel(), PANELS, strict=True)
    ):
        panel(ax, df, metric, ylabel, methods, pal, inscale, xlabels=k >= 2)
    fig.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=4, color=roles["ink"],
                   label="Rao-Blackwellized fit at export"),
            Line2D([], [], ls="", marker="s", ms=4, color=roles["ink"],
                   markerfacecolor="none", label="the integrator's posterior"),
        ],
        loc="outside upper center",
        ncols=2,
        frameon=False,
        handletextpad=0.4,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    paths = figstyle.save(fig, OUT / "arm_metrics")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
