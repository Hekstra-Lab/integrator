"""Does a profile model follow the spot shape of the reflections beside it?

Two panels, both from `docs/figures/data/`:

left    the axial orientation difference between a weak reflection's profile
        and the mean profile of the k bright reflections nearest it (filled),
        against the same quantity for bright reflections far away on the
        detector (open). A model that carries no position information scores
        the same in both. The null is 45 degrees, because the second-moment
        axis is defined modulo 180. The bar at each arm is its own ceiling:
        two disjoint draws of local bright reflections, which is how close
        two estimates of the SAME local shape get.
right   the axis ratio of the weak-reflection profiles, against the band the
        same models fit to their own bright reflections, where the counts
        constrain the shape.

Both metrics are translation invariant -- the orientation comes from the
central second moment and the shape agreement from the modulus of the Fourier
transform with the DC term dropped -- because the first version of this
measurement compared profiles by cosine similarity and read back 0.42 to 0.65,
which is the chance level for nonnegative vectors, not a result.

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_profile_shape.py
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
from arms import DTHETA_NULL, SUPPORTED_RATIO, label, ordered, palette  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "m9"
DX = 0.2


def orientation_panel(ax, shape: pl.DataFrame, methods, pal) -> None:
    roles = figstyle.roles()
    d = shape.filter(pl.col("metric") == "dtheta_deg")
    rows = {(r["method"], r["reference"]): r for r in d.to_dicts()}
    x = np.arange(len(methods), dtype=float)

    ax.axhline(DTHETA_NULL, color=roles["muted"], lw=0.8, ls=(0, (4, 2)), zorder=0)
    ax.text(
        -0.5,
        DTHETA_NULL + 1.5,
        "unrelated orientations",
        ha="left",
        va="bottom",
        color=roles["muted"],
    )
    for i, m in enumerate(methods):
        for ref, off, fill in (("local", -DX, True), ("far", DX, False)):
            r = rows.get((m, ref))
            if r is None:
                continue
            ax.errorbar(
                x[i] + off,
                r["value"],
                yerr=[[r["value"] - r["p25"]], [r["p75"] - r["value"]]],
                marker="o" if fill else "s",
                markersize=4,
                color=pal[m],
                markerfacecolor=pal[m] if fill else "none",
                ls="",
                capsize=2,
                elinewidth=0.8,
                zorder=3,
            )
        c = rows.get((m, "ceiling"))
        if c is not None:
            ax.plot(
                [x[i] - 0.34, x[i] + 0.34],
                [c["value"]] * 2,
                color=roles["ink"],
                lw=1.0,
                zorder=2,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylabel(r"$\Delta\theta$ to the reference profile (deg)")
    ax.set_ylim(0, 92)
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=4, color=roles["ink"],
                   label="bright reflections nearby"),
            Line2D([], [], ls="", marker="s", ms=4, color=roles["ink"],
                   markerfacecolor="none", label="bright reflections far away"),
            Line2D([], [], lw=1.0, color=roles["ink"], label="ceiling"),
        ],
        loc="upper left",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.2,
    )


def width_panel(ax, aniso: pl.DataFrame, methods, pal) -> None:
    """Axis ratio of the individual profiles, bright against weak.

    Bright reflections are where the counts constrain the shape, so a model
    whose weak profiles leave that band is wider than anything the data
    supports. The ratio is taken on each profile separately rather than on a
    mean profile: averaging profiles of genuinely different shapes broadens
    the mean even when every member is re-centred first.
    """
    roles = figstyle.roles()
    rows = {(r["method"], r["population"]): r for r in aniso.to_dicts()}
    x = np.arange(len(methods), dtype=float)
    lo, hi = SUPPORTED_RATIO
    ax.axhspan(lo, hi, color=roles["grid"], zorder=0)
    ax.text(
        -0.5,
        hi + 0.04,
        "bright reflections, every model",
        ha="left",
        va="bottom",
        color=roles["muted"],
    )
    for i, m in enumerate(methods):
        for population, off, fill in (("bright", -DX, False), ("weak", DX, True)):
            r = rows.get((m, population))
            if r is None:
                continue
            ax.errorbar(
                x[i] + off,
                r["axis_ratio"],
                yerr=[[r["axis_ratio"] - r["p25"]], [r["p75"] - r["axis_ratio"]]],
                marker="D" if fill else "o",
                markersize=4,
                color=pal[m],
                markerfacecolor=pal[m] if fill else "none",
                ls="",
                capsize=2,
                elinewidth=0.8,
                zorder=3,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(methods) - 0.4)
    ax.set_ylabel("axis ratio of the profile")
    ax.set_ylim(1.0, None)
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", ms=4, color=roles["ink"],
                   markerfacecolor="none", label="bright reflections"),
            Line2D([], [], ls="", marker="D", ms=4, color=roles["ink"],
                   label="weak reflections"),
        ],
        loc="upper left",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.2,
    )


def main() -> None:
    shape = read(DATA / "profile_shape.csv")
    aniso = read(DATA / "profile_anisotropy.csv")
    methods = ordered(shape["method"].unique().to_list())
    pal = palette(methods)

    figstyle.use()
    fig, axes = plt.subplots(
        1, 2, figsize=figstyle.figsize("full", aspect=0.46), layout="constrained"
    )
    orientation_panel(axes[0], shape, methods, pal)
    width_panel(axes[1], aniso, methods, pal)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = figstyle.save(fig, OUT / "profile_shape")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
