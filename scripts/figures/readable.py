"""Large-panel diagnostic figures: a fixed panel size in inches, readable type, PDF output.

The article figures are sized to a text block, so a figure with nine or twelve
panels gives each one a few centimetres and the labels collide. Diagnostics are
read on a screen, so here each panel has a fixed size (default 3.6 x 3.2 in)
and the figure grows with the number of panels instead. Plain matplotlib, no
figTeX, so these render anywhere; colours per arm are fixed in `COLOURS`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# one colour per arm, the same in every diagnostic; laue.integrate is the reference
COLOURS = {
    "laue_integrate": "#D55E00",
    "svae_rb": "#999999",
    "t0_rb": "#0072B2",
    "p2_rb": "#009E73",
    "b1_rb": "#E69F00",
    "b1_laplace": "#CC79A7",
    "geom_laplace": "#56B4E9",
}
LABELS = {
    "laue_integrate": "laue.integrate",
    "svae_rb": "svae (learned basis), i_rb",
    "t0_rb": "T0, i_rb",
    "p2_rb": "P2, i_rb",
    "b1_rb": "B1, i_rb",
    "b1_laplace": "B1, Laplace + collapsed",
    "geom_laplace": "geometric prior, collapsed",
}
ORDER = ("laue_integrate", "svae_rb", "t0_rb", "p2_rb", "b1_rb", "b1_laplace", "geom_laplace")


def setup(size: float = 11.0) -> None:
    plt.rcParams.update({
        "font.size": size, "axes.titlesize": size + 1, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1, "legend.fontsize": size - 1,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def grid(n: int, ncols: int = 3, panel: tuple[float, float] = (3.6, 3.2), **kw):
    """A figure of n panels in rows of `ncols`, each `panel` inches; returns (fig, flat axes); spares hidden."""
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
                             layout="constrained", squeeze=False, **kw)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def ordered(methods) -> list[str]:
    ms = list(dict.fromkeys(methods))
    return [m for m in ORDER if m in ms] + [m for m in ms if m not in ORDER]


def label(m: str) -> str:
    return LABELS.get(m, m)


def colour(m: str) -> str:
    return COLOURS.get(m, "#333333")


def save(fig, path: Path) -> list[Path]:
    """PDF always; PNG beside it as well when READABLE_PNG is set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".pdf")]
    fig.savefig(out[0])
    if os.environ.get("READABLE_PNG"):
        out.append(path.with_suffix(".png"))
        fig.savefig(out[-1])
    plt.close(fig)
    return out
