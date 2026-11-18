"""The one place the figure scripts get their look: figTeX, witch scheme.

`plot_loss.py`, `plot_merging.py`, `plot_peaks.py` and the rest import this
rather than setting their own colours, so a figure drawn by one matches a
figure drawn by another and both match the document.

figTeX is not a dependency of this project; it is pulled in at the call:

    uv run --with-editable /Users/luis/master/software/figtex \\
        python plot_merging.py --plot-cfg cfg_9b7c.yaml --out-dir out/

Text is typeset by LuaLaTeX with the fonts the documents use, which are
installed on the laptop only, so these scripts run here and not on the
cluster. The cluster's job is to reduce a run to the small tables and dumps
they read.
"""

from __future__ import annotations

try:
    from figtex import color, style
except ModuleNotFoundError as exc:  # pragma: no cover - a setup message
    raise SystemExit(
        "figtex is not importable. Run this script as\n"
        "  uv run --with-editable /Users/luis/master/software/figtex "
        "python <script> ...\n"
        "or add it to the environment with `uv add --editable "
        "~/master/software/figtex`."
    ) from exc

import numpy as np

SCHEME = "witch"
LAYOUT = "article"

__all__ = [
    "LAYOUT",
    "SCHEME",
    "arm_colors",
    "categorical",
    "cmap",
    "diverging_cmap",
    "epoch_ramp",
    "figsize",
    "reference_style",
    "roles",
    "save",
    "use",
]


def use(layout: str = LAYOUT):
    """Fonts, sizes and the witch colour cycle. Call once, before plotting."""
    return style.use(layout, scheme=SCHEME)


def figsize(width="full", aspect: float = 0.618, column: bool = False):
    """Inches for a fraction of the document's text width."""
    return style.figsize(width, aspect=aspect, column=column)


def save(fig, path):
    """Write `path.pdf` and `path.png` at the size the figure was drawn."""
    return style.save(fig, path)


def roles() -> dict[str, str]:
    """`ink`, `muted`, `rule`, `grid`, `paper`, and the status colours."""
    return color.roles(SCHEME)


def epoch_ramp(n: int):
    """`(cmap, colors)` for `n` checkpoints, earliest light to latest dark.

    Replaces the seaborn cubehelix the scripts used to build. The witch
    sequential ramp is a cubehelix in OKLCH whose lightness rises linearly
    while the hue turns from the scheme's primary towards its secondary, so
    the order survives greyscale and colour blindness.
    """
    cmap = color.cmap(f"{SCHEME}_r")
    n = max(int(n), 1)
    return cmap, cmap(np.linspace(0.0, 1.0, n))


def cmap(name: str | None = None):
    """The scheme's sequential colormap, pale through dark.

    Image figures -- a shoebox, a profile, a detector patch -- take this
    rather than `viridis` or `magma`, so that changing the scheme in one
    place changes them too.
    """
    return color.cmap(name or f"{SCHEME}_r")


def diverging_cmap():
    """Two hues through a neutral midpoint, for a signed residual."""
    return color.cmap(f"{SCHEME}-div")


def reference_style() -> dict:
    """How the reference integrator is drawn: one line, no epoch ramp.

    It takes a categorical colour rather than a step of the epoch ramp, so it
    cannot be mistaken for a checkpoint.
    """
    # index 2 is the teal: the epoch ramp runs pale pink to dark violet, so a
    # warm reference line sits inside that range and a cool one does not
    return {"color": color.categorical(SCHEME, 8)[2], "lw": 1.6, "zorder": 5}


def categorical(n: int = 8):
    """`n` categorical colours from the scheme, for a figure with no arms.

    `arm_colors` is the one to use where the names are training arms, so an
    arm keeps its colour across figures. This is for everything else.
    """
    return color.categorical(SCHEME, min(max(int(n), 3), 8))


def arm_colors(names) -> dict[str, str]:
    """A witch categorical colour per name, in the order given."""
    pal = color.categorical(SCHEME, min(max(len(list(names)), 3), 8))
    return {n: pal[i % len(pal)] for i, n in enumerate(names)}
