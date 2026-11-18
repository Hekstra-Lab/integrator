"""The one place the figure scripts get their look: figTeX, the ink theme.

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

**ink is a THEME, not a colour scheme.** `style.use(..., theme="ink")` is
how it is selected and `color.roles("ink")` raises, because the registered
schemes are noir, charcoal, witch, inkblue, noir-helix, noir-dark2 and
mono. The theme's colours live in `figtex.sizing.THEME` and are only
populated once `style.use` has run, so `roles()` below calls it if nothing
has yet.

The function names here are unchanged from the witch version. Forty figure
scripts call them, and the point of a routing module is that a change of
look does not touch any of them.

`FIGTEX_FAST=1` selects figTeX's fast mode, which skips the LaTeX text
pass. Use it while iterating; the committed figures are drawn in final
mode.
"""

from __future__ import annotations

import os

try:
    from figtex import color, parts, sizing, style, texture
except ModuleNotFoundError as exc:  # pragma: no cover - a setup message
    raise SystemExit(
        "figtex is not importable. Run this script as\n"
        "  uv run --with-editable /Users/luis/master/software/figtex "
        "python <script> ...\n"
        "or add it to the environment with `uv add --editable "
        "~/master/software/figtex`."
    ) from exc

THEME = "ink"
LAYOUT = "article"
# `texture.system` carries six distinguishable series and raises IndexError
# above that. Figures with more arms than this fall back to the ordered
# ramp; see `arm_colors`.
MAX_SERIES = 6

__all__ = [
    "LAYOUT",
    "MAX_SERIES",
    "THEME",
    "arm_colors",
    "arm_series",
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

# `SCHEME` was the witch scheme's name and is kept so that a script still
# importing it does not fail at import; nothing routes through it now.
SCHEME = THEME


def _fast() -> bool:
    return os.environ.get("FIGTEX_FAST", "").strip() not in ("", "0", "false")


def use(layout: str = LAYOUT, fast: bool | None = None):
    """Fonts, sizes and the ink look. Call once, before plotting."""
    return style.use(layout, theme=THEME,
                     fast=_fast() if fast is None else fast)


def figsize(width="full", aspect: float = 0.618, column: bool = False):
    """Inches for a fraction of the document's text width."""
    return style.figsize(width, aspect=aspect, column=column)


def save(fig, path):
    """Write `path.pdf` and `path.png` at the size the figure was drawn."""
    return style.save(fig, path)


def roles() -> dict[str, str]:
    """The theme's colours, under the names the scripts already ask for.

    `color.roles` is a SCHEME function and raises for a theme, so this maps
    `figtex.sizing.THEME` instead. Both spellings of each colour are
    returned -- `ink` and `text`, `muted` and `frame` -- because the scripts
    were written against the scheme's names and there is no reason to edit
    forty of them to rename a key.
    """
    if "text_hex" not in sizing.THEME:
        use()
    t = sizing.THEME
    return {
        "ink": t["text_hex"],
        "text": t["text_hex"],
        "muted": t["frame_hex"],
        "frame": t["frame_hex"],
        "rule": t["frame_hex"],
        "grid": t["grid_hex"],
        "paper": t["ground"],
        "ground": t["ground"],
        "accent": t["accent"],
        "ref": t["ref_hex"],
    }


def epoch_ramp(n: int):
    """`(cmap, colors)` for `n` checkpoints, earliest light to latest dark.

    The ink ramp, so the order survives greyscale and colour blindness.
    `figtex.parts.epoch_bar` draws the colour bar that names the epochs.
    """
    n = max(int(n), 1)
    return sizing.ramp_cmap(), parts.ramp(n)


def cmap(name: str | None = None):
    """The theme's sequential colormap, pale through dark.

    Image figures -- a shoebox, a profile, a detector patch -- take this
    rather than `viridis` or `magma`, so that changing the theme in one
    place changes them too.
    """
    return color.cmap(name) if name else sizing.ramp_cmap()


def diverging_cmap():
    """Two hues through a neutral midpoint, for a signed residual.

    `noir-div`: the ink theme has no diverging ramp of its own, and noir's
    is the neutral-midpoint pair that sits with it.
    """
    return color.cmap("noir-div")


def reference_style() -> dict:
    """How the reference integrator is drawn: one line, no epoch ramp.

    Vermilion, solid and heavier, so it cannot be mistaken for a checkpoint
    of the epoch ramp or for one of the textured arms.
    """
    st = dict(parts.ref_style())
    st.setdefault("zorder", 5)
    return st


def arm_series(names) -> dict:
    """A full `figtex.texture.Series` per name: colour, dash, marker, hatch.

    Arms are told apart by texture as well as shade, so a figure survives
    greyscale and print. `texture.plot(ax, x, y, series)` draws a line with
    its texture.

    Raises nothing for more than `MAX_SERIES` names: the series cycle, and
    `arm_colors` is where the ordered fallback lives.
    """
    names = list(names)
    base = texture.system(THEME, min(max(len(names), 1), MAX_SERIES),
                          reference=True)
    return {n: base[i % len(base)] for i, n in enumerate(names)}


def categorical(n: int = 6):
    """`n` categorical colours, for a figure with no arms.

    `arm_colors` is the one to use where the names are training arms, so an
    arm keeps its colour across figures. This is for everything else.
    """
    n = max(int(n), 1)
    if n <= MAX_SERIES:
        return [s.color for s in texture.system(THEME, n, reference=True)]
    return parts.ramp(n)


def arm_colors(names) -> dict[str, str]:
    """A colour per name, in the order given.

    Up to `MAX_SERIES` arms take the theme's categorical series, which are
    separable by hue and by texture. Beyond that the ink theme has no
    categorical set -- `texture.system` raises `IndexError` at seven -- and
    the ordered ramp is used instead.

    A ramp is the right answer when the arms are ordered, as they are for an
    epoch or a sweep value. It is the wrong answer for unordered arms, where
    small multiples or a chosen subset reads better than eight lines of
    graded blue; the figures in that position are the 8-arm m9 set and the
    20-arm 1118 ablation, and they are worth revisiting rather than
    accepting as drawn.
    """
    names = list(names)
    if len(names) <= MAX_SERIES:
        pal = [s.color for s in texture.system(THEME, max(len(names), 1),
                                               reference=True)]
    else:
        pal = parts.ramp(len(names))
    return {n: pal[i % len(pal)] for i, n in enumerate(names)}
