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
# The template the last `use` selected, which is the one every size here is
# measured against.
#
# It has to be remembered on this side because figTeX's own `style.figsize`
# does not: `style.use(name, theme=...)` records the template in its state
# but leaves `_STATE["journal"]` at its default, `iucr`, so `style.figsize`
# returns iucr's 469.76 pt text width whatever template is active. Every
# PDZ2 figure came out 470 pt wide for that reason -- the article text
# block is 374 pt, a paper column 243 -- so the figures were sized for a
# page none of the documents use. `panels` and `fit` below take the
# template explicitly and go through `figtex.sizing`, which reads the
# template's own geometry.
_ACTIVE = LAYOUT
# Two palettes, and which one a figure wants depends on whether its series
# are ordered.
#
# `categorical=True` is Ink Dark2: eight distinct hues, for arms that have
# no order -- integrator families, geometry variants, the reference against
# the models. Amber stands in for Dark2's orange so that nothing competes
# with the vermilion reference.
#
# `categorical=False` is the blue ramp, six shades, for series that ARE
# ordered: epochs, a sweep value, or one emphasised arm against the rest.
# `epoch_ramp` is the usual way to ask for it.
#
# Above eight the palette raises rather than cycling, and so does this; see
# `arm_colors`.
MAX_SERIES = 8
MAX_SHADES = 6

__all__ = [
    "LAYOUT",
    "active",
    "fit",
    "panels",
    "top_legend",
    "MAX_SERIES",
    "MAX_SHADES",
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
    global _ACTIVE
    out = style.use(layout, theme=THEME,
                    fast=_fast() if fast is None else fast)
    _ACTIVE = layout
    return out


def active() -> str:
    """The template `use` last selected."""
    return _ACTIVE


def figsize(width="full", aspect: float = 0.618, column: bool = False):
    """Inches for a fraction of the ACTIVE TEMPLATE's text width.

    Not `style.figsize`, which measures against the default journal rather
    than the template in force; see `_ACTIVE`. A figure that can be drawn
    through `panels` should be, because that also sizes the data areas and
    the gaps; this is for the ones not converted yet.
    """
    t = sizing.TEMPLATES[_ACTIVE]
    frac = _WIDTHS.get(width, width) if isinstance(width, str) else float(width)
    if t.medium == "slides":
        base = t.paper_width - 2 * t.slide_margin
    elif column and t.column:
        base = t.column
    else:
        base = t.text_width
    w = base * frac / 72.27
    return (w, w * aspect)


# `style.WIDTHS` by name, so `figsize("half")` keeps meaning what it did
_WIDTHS = {"full": 1.0, "half": 0.5, "third": 1 / 3, "twothirds": 2 / 3,
           "quarter": 0.25}


def panels(layout: str = "1", **kw):
    """A sized grid for the active template: returns figTeX's `Panels`.

    The template-aware path. `layout` is figTeX's grid spelling ("1",
    "2x2", ...), and the keywords are `sizing.panels`': `column`, `width`,
    `shared`, `images`, `legend`, `header`, `caption`, `height`.
    Call `P.finish()` once everything is drawn, then `save(P.fig, path)`.

    `column=True` asks for one column of a multi-column page and is
    DROPPED where the template has no columns, because there the text
    block is the column: `article`, `book`, `slides` and `poster` all
    carry `column = None` and figTeX computes the panel width as
    `(None - gap) / cols`, which raises. A caller can therefore say
    `column=True` once for every template and get the column where there
    is one (paper, 243 pt) and the text block where there is not.
    """
    if kw.get("column") and not sizing.TEMPLATES[_ACTIVE].column:
        kw = {**kw, "column": False}
    return sizing.panels(_ACTIVE, layout, **kw)


def fit(fig, width="full", **kw):
    """Size an already-built figure to the active template."""
    return sizing.fit(fig, _ACTIVE, width=width, **kw)


def top_legend(P, handles, labels, **kw):
    """A boxed legend strip above the panels, wrapped to the figure width.

    `Panels.top_legend` puts every entry in one row, which overflows as
    soon as there are more than three or four: four arms ran 47 pt past
    the article text block and six ran 146 pt past it, and the overflow is
    silently cut off in the output rather than raised. This draws the
    strip, MEASURES it, and gives it another row until it fits, so the one
    call works at a 243 pt column and at a 1015 pt poster column.
    """
    if not handles:
        return None
    n = len(labels)
    for ncols in [n] + [c for c in (4, 3, 2, 1) if c < n]:
        leg = P.top_legend(handles, labels, frameon=True, ncols=ncols, **kw)
        w_pt = (leg.get_window_extent(P.fig.canvas.get_renderer()).width
                * 72 / P.fig.dpi)
        if w_pt <= P.info["W"]:
            return leg
        leg.remove()
        P._top_legend = None
    return P.top_legend(handles, labels, frameon=True, ncols=1, **kw)


def save(fig, path):
    """Write `path.pdf` and `path.png` at the size the figure was drawn."""
    return style.save(fig, path)


def _ensure() -> None:
    """Make sure the theme is loaded before its colours are read.

    `figtex.sizing.THEME` is empty until `style.use` has run -- it holds
    only its own name -- so every accessor here has to tolerate being the
    first call. `parts.ramp` reads `THEME["ramp"]` and raised
    `KeyError: 'ramp'` from `categorical`, which killed the `shape` and
    `arms` figures: both ask for colours before calling `use`.
    """
    if "ramp" not in sizing.THEME or "text_hex" not in sizing.THEME:
        use()


def roles() -> dict[str, str]:
    """The theme's colours, under the names the scripts already ask for.

    `color.roles` is a SCHEME function and raises for a theme, so this maps
    `figtex.sizing.THEME` instead. Both spellings of each colour are
    returned -- `ink` and `text`, `muted` and `frame` -- because the scripts
    were written against the scheme's names and there is no reason to edit
    forty of them to rename a key.
    """
    _ensure()
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
    _ensure()
    n = max(int(n), 1)
    return sizing.ramp_cmap(), parts.ramp(n)


def cmap(name: str | None = None):
    """The theme's sequential colormap, pale through dark.

    Image figures -- a shoebox, a profile, a detector patch -- take this
    rather than `viridis` or `magma`, so that changing the theme in one
    place changes them too.
    """
    _ensure()
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
    _ensure()
    st = dict(parts.ref_style())
    st.setdefault("zorder", 5)
    # `ls` is dropped: several callers pass their own `linestyle` beside
    # this dict, and matplotlib refuses both spellings of one property --
    #   TypeError: Got both 'linestyle' and 'ls', which are aliases
    # It carried '-', which is matplotlib's default, so nothing that does
    # not set a style looks any different without it.
    st.pop("ls", None)
    return st


def arm_series(names) -> dict:
    """A full `figtex.texture.Series` per name: colour, dash, marker, hatch.

    Arms are told apart by texture as well as shade, so a figure survives
    greyscale and print. `texture.plot(ax, x, y, series)` draws a line with
    its texture.

    Raises nothing for more than `MAX_SERIES` names: the series cycle, and
    `arm_colors` is where the ordered fallback lives.
    """
    _ensure()
    names = list(names)
    base = texture.system(THEME, min(max(len(names), 1), MAX_SERIES),
                          reference=True, categorical=True)
    return {n: base[i % len(base)] for i, n in enumerate(names)}


def categorical(n: int = 6):
    """`n` categorical colours, for a figure with no arms.

    `arm_colors` is the one to use where the names are training arms, so an
    arm keeps its colour across figures. This is for everything else.
    """
    _ensure()
    n = max(int(n), 1)
    if n <= MAX_SERIES:
        return [s.color for s in texture.system(THEME, n, reference=True,
                                                categorical=True)]
    return parts.ramp(n)


def arm_colors(names) -> dict[str, str]:
    """A colour per name, in the order given.

    Training arms have no order, so they take Ink Dark2's eight hues rather
    than shades of one. Shades were what ink offered before the categorical
    palette existed, and six arms of graded blue on one axes -- which is
    what `careless_stats_pdz2_1116` is -- could not be told apart.

    Above eight the palette raises rather than cycling, and repeating a hue
    would be worse than the ramp, so the ramp is used and the figure is
    worth rebuilding: the 20-arm 1118 ablation is the one in that position,
    and small multiples or a chosen subset reads better than twenty lines
    however they are coloured.
    """
    _ensure()
    names = list(names)
    if len(names) <= MAX_SERIES:
        pal = [s.color for s in texture.system(THEME, max(len(names), 1),
                                               reference=True,
                                               categorical=True)]
    else:
        pal = parts.ramp(len(names))
    return {n: pal[i % len(pal)] for i, n in enumerate(names)}
