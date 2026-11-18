"""Names, colours and ordering for the m9 training arms, from figtex's witch.

Every figure drawn from `docs/figures/data/` reads its colours here, so an arm
keeps the same colour across figures. The palette is the witch categorical
palette, which is built so that the first k colours are the most distinct k in
OKLab under normal, deuteranope and protanope vision; taking a slice of
`ORDER` therefore still gives a legible figure.

Rendering happens on the laptop because the documents are set in Minion Pro
with mtpro2 maths through LuaLaTeX, and those fonts are installed only there.
The cluster writes the long-format CSVs under `docs/figures/data/` and nothing
else travels.
"""

import figstyle

# The arms in the order they are compared: the control first, then one
# geometry term at a time, then the caps, then the responsibility gate.
ORDER = (
    "m9_gate_learned",
    "m9_geomshift",
    "m9_geom",
    "m9_geomcap3",
    "m9_geomcap",
    "m9_geomcap2",
    "m9_geomresp",
    "m9_geomrespsteer",
)

LABELS = {
    "m9_gate_learned": "learned basis",
    "m9_geomshift": r"$+$ local shift",
    "m9_geom": r"$+$ shift, steer",
    "m9_geomcap3": r"steer, cap $1.49$",
    "m9_geomcap": r"steer, cap $1.90$",
    "m9_geomcap2": r"steer, cap $2.72$",
    "m9_geomresp": "responsibility gate",
    "m9_geomrespsteer": r"responsibility $+$ steer",
}

# The axis ratio of the profiles each model fits to its OWN bright
# reflections, where the counts constrain the shape: 1.41 on m9_geom, 1.58 on
# m9_geomshift, 1.48 on m9_gate_learned. A weak-reflection profile outside
# this band is wider than anything the data supports.
SUPPORTED_RATIO = (1.41, 1.58)

# The null for the axial orientation difference. The orientation comes from
# the central second-moment tensor, whose axis is defined modulo 180 degrees,
# so two independent directions differ by 45 degrees on average, not 90.
DTHETA_NULL = 45.0


def palette(methods) -> dict[str, str]:
    """A witch colour per method, assigned by position in `ORDER`."""
    pal = figstyle.categorical(len(ORDER))
    idx = {m: i for i, m in enumerate(ORDER)}
    return {m: pal[idx.get(m, len(pal) - 1) % len(pal)] for m in methods}


def label(method: str) -> str:
    return LABELS.get(method, method.replace("m9_", "").replace("_", " "))


def ordered(methods) -> list[str]:
    """`methods` in `ORDER`, with anything unknown kept at the end."""
    seen = list(methods)
    known = [m for m in ORDER if m in seen]
    return known + [m for m in seen if m not in known]
