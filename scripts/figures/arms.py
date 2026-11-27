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
    # mos010 photon-counting simulation, careless -> phenix arms (2026-11-25)
    "b1_laplace": r"B1, Laplace $+$ collapsed",
    "geom_laplace": "geometric prior, collapsed",
    "b1_rb": r"B1, $i_\mathrm{rb}$",
    "p2_rb": r"P2, $i_\mathrm{rb}$",
    "t0_rb": r"T0, $i_\mathrm{rb}$",
    "laue_integrate": "laue.integrate",
    "svae_rb": r"svae (learned basis), $i_\mathrm{rb}$",
    "svae_bg_rb": r"svae\_bg (learned basis), $i_\mathrm{rb}$",
    # mos025 arms (2026-11-27)
    "geom_both_rb": r"geom\_both (geometric basis), $i_\mathrm{rb}$",
    "geom_both_posterior": r"geom\_both, posterior",
    "svae_bg_posterior": r"svae\_bg, posterior",
    "t0_posterior": "T0, posterior",
    "p2_posterior": "P2, posterior",
    "b1_posterior": "B1, posterior",
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


# The mos010 careless -> phenix arms, in the order their colours are taken.
# laue.integrate is the reference and is drawn in figTeX's reference colour;
# the five model arms take the categorical palette in this order, so every
# figure of that analysis shows an arm in one colour.
SIM_ORDER = ("laue_integrate", "svae_rb", "svae_bg_rb", "t0_rb", "p2_rb", "b1_rb", "b1_laplace", "geom_laplace")
SIM_REFERENCE = "laue_integrate"


def colours(methods) -> dict[str, str]:
    """A colour per method: fixed by `SIM_ORDER` when every method is one of
    those arms, otherwise figTeX's categorical colours by position."""
    methods = list(methods)
    if methods and all(m in SIM_ORDER for m in methods):
        pal = dict(figstyle.arm_colors(list(SIM_ORDER[1:])))
        pal[SIM_REFERENCE] = figstyle.reference_style()["color"]
        return {m: pal[m] for m in methods}
    return dict(figstyle.arm_colors(methods))

