"""One patch of one diffraction image, and the profiles drawn inside it.

The left axes is a square of the detector painted from the shoeboxes that
fall on it: squares are bright reflections, circles weak ones, and the
labelled circles are the weak reflections whose profiles are shown on the
right. The right columns are, for each labelled weak reflection, the observed
shoebox, the mean shape of the 25 bright reflections nearest it, and the
profile each model drew for it.

The number under each model panel is the axial orientation difference from
that local bright mean, in degrees, with 45 as the null. It replaces the
cosine similarity the first version of this figure printed, which sat at the
chance level for nonnegative vectors throughout.

One figure per named detector location -- the beam centre, mid radius, a near
and a far corner -- which is what says whether a model tracks position or
just draws the same shape everywhere.

Run (after `bash scripts/figures/pull_figdata.sh patch`):
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_detector_patch.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

import figstyle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import DTHETA_NULL, label  # noqa: E402
from shapes import dtheta_images  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "m9"
MODELS = ("m9_geom", "m9_geomshift", "m9_gate_learned")
PATCHES = ("beam_centre", "mid_radius", "corner_near", "corner_far")
BOX = 21


def patch_names(z) -> list[str]:
    present = {k.rsplit("_pix", 1)[0] for k in z.files if k.endswith("_pix")}
    return [p for p in PATCHES if p in present] + sorted(present - set(PATCHES))


def draw(z, name: str):
    roles = figstyle.roles()
    cat = figstyle.categorical(8)
    c_bright, c_weak = cat[3], cat[1]

    pix = z[f"{name}_pix"]
    x0, x1, y0, y1 = z[f"{name}_extent"]
    frame = int(z[f"{name}_frame"][0])
    boxes = z[f"{name}_boxes"]
    marks = z[f"{name}_marks"]
    obs = z[f"{name}_obs"]
    local = z[f"{name}_local"]
    n = len(marks)

    figstyle.use()
    ncol = 2 + len(MODELS)
    # Every panel on the right is a square 21x21 shoebox and the patch on the
    # left is a square of detector, so the whole figure is fixed once the
    # panel is the unit: the patch is n units on a side, the block of
    # profiles is ncol wide and n tall, and the height follows. Asking for a
    # taller figure than that only makes constrained layout push the rows
    # apart to fill it.
    pad = 0.20  # the three-line title, the axis labels and the delta-theta row
    fig = plt.figure(
        figsize=figstyle.figsize("full", aspect=n / (n + ncol) + pad),
        layout="constrained",
    )
    gs = fig.add_gridspec(1, 2, width_ratios=[n, ncol], wspace=0.04)
    sub = gs[0, 1].subgridspec(n, ncol, hspace=0.3, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])

    lit = pix[pix > 0]
    vmax = np.percentile(lit, 99.3) if lit.size else 1.0
    ax.imshow(
        np.log1p(np.maximum(pix, 0.0)),
        cmap=figstyle.cmap(),
        origin="lower",
        vmin=0.0,
        vmax=np.log1p(vmax),
        extent=[x0, x1, y0, y1],
        interpolation="nearest",
    )
    for bx, by, _cx, _cy, is_bright in boxes:
        if is_bright > 0.5:
            ax.add_patch(
                Rectangle((bx, by), BOX, BOX, fill=False, ec=c_bright, lw=0.7)
            )
        else:
            ax.add_patch(
                Circle((_cx, _cy), 11, fill=False, ec=c_weak, lw=0.6, alpha=0.8)
            )
    for i, (mx, my, _snr, _rad, _rb) in enumerate(marks):
        ax.add_patch(Circle((mx, my), 16, fill=False, ec=c_weak, lw=1.6))
        ax.annotate(
            chr(ord("a") + i),
            (mx, my),
            xytext=(13, 9),
            textcoords="offset points",
            color=c_weak,
        )
    ax.set_xlabel("detector $x$ (px)")
    ax.set_ylabel("detector $y$ (px)")
    # the title sits over the left column only, which is n/(n+ncol) of the
    # text width, so it is broken into lines that fit there rather than
    # running on under the profile panels
    ax.set_title(
        f"{name.replace('_', ' ')}, frame {frame}, {int(x1 - x0)} px\n"
        f"$r$ from the beam centre {np.median(marks[:, 4]):.0f} px\n"
        "square = bright, circle = weak",
        loc="left",
        fontsize=7,
    )
    ax.grid(visible=False)

    heads = ["observed", "local bright\nmean of 25"] + [label(m) for m in MODELS]
    stacks = [obs, local] + [z[f"{name}_{m}"] for m in MODELS]
    dth = {m: dtheta_images(z[f"{name}_{m}"], local) for m in MODELS}
    for i in range(n):
        for j, (head, stack) in enumerate(zip(heads, stacks, strict=True)):
            a = fig.add_subplot(sub[i, j])
            im = stack[i]
            a.imshow(
                im,
                cmap=figstyle.cmap(),
                origin="lower",
                vmin=0.0,
                vmax=max(float(im.max()), 1e-9),
                interpolation="nearest",
            )
            a.set_xticks([])
            a.set_yticks([])
            a.grid(visible=False)
            if i == 0:
                a.set_title(head, fontsize=6)
            if j == 0:
                a.set_ylabel(
                    rf"{chr(ord('a') + i)}  $I/\sigma$ {marks[i, 2]:.1f}",
                    fontsize=6,
                    color=c_weak,
                )
            if j >= 2:
                d = dth[MODELS[j - 2]][i]
                a.set_xlabel(
                    rf"$\Delta\theta$ {d:.0f}$^\circ$",
                    fontsize=6,
                    color=roles["ink"] if d < DTHETA_NULL / 2 else roles["muted"],
                )
    return fig


def main() -> None:
    src = DATA / "patch_data.npz"
    if not src.exists():
        raise SystemExit(
            f"{src} is missing; run `bash scripts/figures/pull_figdata.sh patch`"
        )
    z = np.load(src)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in patch_names(z):
        fig = draw(z, name)
        paths = figstyle.save(fig, OUT / f"detector_patch_{name}")
        plt.close(fig)
        print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
