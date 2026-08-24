"""Plot calibrated (real-valued) vs rounded (integer) shoeboxes side by side.

Three panels per sample:

  counts_real     what the detector gives: (adu - pedestal)/gain, real-valued, negative
                  on background pixels.
  counts_poisson  the same pixels rounded and clamped -- what a Poisson likelihood eats.
  difference      counts_real - counts_poisson, i.e. exactly what rounding removed. This
                  is the panel that carries the result: in G0 it is pure sub-photon read
                  noise (+-0.024 ph) and the two left panels are identical, because
                  rounding recovers the true count. Where a pixel switches to G1 the
                  residual jumps ~30x and rounding starts genuinely losing counts.

The left two panels share a color scale per row so they are directly comparable; that
they look identical is the point. Colors follow the design system: magnitude takes the
one-hue blue sequential ramp (100->700, light->dark), and the signed residual takes the
blue<->red diverging pair with the neutral gray midpoint pinned at zero.

Run:  uv run python scripts/jungfrau_sim/plot_shoeboxes.py --data data/jf_sim
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Design-system tokens (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"

# Sequential: blue 100 -> 700, the documented single-hue ramp for magnitude.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Diverging: blue <-> red poles, neutral gray midpoint (never a hue at the midpoint).
DIVERGING = ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f2a09f", "#e34948", "#b02a2a"]

SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)


def load(data_dir: Path) -> dict[str, np.ndarray]:
    keys = ["counts_real", "counts_poisson", "counts_true", "intensity", "gain_stage"]
    return {k: np.load(data_dir / f"{k}.npy") for k in keys}


def pick(intensity: np.ndarray, targets: list[float]) -> list[int]:
    return [int(np.argmin(np.abs(intensity - t))) for t in targets]


def _bar(fig, im, ax) -> None:
    """Recessive colorbar: the scale is the legend, so it stays in muted ink."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("photons", color=INK_MUTED, fontsize=7)
    cb.ax.tick_params(labelsize=6, colors=INK_MUTED)
    cb.outline.set_visible(False)


def plot(data: dict[str, np.ndarray], idx: list[int], h: int, w: int, out: Path) -> None:
    n = len(idx)
    fig, axes = plt.subplots(
        n, 3, figsize=(8.4, 2.55 * n), facecolor=SURFACE, constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    titles = [
        "counts_real  (detector output)",
        "counts_poisson  (rounded)",
        "difference  (what rounding removed)",
    ]

    for r, i in enumerate(idx):
        real = data["counts_real"][i].reshape(h, w)
        pois = data["counts_poisson"][i].reshape(h, w).astype(float)
        stage = data["gain_stage"][i].reshape(h, w)
        diff = real - pois

        # Shared scale for the two left panels: comparability is the whole point.
        vmax = max(real.max(), pois.max())
        axes[r, 0].imshow(real, cmap=SEQ, vmin=0.0, vmax=vmax, interpolation="nearest")
        im = axes[r, 1].imshow(pois, cmap=SEQ, vmin=0.0, vmax=vmax, interpolation="nearest")
        _bar(fig, im, axes[r, 1])  # one bar for both left panels; they share the scale

        # Residual: symmetric about zero so the neutral midpoint means "nothing removed".
        lim = float(np.abs(diff).max()) or 1e-6
        im = axes[r, 2].imshow(
            diff, cmap=DIV, norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
            interpolation="nearest",
        )
        _bar(fig, im, axes[r, 2])

        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=8.5, color=INK, pad=7)

        exact = 100.0 * (pois == data["counts_true"][i].reshape(h, w)).mean()
        n_g1 = int((stage > 0).sum())
        stage_note = f"{n_g1} px in G1" if n_g1 else "all G0"
        axes[r, 0].set_ylabel(
            f"I = {data['intensity'][i]:.0f}\n{stage_note}",
            fontsize=8.5, color=INK, rotation=0, ha="right", va="center", labelpad=30,
        )
        # Direct-label each residual with what it cost -- selectively, one per row.
        axes[r, 2].set_xlabel(
            f"largest removed {lim:.3f} ph   ·   counts recovered {exact:.2f}%",
            fontsize=7, color=INK_MUTED, labelpad=5,
        )

    fig.suptitle(
        "JUNGFRAU shoeboxes: calibrated real-valued vs rounded to integers",
        fontsize=10.5, color=INK, y=1.015,
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/jf_sim"))
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--w", type=int, default=20)
    ap.add_argument(
        "--intensities", type=float, nargs="+", default=[5.0, 40.0, 200.0, 1200.0],
        help="pick the sample nearest each of these true intensities",
    )
    ap.add_argument("--out", type=Path, default=Path("scripts/jungfrau_sim/shoeboxes.png"))
    args = ap.parse_args()

    data = load(args.data)
    idx = pick(data["intensity"], args.intensities)
    plot(data, idx, args.h, args.w, args.out)


if __name__ == "__main__":
    main()
