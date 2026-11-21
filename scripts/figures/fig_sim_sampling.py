"""Noiseless datsimx spots at each (enSteps, mosDoms) sampling, one figure per mosaic spread.

Rows are the reflections ranked 1, 10, 30, 100, 300, 1000 by their truth sum
in the (322, 100) reference; columns are the sampling settings of the grid
`scripts/sim_truth/grids/sampling_check.tsv`. Every panel is a 61 x 61 px crop
centred on the reference centroid, on a linear scale shared along the row, so
a setting that renders a streak as separate dots shows them.

Data: `compare_sampling.py --patches`, pulled to
`docs/figures/data/sim_sampling_patches.npz`.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_sampling.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "figures" / "data" / "sim_sampling_patches.npz"
OUT = ROOT / "docs" / "figures" / "sim"


def draw(z, tag: str):
    names = list(z[f"{tag}_names"])
    settings = z[f"{tag}_settings"]
    ranks, i_ref, lam = z[f"{tag}_rank"], z[f"{tag}_i_ref"], z[f"{tag}_lam"]
    stacks = [z[f"{tag}_{n}"] for n in names]
    nrow, ncol = len(ranks), len(names)

    figstyle.use()
    fig, axes = plt.subplots(
        nrow, ncol, figsize=figstyle.figsize("full", aspect=nrow / ncol * 1.08),
        layout="constrained", squeeze=False,
    )
    for i in range(nrow):
        vmax = max(float(s[i].max()) for s in stacks) or 1e-9
        for j, stack in enumerate(stacks):
            a = axes[i, j]
            a.imshow(stack[i], cmap=figstyle.cmap(), origin="lower", vmin=0.0, vmax=vmax,
                     interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.grid(visible=False)
            if i == 0:
                en, doms = settings[j]
                a.set_title(f"{names[j]}\n{en} energies, {doms} domains", fontsize=5)
            if j == 0:
                a.set_ylabel(f"rank {ranks[i]}\n{i_ref[i]:.3g} ph, {lam[i]:.3f} \\AA", fontsize=6)
            a.set_xlabel(f"sum {stack[i].sum():.3g}", fontsize=5)
    fig.suptitle(f"mosaic spread {tag.removeprefix('mos')}$^\\circ$, noiseless Bragg, 61 px crops",
                 fontsize=7)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    a = ap.parse_args()
    if not a.src.exists():
        raise SystemExit(f"{a.src} is missing; copy it from the sampling_check output directory")
    z = np.load(a.src)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in sorted({k.split("_")[0] for k in z.files}):
        fig = draw(z, tag)
        paths = figstyle.save(fig, OUT / f"sim_sampling_{tag.replace('.', 'p')}{a.suffix}")
        plt.close(fig)
        print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
