"""True spot profiles beside each model's prior profile and Laplace-mode profile, and their averages by resolution.

Reads `profile_compare_<dataset>.npz` of `scripts/sim_truth/dump_profile_compare.py`.

Figure `sim_profile_examples_<dataset>`: six sampled reflections (rows), chosen as
the nearest to 20, 60, 200, 700, 2000 and 6000 true photons; columns are the
shoebox counts, the true profile, and for each model its prior profile (what a
reflection too faint to move its shape is integrated with) and its Laplace-mode
profile (the shape fitted to this reflection's counts under that prior). Each
panel on its own scale; the row label gives I_true and d.

Figure `sim_profile_means_<dataset>`: every profile rotated by phi (the streak
azimuth; ndimage.rotate with rows pointing down) about the window centre, so the streak runs along x, and averaged over
the reflections in each resolution bin (columns). Top row: the mean true
profile; then, per model, the mean prior profile minus the mean true profile on
a diverging scale symmetric about 0, with kappa = sum q p / sum q^2 of the two
means printed.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_profile_compare.py --src docs/figures/data/profile_compare_mos010_pc_default.npz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "sim"
TARGETS = (20, 60, 200, 700, 2000, 6000)
D_EDGES = (1.8, 2.0, 2.5, 3.0, 4.0, 99.0)
NAMES = {"B1": "B1", "geom": "geometric"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    a = ap.parse_args()
    figstyle.use()
    z = np.load(a.src, allow_pickle=True)
    dataset = a.src.stem.replace("profile_compare_", "")
    models = [str(m) for m in z["models"]]
    I, d, phi = z["i_true"], z["d"], z["phi"]
    cm, dm = figstyle.cmap(), figstyle.diverging_cmap()

    picks = [int(np.argmin(np.abs(np.log(I) - np.log(t)))) for t in TARGETS]
    cols = [("counts", z["counts"]), ("true profile", z["true_profile"])]
    for m in models:
        cols += [(f"{NAMES.get(m, m)} prior", z[f"{m}_prior"]), (f"{NAMES.get(m, m)} mode", z[f"{m}_mode"])]
    fig, ax = plt.subplots(len(picks), len(cols), figsize=figstyle.figsize("full", aspect=0.17 * len(picks)),
                           layout="constrained", squeeze=False)
    for i, j in enumerate(picks):
        for k, (lab, arr) in enumerate(cols):
            ax[i, k].imshow(arr[j], cmap=cm, interpolation="nearest")
            ax[i, k].set_xticks([])
            ax[i, k].set_yticks([])
            if i == 0:
                ax[i, k].set_title(lab, fontsize="xx-small")
        ax[i, 0].set_ylabel(f"{I[j]:.0f} ph\n{d[j]:.2f} A", fontsize="xx-small")
    for p in figstyle.save(fig, OUT / f"sim_profile_examples_{dataset}"):
        print(p)

    def aligned(arr):
        return np.stack([ndimage.rotate(x, np.degrees(f), reshape=False, order=1, mode="constant") for x, f in zip(arr, phi)])

    true_al = aligned(z["true_profile"])
    pri_al = {m: aligned(z[f"{m}_prior"]) for m in models}
    nb = len(D_EDGES) - 1
    fig, ax = plt.subplots(1 + len(models), nb, figsize=figstyle.figsize("full", aspect=0.2 * (1 + len(models))),
                           layout="constrained", squeeze=False)
    for b in range(nb):
        sel = (d >= D_EDGES[b]) & (d < D_EDGES[b + 1])
        pt = true_al[sel].mean(0)
        ax[0, b].imshow(pt, cmap=cm, interpolation="nearest")
        ax[0, b].set_title(f"{D_EDGES[b]:g}-{D_EDGES[b + 1]:g} A (n {sel.sum()})" if D_EDGES[b + 1] < 50
                           else f"> {D_EDGES[b]:g} A (n {sel.sum()})", fontsize="xx-small")
        for r, m in enumerate(models, start=1):
            q = pri_al[m][sel].mean(0)
            diff = q - pt
            lim = np.abs(diff).max()
            ax[r, b].imshow(diff, cmap=dm, vmin=-lim, vmax=lim, interpolation="nearest")
            kappa = float((q * pt).sum() / (q * q).sum())
            ax[r, b].text(0.03, 0.04, f"kappa {kappa:.3f}", transform=ax[r, b].transAxes, fontsize="xx-small")
            if b == 0:
                ax[r, b].set_ylabel(f"{NAMES.get(m, m)} prior\n- truth", fontsize="xx-small")
        if b == 0:
            ax[0, b].set_ylabel("mean true\nprofile", fontsize="xx-small")
    for x in ax.ravel():
        x.set_xticks([])
        x.set_yticks([])
    for p in figstyle.save(fig, OUT / f"sim_profile_means_{dataset}"):
        print(p)


if __name__ == "__main__":
    main()
