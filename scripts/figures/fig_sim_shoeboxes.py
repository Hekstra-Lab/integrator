"""Example shoeboxes of several simulated datasets, one row per dataset, columns at the same laue I/sigma.

Reads `docs/figures/data/shoebox_examples.npz` from
`scripts/sim_truth/dump_shoebox_examples.py`. Each panel is the central
box x box of one reflection in the units its dataset stores (ADU for the
laue-dials paper sims, photons for datsimx), on its own grey scale from the
box's 1st to its maximum value; masked pixels are blank. The panel title is
laue's I/sigma and d; the row label carries the dataset's stored dtype, the
fraction of non-integer stored values, and the median border level and
border variance / mean over 5,000 random boxes.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_shoeboxes.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "figures" / "data" / "shoebox_examples.npz"
OUT = ROOT / "docs" / "figures" / "sim"
LABELS = {
    "paper": "laue-dials paper sim (ADU)",
    "mos010_wrapper": "datsimx mos010, our nanoBragg\nadd_noise wrapper (photons)",
    "mos010_pc_default": "datsimx mos010, nanoBragg-noise\nphoton-counting, default (photons)",
    "mos010_pc_all": "datsimx mos010, nanoBragg-noise\nphoton-counting, permissive (photons)",
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} is missing; run dump_shoebox_examples.py on the cluster and copy it here")
    z = np.load(SRC)
    names = [str(n) for n in z["names"]]
    ncol = len(z["targets"])
    figstyle.use()
    fig, axes = plt.subplots(len(names), ncol, figsize=figstyle.figsize("full", aspect=len(names) / ncol * 1.32),
                             layout="constrained", squeeze=False)
    for i, n in enumerate(names):
        cs, ms = z[f"{n}_counts"], z[f"{n}_mask"]
        for j in range(ncol):
            a = axes[i, j]
            img = np.where(ms[j], cs[j], np.nan)
            lo, hi = np.nanpercentile(img, 1), np.nanmax(img)
            im = a.imshow(img, cmap="gray_r", vmin=lo, vmax=max(hi, lo + 1), interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.grid(visible=False)
            a.set_title(f"I/σ {z[f'{n}_isigi'][j]:.1f}, d {z[f'{n}_d'][j]:.2f} Å", fontsize=5)
            cb = fig.colorbar(im, ax=a, shrink=0.75, ticks=[lo, hi])
            cb.ax.tick_params(labelsize=4)
        axes[i, 0].set_ylabel(
            f"{LABELS.get(n, n)}\n{z[f'{n}_dtype']}, non-integer {float(z[f'{n}_nonint']):.3f}\n"
            f"border {float(z[f'{n}_border_median']):.1f}, var/mean {float(z[f'{n}_border_var_over_mean']):.2f}",
            fontsize=5)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in figstyle.save(fig, OUT / "sim_shoeboxes_compare"):
        print(p)


if __name__ == "__main__":
    main()
