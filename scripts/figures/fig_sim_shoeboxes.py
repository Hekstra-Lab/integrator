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
        python scripts/figures/fig_sim_shoeboxes.py [--src NPZ --out NAME]

An npz written with --same-reflections is drawn transposed: one row per
reflection, one column per dataset, the same reflection (image, h k l) across
each row on one grey scale, from the row's 1st percentile to its maximum, so
the datasets can be compared pixel for pixel. Its titles are the laue I/sigma
each dataset's own laue-dials run gave that reflection.
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
    "mos010_pc_default_mosaic": "mosaic spread 0.10°",
    "mos025_pc_m0_default_mosaic": "mosaic spread 0.25°",
}


def draw_same(z, names, out_name) -> None:
    nrow = len(z["targets"])
    fig, axes = plt.subplots(nrow, len(names), figsize=figstyle.figsize("half", aspect=nrow / len(names) * 1.05),
                             layout="constrained", squeeze=False)
    for i in range(nrow):
        imgs = [np.where(z[f"{n}_mask"][i], z[f"{n}_counts"][i], np.nan) for n in names]
        lo = min(np.nanpercentile(im, 1) for im in imgs)
        hi = max(np.nanmax(im) for im in imgs)
        for j, (n, img) in enumerate(zip(names, imgs)):
            a = axes[i, j]
            im = a.imshow(img, cmap="gray_r", vmin=lo, vmax=max(hi, lo + 1), interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.grid(visible=False)
            a.set_title(f"I/σ {z[f'{n}_isigi'][i]:.1f}", fontsize=5)
            if i == 0:
                a.set_title(f"{LABELS.get(n, n)}\nI/σ {z[f'{n}_isigi'][i]:.1f}", fontsize=5)
        n0 = names[0]
        axes[i, 0].set_ylabel(f"({z[f'{n0}_h'][i]} {z[f'{n0}_k'][i]} {z[f'{n0}_l'][i]}), image {z[f'{n0}_image_num'][i]}\n"
                              f"d {z[f'{n0}_d'][i]:.2f} Å", fontsize=5)
        cb = fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.8, ticks=[lo, hi])
        cb.ax.tick_params(labelsize=4)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in figstyle.save(fig, OUT / out_name):
        print(p)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=None, help="output name (default sim_shoeboxes_compare, or "
                                                "sim_shoeboxes_same for a --same-reflections npz)")
    a = ap.parse_args()
    src = Path(a.src)
    if not src.exists():
        raise SystemExit(f"{src} is missing; run dump_shoebox_examples.py on the cluster and copy it here")
    z = np.load(src)
    names = [str(n) for n in z["names"]]
    ncol = len(z["targets"])
    figstyle.use()
    if "same_reflections" in z.files and int(z["same_reflections"]):
        draw_same(z, names, a.out or "sim_shoeboxes_same")
        return
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
    for p in figstyle.save(fig, OUT / (a.out or "sim_shoeboxes_compare")):
        print(p)


if __name__ == "__main__":
    main()
