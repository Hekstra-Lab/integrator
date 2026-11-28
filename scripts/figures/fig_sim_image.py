"""Simulated Laue shots side by side: whole detector (block means) and a full-resolution crop.

Columns are the panels of `docs/figures/data/sim_image_view.npz` (written by
`scripts/sim_truth/dump_image_view.py`), typically the laue-dials paper's
hkl_data shot 1, the datsimx paper-matched noisy shot and its noiseless Bragg
image. Photons per pixel throughout. Noisy panels share one grey scale per row,
from the 50th to the 99.9th percentile of the paper panel; the noiseless panel
has its own scale.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_image.py [--src NPZ --out NAME --shared-noiseless]

--shared-noiseless puts every noiseless panel of a row on one scale, 0 to the
largest 99.9th percentile among them, so their brightness can be compared (two
runs that differ in one parameter); the default gives each its own. An npz with
`centre_xy` (dump_image_view.py --centre) labels the crop by that pixel.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "figures" / "data" / "sim_image_view.npz"
OUT = ROOT / "docs" / "figures" / "sim"
LABELS = {
    "paper": "laue-dials paper sim\n(hkl_data shot 1, noisy)",
    "datsimx": "datsimx, paper-matched\n(noisy)",
    "truth": "datsimx\n(noiseless Bragg)",
    "datsimx_nodiv": "datsimx, paper-matched\nno divergence (noisy)",
    "datsimx_div": "datsimx, paper-matched\ndiv 0.1 mrad, 4 steps (noisy)",
    "mos005truth": "mosaic spread 0.05°\n(noiseless Bragg)",
    "mos050truth": "mosaic spread 0.50°\n(noiseless Bragg)",
    "mos005pc": "mosaic spread 0.05°\n(photon counting)",
    "mos050pc": "mosaic spread 0.50°\n(photon counting)",
    "mos010truth": "mosaic spread 0.10°\n(noiseless Bragg)",
    "mos025truth": "mosaic spread 0.25°\n(noiseless Bragg)",
    "mos010pc": "mosaic spread 0.10°\n(photon counting)",
    "mos025pc": "mosaic spread 0.25°\n(photon counting)",
}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default="sim_image_paper_vs_datsimx")
    ap.add_argument("--shared-noiseless", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"{src} is missing; run dump_image_view.py on the cluster and copy it here")
    z = np.load(src)
    names = [str(n) for n in z["names"]]
    pool, crop = int(z["pool"]), int(z["crop"])
    ref = names[0]

    figstyle.use()
    fig, axes = plt.subplots(2, len(names), figsize=figstyle.figsize("full", aspect=2 / len(names) * 1.05),
                             layout="constrained", squeeze=False)
    for row, key in enumerate(("pooled", "crop")):
        r = z[f"{ref}_{key}"]
        lo, hi = np.percentile(r, 50), np.percentile(r, 99.9)
        nl = [z[f"{n}_{key}"] for n in names if str(z[f"{n}_kind"]) == "noiseless"]
        shared_hi = max([np.percentile(x, 99.9) for x in nl] + [1e-6])
        for j, n in enumerate(names):
            img = z[f"{n}_{key}"]
            noiseless = str(z[f"{n}_kind"]) == "noiseless"
            vmin, vmax = ((0.0, shared_hi if args.shared_noiseless else max(np.percentile(img, 99.9), 1e-6))
                          if noiseless else (lo, hi))
            a = axes[row, j]
            im = a.imshow(img, cmap="gray_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.grid(visible=False)
            if row == 0:
                a.set_title(LABELS.get(n, LABELS.get(n.split("_")[0], n)), fontsize=6)
            fig.colorbar(im, ax=a, shrink=0.8, label="photons / px" if j == len(names) - 1 else None)
        where = (f"centred on ({z['centre_xy'][0]:.0f}, {z['centre_xy'][1]:.0f}) px" if "centre_xy" in z.files
                 else "beam centre + (500, 500) px")
        axes[row, 0].set_ylabel(f"whole detector\n{pool}x{pool} px means" if key == "pooled"
                                else f"{crop}x{crop} px crop\n{where}", fontsize=6)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in figstyle.save(fig, OUT / args.out):
        print(p)


if __name__ == "__main__":
    main()
