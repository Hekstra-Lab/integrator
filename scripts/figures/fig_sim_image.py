"""Simulated Laue shots side by side: whole detector (block means) and a full-resolution crop.

Columns are the panels of `docs/figures/data/sim_image_view.npz` (written by
`scripts/sim_truth/dump_image_view.py`), typically the laue-dials paper's
hkl_data shot 1, the datsimx paper-matched noisy shot and its noiseless Bragg
image. Photons per pixel throughout. Noisy panels share one grey scale per row,
from the 50th to the 99.9th percentile of the paper panel; the noiseless panel
has its own scale.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_image.py
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
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} is missing; run dump_image_view.py on the cluster and copy it here")
    z = np.load(SRC)
    names = [str(n) for n in z["names"]]
    pool, crop = int(z["pool"]), int(z["crop"])
    ref = names[0]

    figstyle.use()
    fig, axes = plt.subplots(2, len(names), figsize=figstyle.figsize("full", aspect=2 / len(names) * 1.05),
                             layout="constrained", squeeze=False)
    for row, key in enumerate(("pooled", "crop")):
        r = z[f"{ref}_{key}"]
        lo, hi = np.percentile(r, 50), np.percentile(r, 99.9)
        for j, n in enumerate(names):
            img = z[f"{n}_{key}"]
            noiseless = str(z[f"{n}_kind"]) == "noiseless"
            vmin, vmax = (0.0, max(np.percentile(img, 99.9), 1e-6)) if noiseless else (lo, hi)
            a = axes[row, j]
            im = a.imshow(img, cmap="gray_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            a.grid(visible=False)
            if row == 0:
                a.set_title(LABELS.get(n.split("_")[0], n), fontsize=6)
            fig.colorbar(im, ax=a, shrink=0.8, label="photons / px" if j == len(names) - 1 else None)
        axes[row, 0].set_ylabel(f"whole detector\n{pool}x{pool} px means" if key == "pooled"
                                else f"{crop}x{crop} px crop\nbeam centre + (500, 500) px", fontsize=6)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in figstyle.save(fig, OUT / "sim_image_paper_vs_datsimx"):
        print(p)


if __name__ == "__main__":
    main()
