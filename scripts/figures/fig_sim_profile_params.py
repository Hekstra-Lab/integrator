"""Learned profile parameters of trained runs: the K-component widths and weights against geometry, and residual bases.

Reads `kmix_params.csv` and `residual_basis.npz` of
`scripts/sim_truth/learned_params_vs_truth.py`.

Figure `sim_kmix_params_<dataset>`: for each `streak_kmix_profile` or
`streak_kmix_v2_profile` run, the metadata prior mean theta0 + f(m) unpacked into
the along-streak and across-streak standard deviations a_k, c_k (px, before the
1/12 pixel variance is added) and the weights w_k of the K components; rows are the
three quantities, columns detector radius from the beam centre and wavelength.
Lines are bin medians, bands the interquartile range; colour is the run, line
style the component (solid, dashed, dotted for k = 0, 1, 2).

Figure `sim_residual_basis_<dataset>`: the columns of each trained residual basis
W (625 x latent) as 25 x 25 images on a diverging scale symmetric about 0, one row
per run, each row on its own scale.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_profile_params.py --src DIR [--tag _t0]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "sim"
LABEL = {"svae": "svae", "svae_streak": "N1", "svae_kmix_base": "T0",
         "svae_kmix_h128_grid05": "P2", "svae_kmixv2_post": "B1",
         "svae_kmix_base_cheb40": "T0, G degree 40", "svae_kmix_base_cheb40_cos": "T0-40, cosine",
         "svae_kmix_base_cheb40_cos_lr3e3_b1024": "T0-40, cosine, lr 3e-3", "svae_kmix_base_cheb40_bgmap": "T0-40, background map"}
LS = ("-", "--", ":", "-.")
ROWS = (("along_sd", "along sd $a_k$ (px)"), ("across_sd", "across sd $c_k$ (px)"), ("weight", "weight $w_k$"))
COLS = (("radius_px", "detector radius (px)"), ("wavelength", "wavelength (\\AA)"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    figstyle.use()

    k = pl.read_csv(a.src / "kmix_params.csv")
    dataset = k["dataset"][0]
    runs = list(dict.fromkeys(k["method"].to_list()))
    col = figstyle.arm_colors([LABEL.get(r, r) for r in runs])
    fig, ax = plt.subplots(3, 2, figsize=figstyle.figsize("full", aspect=0.95), sharex="col",
                           sharey="row", layout="constrained")
    for i, (param, ylab) in enumerate(ROWS):
        for j, (axis, xlab) in enumerate(COLS):
            for r in runs:
                lab = LABEL.get(r, r)
                sub = k.filter(method=r, param=param, axis=axis)
                for c in sorted(sub["component"].unique().to_list()):
                    d = sub.filter(component=c).sort("x")
                    ax[i, j].plot(d["x"], d["median"], LS[c % len(LS)], color=col[lab], lw=1.0,
                                  label=lab if (i, j, c) == (0, 0, 0) else None)
                    ax[i, j].fill_between(d["x"], d["q25"], d["q75"], color=col[lab], alpha=0.12, lw=0)
            ax[2, j].set_xlabel(xlab)
        ax[i, 0].set_ylabel(ylab)
    ax[0, 0].legend(fontsize="x-small", frameon=False)
    for p in figstyle.save(fig, OUT / f"sim_kmix_params_{dataset}{a.tag}"):
        print(p)

    z = np.load(a.src / "residual_basis.npz")
    if not z.files:
        return
    n = max(z[f].shape[0] for f in z.files)
    fig, ax = plt.subplots(len(z.files), n, figsize=figstyle.figsize("full", aspect=0.1 * len(z.files) + 0.02),
                           layout="constrained", squeeze=False)
    cm = figstyle.diverging_cmap()
    for i, f in enumerate(z.files):
        W = z[f]
        v = np.abs(W).max()
        for j in range(n):
            ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
            if j < W.shape[0]:
                ax[i, j].imshow(W[j], cmap=cm, vmin=-v, vmax=v, interpolation="nearest")
            else:
                ax[i, j].axis("off")
        ax[i, 0].set_ylabel(LABEL.get(f, f))
    for p in figstyle.save(fig, OUT / f"sim_residual_basis_{dataset}{a.tag}"):
        print(p)


if __name__ == "__main__":
    main()
