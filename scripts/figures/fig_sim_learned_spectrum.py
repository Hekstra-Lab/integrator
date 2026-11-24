"""Learned Wilson spectrum G(lambda) of trained runs against the simulation's spectrum and its truth table.

Reads `spectrum_vs_truth.csv` and `wilson_B.csv` of
`scripts/sim_truth/learned_params_vs_truth.py`. Three panels sharing the
wavelength axis:

- G(lambda) on a log scale: each run's learned G (line); the truth's
  G(lambda | B) = mean of I_true exp(B / (2 d^2)) over the shoebox reflections
  in a 0.005 A bin, at the first run's B (ink markers); the datsimx source
  spectrum (muted line) and the `--enSteps` points datsimx simulated on it
  (small ink dots, on the line by construction), multiplied by the one constant that makes the count-weighted
  mean of log(G_true / spectrum) zero;
- learned G / G_true(lambda | B), each run at its own B;
- the number of shoebox reflections per 0.002 A.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_learned_spectrum.py --src DIR [--tag _t0]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    s = pl.read_csv(a.src / "spectrum_vs_truth.csv", infer_schema_length=None)
    Bt = dict(pl.read_csv(a.src / "wilson_B.csv").select("method", "B").iter_rows())
    runs = list(dict.fromkeys(s.filter(kind="G_learned")["method"].to_list()))
    dataset = s["dataset"][0]

    figstyle.use()
    roles = figstyle.roles()
    col = figstyle.arm_colors([LABEL.get(r, r) for r in runs])
    fig, ax = plt.subplots(3, 1, figsize=figstyle.figsize("full", aspect=0.95), sharex=True,
                           layout="constrained", height_ratios=[2.2, 1.2, 0.8])

    spec = s.filter(kind="spectrum").sort("wavelength")
    samp = s.filter(kind="spectrum_sample").sort("wavelength")
    ref = s.filter(kind="G_truth_given_B", method=runs[0]).sort("wavelength")
    sv = np.interp(ref["wavelength"].to_numpy(), spec["wavelength"].to_numpy(), spec["value"].to_numpy(), left=0, right=0)
    ok = sv > 0
    w = ref["n"].to_numpy()[ok]
    scale = np.exp(np.sum(w * np.log(ref["value"].to_numpy()[ok] / sv[ok])) / w.sum())
    ax[0].plot(spec["wavelength"], spec["value"] * scale, color=roles["muted"], lw=1.0,
               label=f"source spectrum $\\times$ {scale:.3g}")
    ax[0].plot(samp["wavelength"], samp["value"] * scale, ".", color=roles["ink"], ms=1.2,
               label=f"{samp.height} simulated energies")
    ax[0].plot(ref["wavelength"], ref["value"], "o", color=roles["ink"], ms=3, mfc="none",
               label=f"truth, $B = {Bt[runs[0]]:.1f}$")
    for r in runs:
        g = s.filter(kind="G_learned", method=r).sort("wavelength")
        lab = LABEL.get(r, r)
        ax[0].plot(g["wavelength"], g["value"], color=col[lab], lw=1.1, label=f"{lab}, $B = {Bt[r]:.1f}$")
        t = s.filter(kind="G_truth_given_B", method=r).sort("wavelength")
        gl = np.interp(t["wavelength"].to_numpy(), g["wavelength"].to_numpy(), g["value"].to_numpy())
        ax[1].plot(t["wavelength"], gl / t["value"].to_numpy(), "-o", color=col[lab], ms=2, lw=0.9)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("$G(\\lambda)$ (photons)")
    ax[0].legend(fontsize="x-small", ncol=2, frameon=False)
    ax[1].axhline(1, **figstyle.reference_style())
    ax[1].set_ylabel("learned / truth")
    c = s.filter(kind="lambda_count").sort("wavelength")
    ax[2].step(c["wavelength"], c["value"], where="mid", color=roles["ink"], lw=0.8)
    ax[2].set_ylabel("reflections")
    ax[2].set_xlabel("wavelength (\\AA)")
    for p in figstyle.save(fig, OUT / f"sim_learned_spectrum_{dataset}{a.tag}"):
        print(p)


if __name__ == "__main__":
    main()
