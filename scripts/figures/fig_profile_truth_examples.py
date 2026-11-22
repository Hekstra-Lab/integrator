"""Shoeboxes of the datsimx sims beside their true profile and each arm's predicted profile and gate.

One reflection per row, chosen by truth intensity $I_\\text{true}$ (rows run
from faint to bright), from `scripts/sim_truth/dump_profile_truth_examples.py`.
Two figures per dataset:

- **profiles**: the counts the model read; the noiseless expected rate on the
  window (`sim_image` plus `background`); the true profile, `sim_image` on the
  pixels whose true $(h, k, l)$ is this reflection's, normalised to sum to one
  on the window; then each arm's `qp_mean`. The true profile and the arms'
  profiles share one colour scale per row, from 0 to the largest of them, so
  a profile wider than the truth shows as fainter. Panel titles carry
  $n_\\text{eff} = 1/\\sum p^2$ on the window.
- **gates**: the counts; the true signal fraction, the expected rate of this
  reflection over the expected rate of everything on that pixel,
  `sim_image * own / (sim_image + background)`, which is the quantity
  `signal_prob` estimates; then each arm's `signal_prob` (laue.integrate,
  which has no gate, is left out). All on the fixed
  range 0 to 1, with the sum over the window in the corner.

A third figure draws, per bin of $I_\\text{true}$, the median $n_\\text{eff}$
of each arm's `qp_mean` over every in-domain reflection beside the median
$n_\\text{eff}$ of the truth table, which is computed over the whole detector
rather than the window (the 25 x 25 window holds 98 % of the true photons).

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_profile_truth_examples.py \\
            --data docs/figures/data/profile_truth_examples_mos010_pc_default.npz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "sim"


def neff(p) -> float:
    return float(1.0 / np.sum(np.asarray(p, float) ** 2))


def bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def draw_profiles(b, arms, out: Path) -> None:
    n = len(b["i_true"])
    cols = ["counts", "expected rate", "true profile"] + list(arms)
    fig, axes = plt.subplots(n, len(cols), squeeze=False, layout="constrained",
                             figsize=figstyle.figsize("full", aspect=n / len(cols) * 1.08))
    cm = figstyle.cmap()
    for i in range(n):
        tp = b["true_profile"][i]
        profs = [tp] + [b[f"{a}_profile"][i] for a in arms]
        vmax = max(float(p.max()) for p in profs)
        rate = b["bragg"][i] + b["background"][i]
        imgs = [(b["counts"][i], None, None, ""), (rate, None, None, ""),
                (tp, 0, vmax, f"$n_\\mathrm{{eff}}$ {neff(tp):.0f}")]
        imgs += [(b[f"{a}_profile"][i], 0, vmax, f"$n_\\mathrm{{eff}}$ {neff(b[f'{a}_profile'][i]):.0f}")
                 for a in arms]
        for j, (img, lo, hi, note) in enumerate(imgs):
            ax = axes[i][j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            bare(ax)
            if i == 0:
                ax.set_title(cols[j], fontsize=6)
            if note:
                ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=4.5, va="top", ha="left",
                        color=figstyle.roles()["ink"])
        axes[i][0].set_ylabel(f"$I_\\mathrm{{true}}$ {b['i_true'][i]:.0f}\n$d$ {b['d'][i]:.2f} \\AA", fontsize=5)
    fig.suptitle(f"{b['dataset']}: counts, truth and predicted profile (qp_mean)", fontsize=7)
    for p in figstyle.save(fig, out):
        print(p)
    plt.close(fig)


def draw_gates(b, arms, out: Path) -> None:
    n = len(b["i_true"])
    arms = [a for a in arms if f"{a}_gate" in b]
    cols = ["counts", "true signal fraction"] + [f"{a} gate" for a in arms]
    fig, axes = plt.subplots(n, len(cols), squeeze=False, layout="constrained",
                             figsize=figstyle.figsize("full", aspect=n / len(cols) * 1.08))
    cm = figstyle.cmap()
    for i in range(n):
        frac = b["bragg"][i] * b["own"][i] / np.maximum(b["bragg"][i] + b["background"][i], 1e-30)
        imgs = [(b["counts"][i], None, None), (frac, 0, 1)]
        imgs += [(b[f"{a}_gate"][i], 0, 1) for a in arms]
        for j, (img, lo, hi) in enumerate(imgs):
            ax = axes[i][j]
            ax.imshow(img, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            bare(ax)
            if i == 0:
                ax.set_title(cols[j], fontsize=6)
            if j >= 1:
                ax.text(0.03, 0.97, f"$\\sum$ {float(img.sum()):.0f}", transform=ax.transAxes, fontsize=4.5,
                        va="top", ha="left", color=figstyle.roles()["ink"])
        axes[i][0].set_ylabel(f"$I_\\mathrm{{true}}$ {b['i_true'][i]:.0f}\n$d$ {b['d'][i]:.2f} \\AA", fontsize=5)
    fig.suptitle(f"{b['dataset']}: counts, true signal fraction and gate (signal_prob), 0 to 1", fontsize=7)
    for p in figstyle.save(fig, out):
        print(p)
    plt.close(fig)


def draw_neff(csv: Path, dataset: str, out: Path) -> None:
    import polars as pl

    df = pl.read_csv(csv)
    arms = df["method"].unique(maintain_order=True).to_list()
    colour = figstyle.arm_colors(arms)
    fig, ax = plt.subplots(figsize=figstyle.figsize("full", aspect=0.42), layout="constrained")
    for a in arms:
        d = df.filter(pl.col("method") == a).sort("i_true_lo")
        x = np.sqrt(d["i_true_lo"].to_numpy().clip(min=0.3) * d["i_true_hi"].to_numpy().clip(max=1e4))
        ax.plot(x, d["n_eff_model_median"].to_numpy(), color=colour[a], marker="o", ms=2, lw=1, label=a)
    d = df.filter(pl.col("method") == arms[0]).sort("i_true_lo")
    x = np.sqrt(d["i_true_lo"].to_numpy().clip(min=0.3) * d["i_true_hi"].to_numpy().clip(max=1e4))
    ax.plot(x, d["n_eff_true_median"].to_numpy(), **{**figstyle.reference_style(), "marker": "s", "ms": 2,
                                                     "label": "truth"})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$I_\\mathrm{true}$ bin (photons)")
    ax.set_ylabel("median $n_\\mathrm{eff} = 1/\\sum p^2$ (pixels)")
    ax.set_title(dataset, fontsize=7)
    ax.legend(fontsize=5)
    for p in figstyle.save(fig, out):
        print(p)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, nargs="+",
                    default=sorted(DATA.glob("profile_truth_examples_*.npz")))
    a = ap.parse_args()
    if not a.data:
        raise SystemExit(f"no profile_truth_examples_*.npz under {DATA}; run dump_profile_truth_examples.py")
    figstyle.use()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in a.data:
        b = dict(np.load(path, allow_pickle=False))
        b["dataset"] = str(b["dataset"])
        arms = [str(x) for x in b["arms"]]
        draw_profiles(b, arms, OUT / f"profile_truth_examples_{b['dataset']}")
        draw_gates(b, arms, OUT / f"gate_truth_examples_{b['dataset']}")
        csv = path.with_name(f"profile_neff_vs_truth_{b['dataset']}.csv")
        if csv.is_file():
            draw_neff(csv, b["dataset"], OUT / f"profile_neff_vs_truth_{b['dataset']}")


if __name__ == "__main__":
    main()
