"""Merged structure factors and careless's per-observation scale against the simulation's truth.

Reads, from --src, the tables of `scripts/sim_truth/score_merged_vs_truth.py`:
`scores_merged.csv` (--out), `scores_scale.csv` (--scale-out) and, when present,
`merged_joined_<method>.parquet` (--joined-out). One colour per method.

Figure `sim_merged_vs_truth_<dataset>`, against 1/d^2 at each shell's median d:

- cc_F, the correlation of merged F with F_true (dashed), and cc_F_kB, the same
  after removing the fitted scale k and B (solid);
- r_F = sum |F' - F_true| / sum F_true after that fit;
- cc_I, the correlation of merged I with F_true^2;
- cc_dano, the correlation of F(+) - F(-) with the true difference (solid), and
  cc_dano_crossed (dotted), which must mirror it about 0;
- completeness against the truth;
- per reflection, log10 F' against log10 F_true for the first method, as a 2-D
  histogram, with the identity line.

Figure `sim_scale_vs_truth_<dataset>`, careless's per-observation `Scale` against
I_true / |F_true|^2 with the same k and B removed, per 0.01 A of wavelength: the
sd of the log ratio, its mean, the correlation of the logs, and the mean of
log(Ipred / I_true); the last panel gives the sd and mean of the log scale ratio
per decade of I_true.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_sim_merged_vs_truth.py --src DIR [--tag _ref]
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
LABEL = {"laue_integrate": "laue"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    figstyle.use()
    roles = figstyle.roles()
    ref = figstyle.reference_style()

    m = pl.read_csv(a.src / "scores_merged.csv", infer_schema_length=None)
    dataset = m["dataset"][0]
    methods = list(dict.fromkeys(m["method"].to_list()))
    from arms import colours, label as arm_label

    col = colours(methods)
    sh = m.filter(pl.col("shell") != "all").with_columns(
        x=1 / ((pl.col("d_min") + pl.col("d_max").clip(upper_bound=6.0)) / 2) ** 2)

    fig, ax = plt.subplots(2, 3, figsize=figstyle.figsize("full", aspect=0.62), layout="constrained")
    ax = ax.ravel()
    for meth in methods:
        g = sh.filter(method=meth).sort("x")
        c = col[meth]
        lab = arm_label(meth)
        ax[0].plot(g["x"], g["cc_F_kB"], "-o", color=c, ms=2.5, lw=1, label=lab)
        ax[0].plot(g["x"], g["cc_F"], "--", color=c, lw=0.8)
        ax[1].plot(g["x"], g["r_F"], "-o", color=c, ms=2.5, lw=1)
        if "cc_I" in g.columns:
            ax[2].plot(g["x"], g["cc_I"], "-o", color=c, ms=2.5, lw=1)
        ax[3].plot(g["x"], g["cc_dano"], "-o", color=c, ms=2.5, lw=1)
        ax[3].plot(g["x"], g["cc_dano_crossed"], ":", color=c, lw=0.8)
        ax[4].plot(g["x"], g["completeness"], "-o", color=c, ms=2.5, lw=1)
    ax[3].axhline(0, **ref)
    for i, yl in enumerate(["cc$(F, F_\\mathrm{true})$", "$R_F$ after $k$, $B$", "cc$(I, F_\\mathrm{true}^2)$",
                            "cc$(\\Delta F, \\Delta F_\\mathrm{true})$", "completeness"]):
        ax[i].set_ylabel(yl)
        ax[i].set_xlabel("$1/d^2$ (\\AA$^{-2}$)")
    jp = a.src / f"merged_joined_{methods[0]}.parquet"
    if jp.exists():
        j = pl.read_parquet(jp).filter((pl.col("F_kB") > 0) & (pl.col("F_true") > 0))
        x, y = np.log10(j["F_true"].to_numpy()), np.log10(j["F_kB"].to_numpy())
        lo, hi = np.quantile(np.r_[x, y], [0.001, 0.999])
        ax[5].hist2d(x, y, bins=120, range=[[lo, hi], [lo, hi]], cmap=figstyle.cmap(), cmin=1)
        ax[5].plot([lo, hi], [lo, hi], **ref)
        ax[5].set_xlabel("$\\log_{10} F_\\mathrm{true}$")
        ax[5].set_ylabel(f"$\\log_{{10}} F'$, {LABEL.get(methods[0], methods[0])}")
    else:
        # no per-reflection table: the spare cell holds the legend, off the data
        ax[5].axis("off")
        h, lab = ax[0].get_legend_handles_labels()
        ax[5].legend(h, lab, loc="center", frameon=False, fontsize="small",
                     title="solid: after $k$, $B$; dashed: raw\n(cc$(\Delta F)$: dotted, crossed)")
    if jp.exists():
        ax[0].legend(fontsize="x-small", frameon=False)
    for p in figstyle.save(fig, OUT / f"sim_merged_vs_truth_{dataset}{a.tag}"):
        print(p)

    sp = a.src / "scores_scale.csv"
    if not sp.exists():
        return
    s = pl.read_csv(sp, infer_schema_length=None)
    fig, ax = plt.subplots(1, 4, figsize=figstyle.figsize("full", aspect=0.3), layout="constrained")
    for meth in list(dict.fromkeys(s["method"].to_list())):
        c = col.get(meth, roles["ink"])
        w = s.filter(method=meth, axis="wavelength").sort("lam_mean")
        ax[0].plot(w["lam_mean"], w["sd_log_ratio"], "-o", color=c, ms=2.5, lw=1, label=arm_label(meth))
        ax[0].plot(w["lam_mean"], w["cc_log"], "--", color=c, lw=0.8)
        ax[1].plot(w["lam_mean"], w["mean_log_ratio"], "-o", color=c, ms=2.5, lw=1)
        ax[2].plot(w["lam_mean"], w["mean_log_ipred"], "-o", color=c, ms=2.5, lw=1)
        ax[2].plot(w["lam_mean"], w["sd_log_ipred"], "--", color=c, lw=0.8)
        dd = s.filter(method=meth, axis="i_true_decade").with_columns(pl.col("bin").cast(int)).sort("bin")
        ax[3].plot(dd["bin"], dd["sd_log_ratio"], "-o", color=c, ms=2.5, lw=1)
        ax[3].plot(dd["bin"], dd["mean_log_ratio"], "--", color=c, lw=0.8)
    for x in ax[1:]:
        x.axhline(0, **ref)
    ax[0].set_ylabel("sd (solid), cc (dashed)\nof log scale vs truth")
    ax[1].set_ylabel("mean log(scale / truth)")
    ax[2].set_ylabel("log($I_\\mathrm{pred}/I_\\mathrm{true}$)\nmean (solid), sd (dashed)")
    ax[3].set_ylabel("log(scale / truth)\nsd (solid), mean (dashed)")
    for x in ax[:3]:
        x.set_xlabel("wavelength (\\AA)")
    ax[3].set_xlabel("$\\log_{10} I_\\mathrm{true}$ (photons)")
    ax[0].legend(fontsize="x-small", frameon=False)
    for p in figstyle.save(fig, OUT / f"sim_scale_vs_truth_{dataset}{a.tag}"):
        print(p)


if __name__ == "__main__":
    main()
