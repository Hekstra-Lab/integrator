"""Integrator, merging and refinement diagnostics for the mos010 careless -> phenix arms, as large-panel PDFs.

The per-observation scatters (intensity, background, noise / signal) are drawn by
fig_sim_truth_scatter.py --quantity; this script draws the merging and refinement figures.
One PDF per diagnostic, in --out (default docs/figures/sim/diagnostics/), one panel
per arm where the diagnostic is per arm. Built with `readable.py`: fixed panel size,
not sized to a page.

  careless    every careless.stats statistic by resolution shell, one panel each, all arms
  refinement  phenix's final per-shell R-work and R-free, FOM and phase error, and
              CC_work / CC_free from refined_2.mtz beside CC* = sqrt(2 CC_1/2 / (1 + CC_1/2))
              of careless's CC_1/2 (Pearson), against 1/d^2 (Karplus & Diederichs 2012)
  steps       R-work and R-free at each phenix macro-cycle stage
  summary     final R-work and R-free per arm

Inputs, under --data (docs/figures/data by default): sim_downstream_joined_mos010_pc.csv
(method, dataset, i_true, in_domain, gain, I, SIGI, d), sim_bg_rows_mos010_pc.parquet
(dump_profile_truth_examples.py --bg-rows-out), sim_careless_stats_mos010_pc.csv and
sim_downstream/ (collect_downstream.py: downstream_refine.csv, downstream_refine_shells.csv,
downstream_refine_steps.csv, downstream_ccwork.csv), all seed 1.

    uv run python scripts/figures/fig_sim_diagnostics.py [--only intensity,noise]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

import readable as R

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "sim" / "diagnostics"
ALL = ("careless", "refinement", "steps", "summary")


def log_bins_median(x, y, edges, min_n=20):
    idx = np.digitize(x, edges)
    cx, cy = [], []
    for i in range(1, len(edges)):
        s = idx == i
        if s.sum() >= min_n:
            cx.append(np.sqrt(edges[i - 1] * edges[i]))
            cy.append(np.median(y[s]))
    return np.array(cx), np.array(cy)


def d_mid(shell: str) -> float:
    a, b = (float(x) for x in re.findall(r"[\d.]+", shell)[:2])
    return 0.5 * (a + b)


def fig_careless(c: pl.DataFrame, out: Path, dataset: str) -> None:
    c = c.with_columns(pl.col("correlation").fill_null(""))
    c = c.filter(pl.col("shell").str.contains("-"))
    keys = c.select("tool", "statistic", "correlation").unique().sort("tool", "statistic", "correlation").rows()
    fig, axes = R.grid(len(keys) + 1, ncols=3, panel=(4.0, 3.2))
    methods = R.ordered(c["method"].unique().to_list())
    for ax, (tool, stat, corr) in zip(axes, keys):
        sub = c.filter((pl.col("tool") == tool) & (pl.col("statistic") == stat) & (pl.col("correlation") == corr))
        for m in methods:
            d = sub.filter(method=m)
            if not d.height:
                continue
            g = (d.with_columns(x=pl.col("shell").map_elements(lambda s: 1 / d_mid(s) ** 2, return_dtype=pl.Float64))
                 .group_by("x").agg(pl.col("value").mean(), pl.col("value").min().alias("lo"), pl.col("value").max().alias("hi"))
                 .sort("x"))
            lw = 2.2 if m == "laue_integrate" else 1.4
            ax.plot(g["x"], g["value"], "-o", ms=3, lw=lw, color=R.colour(m), label=R.label(m))
            if (g["hi"] - g["lo"]).max() > 0:
                ax.fill_between(g["x"], g["lo"], g["hi"], color=R.colour(m), alpha=0.12, lw=0)
        ax.set_title(f"{stat}" + (f" ({corr})" if corr else "") + ("" if tool.lower() in stat.lower() else f"  [{tool}]"))
        ax.set_xlabel(r"1/d$^2$ (Å$^{-2}$)")
        dt = [6, 4, 3, 2.5, 2.2, 2.0, 1.8]
        sec = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
        sec.set_xticks([1 / v**2 for v in dt], [f"{v:g}" for v in dt])
        sec.set_xlabel("d (Å)")
    h, lab = axes[0].get_legend_handles_labels()
    axes[len(keys)].legend(h, lab, loc="center")
    axes[len(keys)].set_axis_off()
    fig.suptitle(f"{dataset}: careless.stats by resolution shell (careless seed 1)")
    print(R.save(fig, out / f"careless_stats_{dataset}"))


def fig_refinement(src: Path, careless: pl.DataFrame, out: Path, dataset: str) -> None:
    sh = pl.read_csv(src / "downstream_refine_shells.csv").filter(seed=1)
    cc = pl.read_csv(src / "downstream_ccwork.csv").filter(seed=1)
    methods = R.ordered(sh["method"].unique().to_list())
    sh = sh.with_columns(x=1 / ((pl.col("d_max") + pl.col("d_min")) / 2) ** 2)
    cc = cc.with_columns(x=1 / ((pl.col("d_max") + pl.col("d_min")) / 2) ** 2)
    fig, axes = R.grid(8, ncols=3, panel=(4.0, 3.3))
    for m in methods:
        d = sh.filter(method=m).sort("x")
        kw = dict(color=R.colour(m), lw=2.2 if m == "laue_integrate" else 1.4, ms=3)
        axes[0].plot(d["x"], d["r_work"], "-o", label=R.label(m), **kw)
        axes[1].plot(d["x"], d["r_free"], "-o", **kw)
        axes[2].plot(d["x"], d["r_free"] - d["r_work"], "-o", **kw)
        axes[3].plot(d["x"], d["fom"], "-o", **kw)
        axes[4].plot(d["x"], d["phase_error"], "-o", **kw)
        w, f = cc.filter(method=m, set="work").sort("x"), cc.filter(method=m, set="free").sort("x")
        axes[6].plot(w["x"], w["cc_I"], "-o", **kw)
        axes[7].plot(f["x"], f["cc_I"], "-o", **kw)
        ch = careless.filter((pl.col("method") == m) & (pl.col("tool") == "cchalf") & (pl.col("correlation") == "pearson")
                             & pl.col("shell").str.contains("-"))
        if ch.height:
            g = (ch.with_columns(x=pl.col("shell").map_elements(lambda s: 1 / d_mid(s) ** 2, return_dtype=pl.Float64))
                 .group_by("x").agg(pl.col("value").mean()).sort("x"))
            c12 = g["value"].to_numpy().clip(1e-6, 1)
            cstar = np.sqrt(2 * c12 / (1 + c12))
            axes[5].plot(g["x"], cstar, "-o", **kw)
            for ax in (axes[6], axes[7]):
                ax.plot(g["x"], cstar, "--", color=R.colour(m), lw=0.9, alpha=0.7)
    titles = ["R-work", "R-free", "R-free - R-work", "figure of merit", "phase error (degrees)",
              "CC* from careless CC½ (Pearson)", "CC_work (solid), CC* (dashed)", "CC_free (solid), CC* (dashed)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel(r"1/d$^2$ (Å$^{-2}$)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: refinement by resolution shell (phenix.refine, pinned R-free flags, careless seed 1)")
    print(R.save(fig, out / f"refinement_shells_{dataset}"))


def fig_steps(src: Path, out: Path, dataset: str) -> None:
    st = pl.read_csv(src / "downstream_refine_steps.csv").filter(seed=1)
    methods = R.ordered(st["method"].unique().to_list())
    fig, axes = R.grid(2, ncols=2, panel=(5.5, 3.6))
    for m in methods:
        d = st.filter(method=m).sort("step")
        d = d.filter(pl.col("stage").str.contains("adp|end"))
        kw = dict(color=R.colour(m), lw=2.2 if m == "laue_integrate" else 1.4, ms=4)
        axes[0].plot(range(d.height), d["r_work"], "-o", label=R.label(m), **kw)
        axes[1].plot(range(d.height), d["r_free"], "-o", **kw)
        ticks = d["stage"].to_list()
    for ax, t in zip(axes, ("R-work", "R-free")):
        ax.set_xticks(range(len(ticks)), ticks, rotation=45)
        ax.set_title(f"{t} after each ADP step of refinement")
    axes[0].legend(fontsize=8)
    print(R.save(fig, out / f"refinement_steps_{dataset}"))


def fig_summary(src: Path, out: Path, dataset: str) -> None:
    r = pl.read_csv(src / "downstream_refine.csv").filter(seed=1)
    methods = R.ordered(r["method"].unique().to_list())
    r = r.with_columns(pl.col("method").replace_strict({m: i for i, m in enumerate(methods)}).alias("o")).sort("o")
    fig, axes = R.grid(2, ncols=2, panel=(5.0, 0.45 * len(methods) + 1.0))
    for ax, col in zip(axes, ("r_free", "r_work")):
        y = np.arange(r.height)
        ax.barh(y, r[col], color=[R.colour(m) for m in r["method"]])
        for yi, v in zip(y, r[col]):
            ax.text(v, yi, f" {v:.4f}", va="center")
        ax.set_yticks(y, [R.label(m) for m in r["method"]])
        ax.set_xlim(r[col].min() * 0.95, r[col].max() * 1.04)
        ax.invert_yaxis()
        ax.set_title(col.replace("_", "-").upper().replace("R-", "R-"))
    print(R.save(fig, out / f"refinement_summary_{dataset}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", default=",".join(ALL))
    ap.add_argument("--dataset", default="mos010_pc_default")
    a = ap.parse_args()
    R.setup()
    want = set(a.only.split(","))
    src = a.data / "sim_downstream"
    careless = pl.read_csv(a.data / "sim_careless_stats_mos010_pc.csv", infer_schema_length=None)
    if "careless" in want:
        fig_careless(careless, a.out, a.dataset)
    if "refinement" in want and (src / "downstream_refine_shells.csv").exists():
        fig_refinement(src, careless, a.out, a.dataset)
    if "steps" in want and (src / "downstream_refine_steps.csv").exists():
        fig_steps(src, a.out, a.dataset)
    if "summary" in want:
        fig_summary(src, a.out, a.dataset)


if __name__ == "__main__":
    main()
