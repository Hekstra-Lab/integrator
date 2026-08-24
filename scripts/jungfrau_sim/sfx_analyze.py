"""Post-analysis of the SFX experiment matrix: recovery vs training, and final scatter.

Discovers every run directory under `--runs` (each is one `sfx_experiment.py` config,
named by its tag, e.g. `poisson_known_per_image`), and makes:

  sfx_recovery_curves.png   recovery metrics vs epoch, one line per run.
  sfx_recovery_scatter.png  final recovered-vs-true scatter (intensity, per-image G).

Runs are coloured by likelihood and dashed/solid by profile mode, so any slice of the
matrix (likelihood A/B, known-vs-learned profile, per-image-vs-global G) reads off the
same figure. Point it at a subset by pre-filtering the run dirs, or just let it plot all.

Run:  uv run python scripts/jungfrau_sim/sfx_analyze.py --runs data/sfx_runs --data data/sfx_sim
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sfx_experiment import SFXData, SFXIntegrator

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
# Colour by likelihood (categorical slots), line style by profile mode.
LIK_COLOR = {"poisson": "#2a78d6", "normal_coupled": "#eb6834", "normal_free": "#008300"}
PROFILE_STYLE = {"known": "-", "learned": "--"}


def _lik(tag):        # tag = <likelihood>_<profile>_<scale>[_Bi]
    for k in LIK_COLOR:
        if tag.startswith(k):
            return k
    return "poisson"


def _profile(tag):
    return "learned" if "_learned_" in tag else "known"


def _ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9.5, color=INK, pad=6)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def curves(runs: dict, b_true: float, out: Path) -> None:
    has_prof = any("profile_cos" in h[-1] for h in runs.values())
    panels = [("corr_logG", "per-image scale  G", "corr(log G)", (0, 1)),
              ("corr_logI", "per-obs intensity  I", "corr(log I)", (0, 1)),
              ("B_mean", "Wilson B factor", "B (dashed = truth)", None),
              ("corr_bg", "background", "corr(bg)", (0, 1))]
    if has_prof:
        panels.append(("profile_cos", "learned profile", "cosine vs true", (0, 1)))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.25 * len(panels), 3.1), facecolor=SURFACE)
    for ax, (key, title, ylab, ylim) in zip(axes, panels):
        for tag, hist in runs.items():
            pts = [(h["epoch"], h[key]) for h in hist if key in h]
            if not pts:
                continue
            ep, val = zip(*pts)
            ax.plot(ep, val, PROFILE_STYLE[_profile(tag)], lw=1.8, color=LIK_COLOR[_lik(tag)],
                    marker="o", ms=2.5, label=tag)
        _ax(ax, title, "epoch", ylab)
        if ylim:
            ax.set_ylim(*ylim)
        if key == "B_mean":
            ax.axhline(b_true, ls=":", lw=1.2, color=MUTED)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("SFX ground-truth recovery vs training", fontsize=11, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


@torch.no_grad()
def _predict(run_dir: Path, data_dir: Path):
    ckpts = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    st = torch.load(ckpts[-1], map_location="cpu")
    args = Namespace(**st["args"])
    data = SFXData(data_dir, args.likelihood)
    model = SFXIntegrator(data, args)
    model.load_state_dict(st["model"])
    model.eval()
    i_hat = torch.empty(data.n_obs)
    for s in range(0, data.n_obs, 8192):
        e = slice(s, min(s + 8192, data.n_obs))
        q_i, _, _ = model.encode(data.counts[e])
        i_hat[e] = q_i.mean.squeeze(-1)
    g = torch.exp((model.log_G.weight.squeeze(1) if model.scale_per_image
                   else model.log_G).detach())
    return {"i_hat": i_hat.numpy(), "i_true": data.i_true.numpy(),
            "g_hat": g.numpy(), "g_true": data.g_true.numpy(),
            "per_image": model.scale_per_image}


def scatter(run_dirs: dict, data_dir: Path, out: Path) -> None:
    tags = list(run_dirs)
    fig, axes = plt.subplots(2, len(tags), figsize=(3.7 * len(tags), 7.2),
                             facecolor=SURFACE, squeeze=False)
    for j, tag in enumerate(tags):
        p = _predict(run_dirs[tag], data_dir)
        rows = [(p["i_hat"], p["i_true"], "intensity I (per obs)")]
        rows.append((p["g_hat"], p["g_true"], "scale G (per image)") if p["per_image"]
                    else (None, None, "scale G (global -- n/a)"))
        for i, (hat, true, name) in enumerate(rows):
            ax = axes[i][j]
            if hat is None:
                ax.text(0.5, 0.5, "global G\n(no per-image)", ha="center", va="center",
                        transform=ax.transAxes, color=MUTED, fontsize=9)
                _ax(ax, name, "", "")
                continue
            lo = max(min(true.min(), hat.min()), 1e-2)
            hi = max(true.max(), hat.max())
            ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=MUTED, zorder=1)
            ax.scatter(true, hat, s=5, alpha=0.3, color=LIK_COLOR[_lik(tag)],
                       edgecolors="none", zorder=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _ax(ax, f"{tag}\n{name}" if i == 0 else name, "true", "recovered")
    fig.suptitle("Final recovery: recovered vs ground truth", fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="data/sfx_runs")
    ap.add_argument("--data", default="data/sfx_sim")
    ap.add_argument("--out", default="scripts/jungfrau_sim")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    run_dirs = {p.parent.name: p.parent for p in sorted(runs_root.glob("*/history.json"))}
    if not run_dirs:
        raise SystemExit(f"no runs under {runs_root}/ (expected <tag>/history.json)")

    b_true = json.loads((Path(args.data) / "sim.json").read_text())["b_global"]
    runs = {tag: json.loads((d / "history.json").read_text()) for tag, d in run_dirs.items()}

    print(f"{len(runs)} runs:")
    for tag, h in runs.items():
        m = h[-1]
        prof = f"  prof {m['profile_cos']:.3f}" if "profile_cos" in m else ""
        print(f"  {tag:34} ep{m['epoch']:>4}: corr(logI) {m['corr_logI']:.3f}  "
              f"corr(logG) {m['corr_logG']:.3f}  B {m['B_mean']:.1f}  corr(bg) {m['corr_bg']:.3f}{prof}")

    curves(runs, b_true, Path(args.out) / "sfx_recovery_curves.png")
    scatter(run_dirs, Path(args.data), Path(args.out) / "sfx_recovery_scatter.png")


if __name__ == "__main__":
    main()
