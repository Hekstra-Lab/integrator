#!/usr/bin/env python
"""Reconstruct per-observation intensities from a merging model and check them.

A merging model emits a merged `I_h` per unique reflection and a
per-observation scale `s_i`, so the quantity comparable to an integrator's
output -- and to DIALS -- is

    I_i = s_i * E[J_i]

`E[J_i]` alone lives on a de-scaled gauge and is not comparable to anything.
The point of the comparison is the RESOLUTION DEPENDENCE: the scale may only
carry factors that vary between symmetry mates, so anything that is a
function of `d` belongs in `I_h`. If it has instead been absorbed into
`s(d)`, the merged intensities carry the prior's falloff rather than the
data's, and no downstream scaling can put it back.

Three views, in the order they localize the problem:

1. the overall scale -- medians and the log-log slope against DIALS;
2. the same, SHELL BY SHELL, which is where a gauge error shows;
3. the profile, so a profile defect is not mistaken for a scale defect.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pred_dir", type=Path, help="directory of prediction parquets")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--shells", type=int, default=8)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load(pred_dir: Path) -> pl.DataFrame:
    files = sorted(glob.glob(f"{pred_dir}/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no parquets under {pred_dir}")
    return pl.read_parquet(files)


def col(df: pl.DataFrame, *names: str) -> np.ndarray | None:
    for n in names:
        if n in df.columns:
            return df[n].to_numpy().astype(float)
    return None


def symmetric_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Bracketed log-log slope; OLS alone is attenuated by noise in x."""
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    return float(np.sqrt(np.polyfit(lx, ly, 1)[0] / np.polyfit(ly, lx, 1)[0]))


def main() -> None:
    a = parse_args()
    df = load(a.pred_dir)
    print(f"  {len(df):,} observations, columns: {sorted(df.columns)}\n")

    d = col(df, "d")
    # whichever DIALS column the run's predict_keys carried: a merging config
    # may export the profile-fitted intensity rather than the summed one
    dials = col(df, "intensity.sum.value", "intensity.prf.value")
    scale = col(df, "scale")
    ej = col(df, "qi_mean")
    ih = col(df, "I_h_mean")
    if d is None or dials is None:
        raise SystemExit(
            "need `d` and one of intensity.sum.value / intensity.prf.value "
            f"to compare; got {sorted(df.columns)}"
        )

    cands: dict[str, np.ndarray] = {}
    if ej is not None:
        cands["E[J] (de-scaled)"] = ej
        if scale is not None:
            cands["scale * E[J]"] = scale * ej
    if ih is not None:
        cands["E[I_h] (merged)"] = ih
        if scale is not None:
            cands["scale * E[I_h]"] = scale * ih
    cands["DIALS"] = dials

    print(f"{'quantity':20s} {'median':>12s} {'ratio/DIALS':>12s} {'slope':>7s}")
    print("-" * 56)
    for name, v in cands.items():
        sl = symmetric_slope(dials, v) if name != "DIALS" else 1.0
        print(f"{name:20s} {np.median(v):12.3f} "
              f"{np.median(v) / max(np.median(dials), 1e-12):12.4f} {sl:7.3f}")

    # --- where a gauge error shows: the trend across resolution
    print(f"\nby resolution shell (equal-volume in 1/d^2):")
    s2 = 1.0 / np.clip(d, 1e-6, None) ** 2
    edges = np.quantile(s2, np.linspace(0, 1, a.shells + 1))
    header = f"{'d range (A)':>16s} {'n':>8s}"
    keys = [k for k in cands if k != "DIALS"]
    header += "".join(f"{k.split(' ')[0][:9]:>11s}" for k in keys)
    print(header + f"{'DIALS':>11s}")
    for i in range(a.shells):
        m = (s2 >= edges[i]) & (s2 <= edges[i + 1])
        if m.sum() < 20:
            continue
        lo, hi = 1 / np.sqrt(edges[i + 1]), 1 / np.sqrt(max(edges[i], 1e-12))
        row = f"{hi:7.2f}-{lo:6.2f} {m.sum():8,d}"
        for k in keys:
            row += f"{np.median(cands[k][m]):11.2f}"
        row += f"{np.median(dials[m]):11.2f}"
        print(row)

    # the diagnostic number: does the ratio to DIALS drift with resolution?
    print("\nratio to DIALS across shells (flat = the gauge is right):")
    for k in keys:
        r = []
        for i in range(a.shells):
            m = (s2 >= edges[i]) & (s2 <= edges[i + 1])
            if m.sum() < 20:
                continue
            r.append(np.median(cands[k][m]) / max(np.median(dials[m]), 1e-12))
        r = np.array(r)
        drift = r.max() / max(r.min(), 1e-12)
        print(f"  {k:20s} {np.array2string(r, precision=3, max_line_width=100)}"
              f"   max/min = {drift:.2f}")

    if a.out_dir is None:
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k in keys:
        r = []
        mid = []
        for i in range(a.shells):
            m = (s2 >= edges[i]) & (s2 <= edges[i + 1])
            if m.sum() < 20:
                continue
            r.append(np.median(cands[k][m]) / max(np.median(dials[m]), 1e-12))
            mid.append(np.median(d[m]))
        axes[0].plot(mid, r, marker="o", ms=4, label=k)
    axes[0].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("d (A)")
    axes[0].set_ylabel("median ratio to DIALS")
    axes[0].set_title("a flat line means the resolution dependence is in I_h",
                      fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(alpha=0.25, lw=0.5)

    best = "scale * E[J]" if "scale * E[J]" in cands else keys[0]
    ok = (dials > 0) & (cands[best] > 0)
    axes[1].hexbin(np.log10(dials[ok]), np.log10(cands[best][ok]),
                   gridsize=90, bins="log", cmap="viridis")
    lim = [np.log10(dials[ok]).min(), np.log10(dials[ok]).max()]
    axes[1].plot(lim, lim, "w--", lw=1)
    axes[1].set_xlabel("log10 DIALS I")
    axes[1].set_ylabel(f"log10 {best}")
    axes[1].set_title(f"slope = {symmetric_slope(dials, cands[best]):.3f}",
                      fontsize=9)
    fig.tight_layout()
    out = a.out_dir / "fig_merging_reconstruction.png"
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
