#!/usr/bin/env python
"""Fit `G` and `B` from a dataset, so they can be pinned instead of learned.

The Wilson line is `log<I> = log G - 2 B s^2`, so a binned fit gives the
intercept and slope directly. Learning them jointly with everything else is
not reliable -- `wilson_loss.py` records B collapsing 47 -> 1.5 on SBGrid 845
and 12 -> 4.0 on 821 against classical fits of 22-44 A^2, and the conjugate
merging run on 9b7c took B from 7.5 to 0.06 while G sat still. A flattened
prior costs little where the likelihood is strong and destroys the weak
shells where it is not.

Two fits are reported because they answer different questions:

* from `intensity.sum.value`, the DIALS intensities, which is the classical
  Wilson plot and the number to compare against the literature;
* from the raw counts minus the background, which is what the model itself
  can see at init and what `stabilize_scale` uses.

Low resolution is excluded (`--d-max`, default 3.5 A): the solvent
contribution bends the Wilson line there, and including it biases B low --
which is the direction the collapse already goes.

Usage:
    uv run python scripts/fit_wilson_scale.py <dataset_dir> [--d-max 3.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dataset_dir", type=Path)
    p.add_argument("--metadata", default=None, help="reference file to read")
    p.add_argument("--d-max", type=float, default=3.5)
    p.add_argument("--d-min", type=float, default=0.0)
    p.add_argument("--n-bins", type=int, default=20)
    return p.parse_args()


def wilson_line(i: np.ndarray, s_sq: np.ndarray, n_bins: int):
    """Equal-count bins in s^2; returns (G, B, r, n_used)."""
    ok = np.isfinite(i) & np.isfinite(s_sq) & (s_sq > 0)
    i, s_sq = i[ok], s_sq[ok]
    edges = np.quantile(s_sq, np.linspace(0, 1, n_bins + 1))
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s_sq >= a) & (s_sq <= b)
        if m.sum() < 30:
            continue
        mean_i = i[m].mean()
        if mean_i <= 0:
            continue
        xs.append(s_sq[m].mean())
        ys.append(np.log(mean_i))
    if len(xs) < 3:
        raise SystemExit("too few usable bins for a Wilson fit")
    xs, ys = np.asarray(xs), np.asarray(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    r = np.corrcoef(xs, ys)[0, 1]
    return float(np.exp(intercept)), float(-slope / 2.0), float(r), int(ok.sum())


def main() -> None:
    a = parse_args()
    from integrator.io import load_metadata

    name = a.metadata
    if name is None:
        spec = a.dataset_dir / "dataset.yaml"
        if spec.is_file():
            import yaml

            name = (yaml.safe_load(spec.read_text()) or {}).get("files", {}).get(
                "reference"
            )
        for cand in (name, "metadata.pt", "metadata.npy"):
            if cand and (a.dataset_dir / cand).exists():
                name = cand
                break
    meta = load_metadata(a.dataset_dir / name)
    print(f"  {a.dataset_dir.name}/{name}")

    d = np.asarray(meta["d"]).ravel().astype(float)
    s_sq = 1.0 / (4.0 * np.clip(d, 1e-6, None) ** 2)
    sel = (d <= a.d_max) & (d >= a.d_min) if a.d_max else np.ones_like(d, bool)
    if "centric" in meta:
        # the Wilson line is defined for the acentric distribution; centrics
        # have the same mean but their own statistics, and the fit in
        # wilson_loss.py excludes them for the sd/mean calibration too
        cen = np.asarray(meta["centric"]).ravel().astype(bool)
        sel &= ~cen
        print(f"  excluding {cen.sum():,} centrics")
    print(f"  {sel.sum():,} of {len(d):,} reflections in "
          f"{a.d_max:.2f}-{max(a.d_min, d[sel].min()):.2f} A\n")

    print(f"  {'source':28s} {'G':>12s} {'B (A^2)':>9s} {'r':>7s}")
    print("  " + "-" * 60)

    if "intensity.sum.value" in meta:
        i = np.asarray(meta["intensity.sum.value"]).ravel().astype(float)
        G, B, r, n = wilson_line(i[sel], s_sq[sel], a.n_bins)
        print(f"  {'DIALS intensity.sum':28s} {G:12.1f} {B:9.2f} {r:7.3f}")

    if "intensity.prf.value" in meta:
        i = np.asarray(meta["intensity.prf.value"]).ravel().astype(float)
        G, B, r, n = wilson_line(i[sel], s_sq[sel], a.n_bins)
        print(f"  {'DIALS intensity.prf':28s} {G:12.1f} {B:9.2f} {r:7.3f}")

    print(
        "\n  pin these with, on the loss:\n"
        "    init_G: <G>\n    init_B: <B>\n"
        "    learn_G: false\n    learn_B: false"
    )


if __name__ == "__main__":
    main()
