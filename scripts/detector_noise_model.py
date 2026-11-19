"""The detector's mean-variance relation, from shoebox background alone.

A stored pixel value is not a photon count. For a detector writing
`c = c0 + g*N + eps` with `N ~ Poisson` and `eps ~ N(0, sigma_r^2)`,

    Var(c) = g * (mean(c) - c0) + sigma_r^2

so a straight line through (mean, variance) over background pixels has
**slope `g`** -- the stored units per photon -- and intercept
`sigma_r^2 - g*c0`. That slope is the one number the Rao-Blackwellized
export needs and cannot get anywhere else: it sets how fast the variance
grows with signal, and so the weight of a peak pixel against a background
pixel. The individual pieces -- pedestal, read noise, background photon
rate -- are NOT separately identifiable from these data and are not
needed; `Var = g*(I*p) + V_B` uses the slope and the measured background
variance and nothing else.

Why background pixels only. They carry no signal, so nothing here depends
on the profile, on a fitted intensity, or on a model that might not fit.
The held-out residuals cannot do this job: at bright pixels they are
dominated by rate misfit -- chi-square per pixel of 73 to 86 on
hewl_1118 -- which inflates the apparent variance and with it the slope.

Where the lever arm comes from. One shoebox gives one (mean, variance)
pair and a line cannot be fitted to a point. Different reflections sit at
different detector radii and so on different background levels: on
hewl_1118 the modelled background runs from 7.4 to 65 ADU, an eightfold
range, which is enough. An earlier attempt on 600 reflections failed
because they happened to span 10.1 to 11.3.

The background sample is the outer frame of each shoebox, `--ring` pixels
deep, which holds no spot by construction. Reads `counts.npy` and
`masks.npy` by memory map in chunks, so it needs neither a GPU nor the
model nor the whole array in memory.

Usage:
    uv run python scripts/detector_noise_model.py <data-dir> [--out t.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data_dir", type=Path,
                   help="directory holding counts.npy and masks.npy")
    p.add_argument("--out", type=Path, default=None,
                   help="write the per-shoebox pairs here")
    p.add_argument("--ring", type=int, default=3,
                   help="depth of the outer frame used as background")
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--max-refl", type=int, default=0,
                   help="stop after this many reflections (0 = all)")
    p.add_argument("--bins", type=int, default=14)
    return p.parse_args()


def ring_mask(npx: int, depth: int) -> np.ndarray:
    """The outer `depth` rows and columns of a square shoebox."""
    side = int(round(npx ** 0.5))
    if side * side != npx:
        raise SystemExit(f"{npx} pixels is not a square shoebox")
    img = np.arange(npx).reshape(side, side)
    sel = np.zeros(npx, bool)
    for k in range(depth):
        sel[img[k]] = sel[img[-1 - k]] = True
        sel[img[:, k]] = sel[img[:, -1 - k]] = True
    return sel


def pairs(data_dir: Path, ring: int, chunk: int, max_refl: int):
    """(mean, variance, n) over the background ring of every shoebox."""
    counts = np.load(data_dir / "counts.npy", mmap_mode="r")
    masks = np.load(data_dir / "masks.npy", mmap_mode="r")
    n_refl, npx = counts.shape[0], int(np.prod(counts.shape[1:]))
    if max_refl:
        n_refl = min(n_refl, max_refl)
    sel = ring_mask(npx, ring)
    print(f"  {n_refl:,} reflections, {npx} pixels, "
          f"{int(sel.sum())} in the outer {ring} rings")

    mu, va, nn = [], [], []
    for start in range(0, n_refl, chunk):
        stop = min(start + chunk, n_refl)
        c = np.asarray(counts[start:stop]).reshape(stop - start, npx)
        m = np.asarray(masks[start:stop]).reshape(stop - start, npx) > 0
        use = m & sel
        n = use.sum(1)
        ok = n >= 20
        if not ok.any():
            continue
        c = c[ok].astype(np.float64)
        use = use[ok]
        n = n[ok].astype(np.float64)
        s1 = (c * use).sum(1)
        s2 = ((c ** 2) * use).sum(1)
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - 1.0)
        mu.append(mean)
        va.append(var)
        nn.append(n)
    return (np.concatenate(mu), np.concatenate(va), np.concatenate(nn))


def fit(mu, va, nn, n_bins: int):
    """Slope and intercept of variance against mean, on binned medians.

    Binned and median rather than a straight regression over shoeboxes:
    one shoebox's variance over ~150 pixels has a relative error of about
    12%, and a handful of shoeboxes whose ring clips a neighbouring spot
    have no business setting the slope. The bins are log-spaced in the
    mean, because the background level is spread over nearly a decade and
    quantile bins would put almost all of them together.
    """
    good = np.isfinite(mu) & np.isfinite(va) & (mu > 0)
    mu, va, nn = mu[good], va[good], nn[good]
    lo, hi = float(np.percentile(mu, 0.5)), float(np.percentile(mu, 99.5))
    edges = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), n_bins + 1)
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(mu, edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        s = idx == k
        if s.sum() < 30:
            continue
        rows.append({"bin": k, "mean": float(np.median(mu[s])),
                     "var": float(np.median(va[s])), "n": int(s.sum())})
    if len(rows) < 3:
        raise SystemExit("not enough populated bins to fit a line")
    x = np.array([r["mean"] for r in rows])
    y = np.array([r["var"] for r in rows])
    w = np.sqrt([r["n"] for r in rows])
    g, c = np.polyfit(x, y, 1, w=w)
    return float(g), float(c), rows


def main() -> int:
    a = parse_args()
    mu, va, nn = pairs(a.data_dir, a.ring, a.chunk, a.max_refl)
    g, c, rows = fit(mu, va, nn, a.bins)

    print(f"\n  {'background mean':>16}{'variance':>11}{'var/mean':>10}"
          f"{'shoeboxes':>11}")
    for r in rows:
        print(f"  {r['mean']:>16.3f}{r['var']:>11.3f}"
              f"{r['var'] / max(r['mean'], 1e-9):>10.3f}{r['n']:>11,d}")

    print(f"\n  Var = {g:.4f} * mean + {c:.3f}")
    print(f"    gain            {g:.4f} stored units per photon")
    if g > 1e-6:
        print(f"    pedestal        {-c / g:.3f}  (if the read noise is "
              f"negligible)")
    print(f"    Poisson would be 1.0000 * mean + 0")
    print(f"\n  background mean spans {np.percentile(mu, 1):.2f} to "
          f"{np.percentile(mu, 99):.2f} over {mu.size:,} shoeboxes")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["bin", "mean", "var", "n"])
            w.writeheader()
            w.writerows(rows)
        print(f"  {a.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
