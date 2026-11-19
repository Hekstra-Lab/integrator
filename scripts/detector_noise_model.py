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

There is NO lever arm, and that is the finding. One shoebox gives one
(mean, variance) pair and a line cannot be fitted to a point. Different
reflections sit at different detector radii, so their background levels
might have differed -- but measured over 2.66M shoeboxes the background
ring spans only 9.76 to 11.72 ADU, no wider than the 600-reflection
sample that was dismissed as too narrow. (The 7.4 to 65 range this file
once cited was `qbg_mean`, the model's FITTED background, which is
inflated wherever a shoebox holds signal or a neighbour. It is not the
background level.)

So the slope here is a line through a 20% span and the gain is NOT
identifiable from it: 0.140 through the origin and 0.188 with an
intercept both fit, and the residuals are 20 to 30 times the statistical
floor, so neither model is adequate. Read this fit as a consistency
check, not a measurement. The number the export needs comes from
`autocorr` and `block_ratio` below, which do not need a lever arm.

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


def autocorr(data_dir: Path, chunk: int, max_refl: int, lag: int = 4,
             weak_frac: float = 0.4):
    """The 2D autocorrelation of background pixels, lag by lag.

    `sum over lags of rho` is `1 / sum(k^2)`, the factor by which the
    single-pixel variance understates the per-pixel variance of a SUM.
    Measuring the whole function rather than the sum also says what the
    kernel is: support out to lag 1 only means a two-tap kernel, and a
    two-tap kernel with `rho(1) = 0.25` is the exact signature of bilinear
    resampling, for which `sum(k^2) = 4/9` in 2D and the sum of rho is
    2.25.

    Mean-centring each shoebox removes the zero-frequency component and
    biases the sum LOW -- on simulated independent pixels it returns 0.87
    rather than 1.00, and on a 3-tap kernel 5.90 rather than 7.11. The
    bias is reported alongside, from an independent-pixel control built
    from the same shoeboxes by shuffling each one's background pixels,
    which destroys the correlation and keeps everything else.

    Only the weakest `weak_frac` of shoeboxes are used, where the whole
    box is background, so no spot enters any lag.
    """
    counts = np.load(data_dir / "counts.npy", mmap_mode="r")
    masks = np.load(data_dir / "masks.npy", mmap_mode="r")
    n_refl, npx = counts.shape[0], int(np.prod(counts.shape[1:]))
    if max_refl:
        n_refl = min(n_refl, max_refl)
    side = int(round(npx ** 0.5))
    rng = np.random.default_rng(0)
    acc = {}
    shuf = {}
    kept = 0
    for start in range(0, n_refl, chunk):
        stop = min(start + chunk, n_refl)
        c = np.asarray(counts[start:stop]).reshape(-1, side, side).astype(
            np.float64)
        m = np.asarray(masks[start:stop]).reshape(-1, side, side) > 0
        flat = c.reshape(len(c), -1)
        med = np.median(flat, axis=1, keepdims=True)
        excess = np.maximum(flat - med, 0).sum(1)
        weak = excess <= np.quantile(excess, weak_frac)
        weak &= m.reshape(len(m), -1).sum(1) > 0.9 * npx
        if not weak.any():
            continue
        c, m = c[weak], m[weak]
        kept += len(c)
        for tag, arr in (("data", c), ("shuffled", None)):
            if arr is None:
                flat2 = c.reshape(len(c), -1).copy()
                for row in flat2:
                    rng.shuffle(row)
                arr = flat2.reshape(-1, side, side)
            mu = arr.mean((1, 2), keepdims=True)
            r = arr - mu
            c0 = (r ** 2).mean((1, 2))
            store = acc if tag == "data" else shuf
            for dy in range(-lag, lag + 1):
                for dx in range(-lag, lag + 1):
                    a = r[:, max(0, dy):side + min(0, dy),
                          max(0, dx):side + min(0, dx)]
                    b2 = r[:, max(0, -dy):side + min(0, -dy),
                           max(0, -dx):side + min(0, -dx)]
                    val = (a * b2).mean((1, 2)) / np.maximum(c0, 1e-12)
                    store.setdefault((dy, dx), []).append(val)
    rho = {k: float(np.median(np.concatenate(v))) for k, v in acc.items()}
    ctrl = {k: float(np.median(np.concatenate(v))) for k, v in shuf.items()}
    return rho, ctrl, kept


def block_ratio(data_dir: Path, chunk: int, max_refl: int,
                sizes=(1, 2, 3, 6)):
    """Variance of b x b block sums per pixel, over background strips.

    The single-pixel variance is NOT the per-pixel variance to use for a
    weighted SUM when neighbouring pixels are correlated. Writing the
    stored field as `c = c0 + G*(k conv N) + eps` with `sum k = 1`:

        Var(c_i)       = G^2 * lam * sum(k^2) + sigma_r^2
        Var(S_b) / b^2 -> G^2 * lam + sigma_r^2        as b outgrows k

    because a sum over a region wider than the kernel conserves photons.
    Their ratio rises from 1 to `sum_lags rho = 1 / sum(k^2)`, and the
    pedestal cancels because it lives in the mean and not the variance.
    So this separates a detector gain (ratio stays 1, pixels independent)
    from a point spread (ratio rises), which the mean-variance slope
    alone cannot do, and the large-block value is the quantity the Fisher
    information needs.

    Measured on hewl_1118: 1.000, 1.634, 1.910, 2.137 at b = 1, 2, 3, 6,
    saturating near 2.14 against a lag-1 correlation of 0.25. The pixels
    are correlated, `sum(k^2)` is about 0.47, and the per-pixel variance
    for summing is 2.1x the marginal one.

    The strips are the top and bottom six rows, full width, which clear
    the central spot.
    """
    counts = np.load(data_dir / "counts.npy", mmap_mode="r")
    masks = np.load(data_dir / "masks.npy", mmap_mode="r")
    n_refl, npx = counts.shape[0], int(np.prod(counts.shape[1:]))
    if max_refl:
        n_refl = min(n_refl, max_refl)
    side = int(round(npx ** 0.5))
    rows = np.r_[0:6, side - 6:side]
    acc = {b: [] for b in sizes}
    means = []
    for start in range(0, n_refl, chunk):
        stop = min(start + chunk, n_refl)
        c = np.asarray(counts[start:stop]).reshape(-1, side, side)
        m = np.asarray(masks[start:stop]).reshape(-1, side, side) > 0
        c, m = c[:, rows].astype(np.float64), m[:, rows]
        means.append((c * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1))
        for b in sizes:
            nr, nc = 6 // b, side // b
            vals, good = [], []
            for half in (0, 6):
                for i in range(nr):
                    for j in range(nc):
                        sl = (slice(half + i * b, half + i * b + b),
                              slice(j * b, j * b + b))
                        vals.append(c[:, sl[0], sl[1]].sum((1, 2)))
                        good.append(m[:, sl[0], sl[1]].all((1, 2)))
            v = np.stack(vals, 1)
            g = np.stack(good, 1)
            n = g.sum(1)
            ok = n >= 8
            mu = (v * g).sum(1) / np.maximum(n, 1)
            var = (((v - mu[:, None]) ** 2) * g).sum(1) / np.maximum(n - 1, 1)
            acc[b].append(np.where(ok, var / (b * b), np.nan))
    out = {}
    for b in sizes:
        a = np.concatenate(acc[b])
        out[b] = float(np.nanmedian(a))
    return out, float(np.median(np.concatenate(means)))


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

    blocks, mean_px = block_ratio(a.data_dir, a.chunk, a.max_refl)
    base = blocks[1]
    print(f"\n  {'block':>7}{'Var(S)/b^2':>13}{'ratio to 1x1':>14}")
    for b, v in blocks.items():
        print(f"  {b:>3}x{b:<3}{v:>13.4f}{v / base:>14.3f}")
    big = blocks[max(blocks)]
    print(f"\n  sum_lags rho       {big / base:.3f}   "
          f"(1.0 = independent pixels)")
    print(f"  sum(k^2)           {base / big:.3f}")
    print(f"  variance for a SUM {big / mean_px:.4f} * rate   "
          f"<- this is what the Fisher needs")
    print(f"  variance per PIXEL {base / mean_px:.4f} * rate   "
          f"<- the marginal one, NOT for a sum")

    rho, ctrl, kept = autocorr(a.data_dir, a.chunk, a.max_refl)
    s_data = sum(rho.values())
    s_ctrl = sum(ctrl.values())
    print(f"\n  autocorrelation over {kept:,} weak shoeboxes")
    print(f"  {'lag':>8}{'rho':>9}{'shuffled control':>19}")
    for d in range(0, 4):
        for key in ((0, d), (d, d)):
            if key in rho and (d or key == (0, 0)):
                print(f"  {str(key):>8}{rho[key]:>9.4f}{ctrl[key]:>19.4f}")
    print(f"\n  sum rho            {s_data:.3f}  (control {s_ctrl:.3f}, "
          f"1.000 if unbiased)")
    corrected = s_data / max(s_ctrl, 1e-6)
    print(f"  bias-corrected     {corrected:.3f}  -> sum(k^2) = "
          f"{1 / max(corrected, 1e-6):.3f}")
    print(f"\n  noise_a for a SUM  {base * corrected / mean_px:.4f}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["bin", "mean", "var", "n"])
            w.writeheader()
            w.writerows(rows)
        print(f"  {a.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
