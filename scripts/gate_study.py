"""How should the per-pixel signal gate be built from the profile?

A diagnosis on simulated shoeboxes, where the truth is known: streaky
Gaussians of random orientation and elongation, a log-uniform intensity, a
flat background, and `c_i ~ Poisson(I p_i + B)`.

The algebra first, because it settles the framing. For `c_i = s_i + b_i`
with `s_i ~ Poisson(I p_i)` and `b_i ~ Poisson(B)`, the posterior
probability that a count at pixel `i` came from signal is

    r_i = I p_i / (I p_i + B) = sigmoid( log(I p_i) - log B )

so "put the profile through a sigmoid" and "compute the responsibility"
are the SAME operation -- the sigmoid of the log-odds. There is no third
thing to invent. What the learned head does is apply a sigmoid to a
feature that is not the log-odds, and the question this script answers is
what that costs.

With the exact `r`, `E[sum_i r_i c_i] = sum_i I p_i = I` identically:
unbiased with nothing fitted. Every candidate below is scored against
that, on the estimator it implies, `I_hat = sum_i r_i c_i`.

The candidates:

- `exact`          r from the TRUE I, B and p. The reference; its bias is
                   zero by construction and its spread is the floor.
- `plugin`         r from estimates: B from the box median (robust, a Laue
                   shoebox is mostly background), I from profile-weighted
                   least squares given that B. This is `gate: responsibility`.
- `tempered`       sigmoid(a * log-odds) with a != 1. Asks whether the
                   exact slope is the right one or merely a starting point.
- `sampled`        r averaged over profile draws rather than evaluated at
                   the mean profile, which is the "sample profiles and put
                   them through a sigmoid" option. Dirichlet(kappa * p).
- `untied`         a sigmoid of a smooth bump that is NOT the profile:
                   miscentred and mis-shaped by a controlled amount. This
                   is what a free per-pixel head can do, since nothing ties
                   it to p.
- `topk`           a hard mask over the brightest pixels; no profile at all.

Priors enter only through the intensity posterior, and both forms asked
for are reported: a frozen Wilson Gamma prior and a flat one.

    uv run python scripts/gate_study.py
    uv run python scripts/gate_study.py --n 4000 --out docs/figures/data/gate_study.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

H = W = 21
RNG = np.random.default_rng(0)


def streaky_profiles(n: int, rng) -> np.ndarray:
    """`n` unit-sum profiles: anisotropic Gaussians at random angles.

    Elongation and orientation are drawn per reflection, because a profile
    model that only ever sees round spots is not being asked the question.
    """
    y, x = np.mgrid[0:H, 0:W]
    y = y - (H - 1) / 2.0
    x = x - (W - 1) / 2.0
    out = np.empty((n, H * W))
    for k in range(n):
        theta = rng.uniform(0, np.pi)
        s_major = rng.uniform(1.6, 4.5)
        s_minor = s_major / rng.uniform(1.5, 4.0)  # streaky, never round
        cy, cx = rng.normal(0, 0.6, size=2)
        ct, st = np.cos(theta), np.sin(theta)
        u = (x - cx) * ct + (y - cy) * st
        v = -(x - cx) * st + (y - cy) * ct
        g = np.exp(-0.5 * ((u / s_major) ** 2 + (v / s_minor) ** 2))
        out[k] = (g / g.sum()).ravel()
    return out


def simulate(n: int, rng):
    """Profiles, intensities, backgrounds and the counts they produce."""
    p = streaky_profiles(n, rng)
    intensity = 10.0 ** rng.uniform(1.0, 4.0, size=n)      # weak to bright
    background = 10.0 ** rng.uniform(-0.5, 1.5, size=n)    # per pixel
    rate = intensity[:, None] * p + background[:, None]
    counts = rng.poisson(rate).astype(float)
    return p, intensity, background, counts


def plugin_estimates(p: np.ndarray, counts: np.ndarray):
    """B from the box median, then I from profile-weighted least squares."""
    b0 = np.median(counts, axis=1, keepdims=True)
    s2 = np.maximum((p ** 2).sum(axis=1, keepdims=True), 1e-12)
    i0 = np.maximum(((p * (counts - b0)).sum(axis=1, keepdims=True)) / s2, 0.0)
    return i0, b0


def gate_exact(p, intensity, background, _counts):
    lam = intensity[:, None] * p
    return lam / (lam + background[:, None])


def gate_plugin(p, _intensity, _background, counts):
    i0, b0 = plugin_estimates(p, counts)
    lam = i0 * p
    return lam / (lam + np.maximum(b0, 1e-6))


def gate_tempered(p, _intensity, _background, counts, a=0.7):
    """sigmoid(a * log-odds): the exact form with the slope moved off 1."""
    i0, b0 = plugin_estimates(p, counts)
    odds = np.log(np.maximum(i0 * p, 1e-30)) - np.log(np.maximum(b0, 1e-30))
    return 1.0 / (1.0 + np.exp(-a * odds))


def gate_sampled(p, _intensity, _background, counts, kappa=200.0,
                 draws=16, rng=None):
    """r averaged over profile draws, not evaluated at the mean profile.

    The responsibility is concave in p where the gate is near one and convex
    where it is near zero, so averaging the gate over profile uncertainty is
    not the same as the gate at the average profile. This measures whether
    that difference matters.
    """
    rng = rng or np.random.default_rng(1)
    i0, b0 = plugin_estimates(p, counts)
    acc = np.zeros_like(p)
    for _ in range(draws):
        ps = rng.gamma(np.maximum(kappa * p, 1e-6))
        ps = ps / np.maximum(ps.sum(axis=1, keepdims=True), 1e-30)
        lam = i0 * ps
        acc += lam / (lam + np.maximum(b0, 1e-6))
    return acc / draws


def gate_untied(p, _intensity, _background, counts, shift=2.0,
                widen=1.6):
    """A sigmoid of a bump that is NOT the profile.

    What a free per-pixel head is able to do: it has no term tying it to p,
    so it can sit off-centre and be the wrong width. The miscentring and
    widening here are fixed rather than fitted, so the number is a clean
    read on how much the tie is worth.
    """
    y, x = np.mgrid[0:H, 0:W]
    y = (y - (H - 1) / 2.0 - shift).ravel()
    x = (x - (W - 1) / 2.0).ravel()
    i0, b0 = plugin_estimates(p, counts)
    # a round bump of the profile's typical scale, displaced
    scale = widen * np.sqrt((p * (x ** 2 + y ** 2)).sum(axis=1,
                                                        keepdims=True)) + 1e-9
    bump = np.exp(-0.5 * ((x ** 2 + y ** 2)[None, :] / scale ** 2))
    bump = bump / np.maximum(bump.sum(axis=1, keepdims=True), 1e-30)
    lam = i0 * bump
    return lam / (lam + np.maximum(b0, 1e-6))


def gate_topk(_p, _intensity, _background, counts, k=25):
    """A hard mask over the brightest pixels: no profile, no sigmoid."""
    idx = np.argsort(-counts, axis=1)[:, :k]
    g = np.zeros_like(counts)
    np.put_along_axis(g, idx, 1.0, axis=1)
    return g


def gate_split(p, _intensity, _background, counts, rng=None):
    """Estimate the gate on half the pixels, apply it to the other half.

    `plugin` estimates `i0` and `b0` from the same counts it then weights,
    so a pixel that fluctuated high raises its own gate and is counted
    twice over. Splitting the pixels breaks that correlation at the cost of
    half the data, which is the clean way to see how much of `plugin`'s
    bias is the feedback and how much is the plug-in error itself.
    """
    rng = rng or np.random.default_rng(2)
    n, npix = counts.shape
    half = rng.random((n, npix)) < 0.5
    other = ~half
    b0 = np.array([np.median(counts[k, half[k]]) if half[k].any() else 0.0
                   for k in range(n)])[:, None]
    pa = p * half
    s2 = np.maximum((pa ** 2).sum(axis=1, keepdims=True), 1e-12)
    i0 = np.maximum((pa * (counts - b0)).sum(axis=1, keepdims=True) / s2, 0.0)
    lam = i0 * p
    r = lam / (lam + np.maximum(b0, 1e-6))
    # applied to the held-out half only, rescaled for the mass dropped
    keep = other * r
    weight = np.maximum((p * other).sum(axis=1, keepdims=True), 1e-9)
    return keep / weight * (p.sum(axis=1, keepdims=True))


def estimator_ls(p, counts):
    """The profile-weighted least-squares intensity: no gate sum at all.

    `sum_i r_i c_i` is one way to turn a gate into an intensity, and it is
    the one the integrator uses. It is not the only one, and where the gate
    is estimated from the same counts it is the worst one, so the direct
    least-squares fit is carried as the comparison.
    """
    i0, _b0 = plugin_estimates(p, counts)
    return i0.ravel()


def gate_plugin_noclamp(p, _intensity, _background, counts):
    """`plugin`, with the non-negativity clamp on `i0` removed.

    `i0 = max(., 0)` is the obvious thing to do -- an intensity is not
    negative -- but it is a one-sided operation applied to a noisy
    quantity, so for a reflection whose true intensity is near zero it can
    only move the estimate up. The gate is monotone in `i0`, so the bias
    carries straight through to `sum r c`. This isolates how much of
    `plugin`'s weak-end bias is the clamp.
    """
    b0 = np.median(counts, axis=1, keepdims=True)
    s2 = np.maximum((p ** 2).sum(axis=1, keepdims=True), 1e-12)
    i0 = (p * (counts - b0)).sum(axis=1, keepdims=True) / s2
    lam = i0 * p
    return np.clip(lam / (lam + np.maximum(b0, 1e-6)), 0.0, 1.0)


def gate_plugin_lowmean(p, _intensity, _background, counts):
    """`plugin`, with the background from the mean of the lower half.

    The median of a Poisson is exactly 0 whenever the rate is below about
    0.7, and `r = i0 p / (i0 p + b0)` with `b0 = 0` saturates to 1 wherever
    the profile is non-zero, so the intensity collapses to the box total.
    The integrator's own comment records this for shoeboxes that are mostly
    MASKED; it happens for the same reason on any shoebox whose background
    is faint, with nothing masked at all.

    The mean of the pixels at or below the median is still robust to the
    spot -- it only uses the dim half -- but it is an average rather than
    an order statistic, so it does not quantise to zero.
    """
    lo = np.median(counts, axis=1, keepdims=True)
    dim = counts <= lo
    b0 = (counts * dim).sum(axis=1, keepdims=True) / np.maximum(
        dim.sum(axis=1, keepdims=True), 1)
    b0 = np.maximum(b0, 1e-3)
    s2 = np.maximum((p ** 2).sum(axis=1, keepdims=True), 1e-12)
    i0 = np.maximum((p * (counts - b0)).sum(axis=1, keepdims=True) / s2, 0.0)
    lam = i0 * p
    return lam / (lam + b0)


def joint_fit(p, counts):
    """Least squares for (I, B) together against the design `[p, 1]`.

    Every background estimator above is an order statistic of the counts,
    and each is wrong in its own direction: the median quantises to zero on
    a faint background, and the mean of the dim half is below the mean of
    the whole by construction. Neither is a background estimator so much as
    a guess that happens to be robust to the spot.

    `c_i ~ I p_i + B` is linear in both, and `p` is known, so the fit is a
    2x2 normal equation solved per shoebox. Nothing is clamped or
    quantised.
    """
    n, npix = counts.shape
    spp = (p * p).sum(axis=1)
    sp = p.sum(axis=1)
    spc = (p * counts).sum(axis=1)
    sc = counts.sum(axis=1)
    det = spp * npix - sp * sp
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    i0 = (npix * spc - sp * sc) / det
    b0 = (spp * sc - sp * spc) / det
    return i0[:, None], b0[:, None]


def gate_joint(p, _intensity, _background, counts):
    """The responsibility built on the joint (I, B) fit."""
    i0, b0 = joint_fit(p, counts)
    lam = np.maximum(i0, 0.0) * p
    return lam / (lam + np.maximum(b0, 1e-6))


def gate_joint_iter(p, _intensity, _background, counts, steps=3):
    """`joint`, iterated: refit (I, B) with the gate's own weights.

    `joint` is one least-squares fit, which weights every pixel equally.
    The counts are Poisson, so their variance rises with the rate, and the
    pixels under the spot are both the most informative about I and the
    noisiest. Re-fitting with weights `1 / (I p + B)` from the current
    estimate is the Poisson IRLS step, and repeating it is coordinate
    ascent on the same objective the gate comes from.

    This is the obvious thing to try for the residual low-SNR bias, so it
    is measured rather than assumed to help.
    """
    i0, b0 = joint_fit(p, counts)
    for _ in range(steps):
        rate = np.maximum(i0 * p + b0, 1e-6)
        w = 1.0 / rate
        swpp = (w * p * p).sum(axis=1, keepdims=True)
        swp = (w * p).sum(axis=1, keepdims=True)
        sw = w.sum(axis=1, keepdims=True)
        swpc = (w * p * counts).sum(axis=1, keepdims=True)
        swc = (w * counts).sum(axis=1, keepdims=True)
        det = (swpp * sw - swp * swp)
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        i0 = (sw * swpc - swp * swc) / det
        b0 = (swpp * swc - swp * swpc) / det
        i0 = np.maximum(i0, 0.0)
        b0 = np.maximum(b0, 1e-6)
    lam = i0 * p
    return lam / (lam + b0)


def gate_vb(p, _intensity, _background, counts, steps=1,
            alpha_i=1.0, beta_i=1e-3, alpha_b=1.0, beta_b=1e-3):
    """The mean-field update the model actually implies.

    With the pixel allocation `s_i` known the model is fully conjugate:
    `sum_i s_i` is sufficient for I (the profile sums to one, so the rate
    is just `I`), `sum_i (c_i - s_i)` is sufficient for B over N pixels,
    and `{s_i}` is sufficient for p. The only non-conjugate piece is the
    missing allocation, which makes this Poisson NMF with two components --
    one shaped like the profile, one flat.

    Coordinate ascent on that bound does NOT plug point estimates into
    `I p / (I p + B)`. It uses geometric expectations:

        r_i  proportional to  exp(E[log I] + log p_i)
        1-r_i proportional to exp(E[log B])

    and for `Gamma(a, b)`, `E[log x] = psi(a) - log b`, which is strictly
    below `log(a/b)`. That gap is the point: a component with little
    evidence is pulled DOWN rather than left at its point estimate, so a
    background that happens to be faint cannot drive the gate to one. The
    saturation that the median background produces is impossible here, not
    merely unlikely.

    Priors are weak and fixed (`beta = 1e-3`), so this is the structure
    being tested and not a prior doing the work.

    **`steps` defaults to 1, because iterating makes it worse.** Measured
    at low SNR, the median of `sum r c / I` over the step count: 1.022,
    1.051, 1.097, 1.160, 1.269 at 1, 2, 4, 8 and 16. It does not converge,
    it drifts, and the mechanism is already written down elsewhere in this
    repository. The conjugate shape for I is `alpha + sum_i r_i c_i` and
    the conjugate RATE is `beta + sum_i p_i`, which is `beta + 1` for every
    reflection because the profile is normalised. So assigning more counts
    to signal raises the shape and leaves the rate alone, `E[log I]` rises,
    and the gate assigns more still -- a feedback with nothing opposing it.

    `svae_integrator._get_qi` records the same thing from the other end:
    "the conjugate rate is tau + sum(profile*mask), and the profile is
    normalized, so it is tau + 1 for every reflection ... only the shape
    carries data." The one-parameter variational family and this runaway
    are the same defect seen twice.

    One step from a least-squares start is therefore the stable choice,
    and one step from a least-squares start is what `joint` already is.
    """
    from scipy.special import digamma

    npix = p.shape[1]
    i0, b0 = joint_fit(p, counts)
    i0 = np.maximum(i0, 1e-3)
    b0 = np.maximum(b0, 1e-6)
    logp = np.log(np.maximum(p, 1e-30))
    r = None
    for _ in range(steps):
        # E-step: responsibilities from geometric expectations
        a_i = alpha_i + (r * counts).sum(axis=1, keepdims=True) if r is not None \
            else alpha_i + i0
        b_i = beta_i + 1.0
        a_b = alpha_b + ((1 - r) * counts).sum(axis=1, keepdims=True) \
            if r is not None else alpha_b + b0 * npix
        b_b = beta_b + npix
        e_log_i = digamma(np.maximum(a_i, 1e-6)) - np.log(b_i)
        e_log_b = digamma(np.maximum(a_b, 1e-6)) - np.log(b_b)
        log_sig = e_log_i + logp
        log_bkg = np.broadcast_to(e_log_b, log_sig.shape)
        mx = np.maximum(log_sig, log_bkg)
        es, eb = np.exp(log_sig - mx), np.exp(log_bkg - mx)
        r = es / (es + eb)
    return r


GATES = {
    "exact": gate_exact,
    "plugin": gate_plugin,
    "tempered_0.7": lambda *a: gate_tempered(*a, a=0.7),
    "tempered_1.3": lambda *a: gate_tempered(*a, a=1.3),
    "sampled": gate_sampled,
    "untied": gate_untied,
    "topk": gate_topk,
    "split": gate_split,
    "plugin_noclamp": gate_plugin_noclamp,
    "plugin_lowmean": gate_plugin_lowmean,
    "joint": gate_joint,
    "joint_iter": gate_joint_iter,
    "vb": gate_vb,
}


def posterior_mean(i_hat, prior: str, wilson_scale=1e3):
    """q(I) mean under a frozen Wilson Gamma prior, or a flat one.

    The gate decides `i_hat`; the prior only shrinks it. Both are reported
    because a prior can mask a biased gate at the weak end, which is exactly
    where the gate matters.
    """
    if prior == "flat":
        return i_hat
    # frozen Wilson: Gamma(alpha=1, beta=1/Sigma), conjugate to the Poisson
    # count total, so the posterior mean is (alpha + sum) / (beta + exposure)
    alpha, beta = 1.0, 1.0 / wilson_scale
    return (alpha + np.maximum(i_hat, 0.0)) / (beta + 1.0)


def coupling(r, counts):
    """Does the split actually partition the counts?

    The reason to want an SVAE at all: in an amortized model q(I) and q(B)
    are separate heads with nothing holding them together, so the pair can
    drift along a direction that leaves the likelihood alone -- raise the
    background, lower the intensity, fit the same counts. In the SVAE the
    gate is supposed to tie them, because every count is assigned to signal
    or to background and `r + (1 - r) = 1` pixel by pixel.

    That identity holds for ANY r, which is the point: the tie is
    structural, not fitted. What varies is whether the split is in the
    right PLACE. So this returns both the partition residual, which should
    be zero to floating point for any gate, and the share of the total
    assigned to signal, which is what the gate actually decides.
    """
    total = counts.sum(axis=1)
    sig = (r * counts).sum(axis=1)
    bkg = ((1.0 - r) * counts).sum(axis=1)
    residual = np.abs(sig + bkg - total) / np.maximum(total, 1e-9)
    return float(np.max(residual)), sig / np.maximum(total, 1e-9)


REGIMES = {
    # name:           (I range as 10^x,  B range as 10^x)
    "laue_like": ((1.0, 4.0), (-0.5, 1.5)),
    "low_background": ((1.0, 4.0), (-2.0, -0.5)),
    "mono_like": ((2.0, 4.5), (-3.0, -1.5)),
    "low_snr": ((0.5, 1.5), (-0.5, 1.0)),
    "low_snr_low_bg": ((0.5, 1.5), (-2.5, -1.0)),
}


def simulate_regime(n: int, rng, regime: str):
    """Profiles and counts for one (intensity, background) regime."""
    (i_lo, i_hi), (b_lo, b_hi) = REGIMES[regime]
    p = streaky_profiles(n, rng)
    intensity = 10.0 ** rng.uniform(i_lo, i_hi, size=n)
    background = 10.0 ** rng.uniform(b_lo, b_hi, size=n)
    rate = intensity[:, None] * p + background[:, None]
    counts = rng.poisson(rate).astype(float)
    return p, intensity, background, counts


def sweep(n: int, seed: int) -> list[dict]:
    """Every gate in every regime, with the coupling diagnostic."""
    rows = []
    print(f"{'regime':<16}{'gate':<14}{'mean %':>8}{'median %':>10}"
          f"{'aggregate %':>13}{'share':>8}{'partition':>11}")
    for regime in REGIMES:
        rng = np.random.default_rng(seed)
        p, intensity, background, counts = simulate_regime(n, rng, regime)
        snr = float(np.median(intensity / (background * p.shape[1])))
        for name, fn in GATES.items():
            r = fn(p, intensity, background, counts)
            resid, share = coupling(r, counts)
            i_hat = (r * counts).sum(axis=1)
            ratio = i_hat / intensity
            # the mean of a ratio is not the ratio of means, and at low SNR
            # the per-reflection ratio is strongly right-skewed: a mean of
            # 1.25 sat on a median of 0.99. All three are reported so that
            # skew is never read as bias.
            rows.append({
                "regime": regime, "gate": name,
                "median_snr": round(snr, 3),
                "bias_pct": round(100 * (np.mean(ratio) - 1), 2),
                "median_bias_pct": round(100 * (np.median(ratio) - 1), 2),
                "aggregate_bias_pct": round(
                    100 * (float(np.sum(i_hat) / np.sum(intensity)) - 1), 2),
                "rel_sd_pct": round(100 * np.std(ratio), 2),
                "signal_share": round(float(np.mean(share)), 4),
                "partition_residual": float(f"{resid:.2e}"),
                "n": n,
            })
            print(f"{regime:<16}{name:<14}{rows[-1]['bias_pct']:>8.1f}"
                  f"{rows[-1]['median_bias_pct']:>10.1f}"
                  f"{rows[-1]['aggregate_bias_pct']:>13.1f}"
                  f"{rows[-1]['signal_share']:>8.3f}"
                  f"{rows[-1]['partition_residual']:>11.1e}")
        print()
    return rows


def measure_trained(path: str, out: str = "") -> list[dict]:
    """The same measurements, on TRAINED gates read out of a run.

    The simulation above scores candidate gates against an exact `r`
    computed from the true `(I, B)`. On real data there is no true
    `(I, B)`, so the reference is `joint_fit` -- least squares for both
    against `[p, 1]`, the same estimator `gate_joint` uses -- and the
    question becomes how far a trained gate sits from the responsibility
    its OWN profile and the observed counts imply.

    Reads an npz from `scripts/figures/dump_profile_maps.py` run with
    `REFL_IDS`, which stores `<arm>_named_*` for the SAME reflections in
    every arm. Positional arrays will not do: the training loader shuffles
    per run and the train/val split is drawn per run, so two arms' batches
    share no reflections at all.

    Reported per arm, over reflections and then per resolution third:

    - `rho_gate_rstar`  correlation of the gate with that reference `r`,
      within a shoebox, median over shoeboxes. 1.0 means the gate IS the
      responsibility of its own profile.
    - `rho_gate_excess` correlation of the gate with `c - b`, the
      background-subtracted counts. This is the question "is the gate
      where the signal is" asked of the data rather than of the model,
      with `b` the median of the shoebox's outer frame, which is
      background by construction.
    - `captured` the share of the positive excess counts the gate
      collects, `sum_i r_i (c_i - b)_+ / sum_i (c_i - b)_+`. A gate on the
      spot approaches 1; a gate on four pixels of a 25-pixel spot cannot.
    - `i_ratio` the intensity the gate implies over the one the reference
      responsibility implies, `sum r c / sum r* c`, median.
    - `n_eff_gate` `(sum r)^2 / sum r^2`, the number of pixels the gate
      effectively sums, which sets how much background it admits.
    - `precision` the share of what the gate ADMITS that is excess rather
      than background, `sum_i r_i (c_i - b)_+ / sum_i r_i c_i`. With
      `captured` it makes a precision/recall pair, and the pair is what
      separates "the gate covers the signal" from "the gate covers the
      detector": `captured` alone goes to 1 for a gate that is 1
      everywhere, and in a shoebox that is 96.6% background that gate is
      the worst one available.
    - `calib_err` how far the gate is from being a probability. If `r_i`
      is the posterior probability that a count at pixel `i` is signal,
      then over pixels binned by `r` the observed excess share
      `(c - b)_+ / c` should equal the bin's mean `r`. This is the mean
      absolute gap over ten bins, weighted by the counts in each -- the
      expected calibration error. A gate reading 0.8 where three tenths of
      the counts are excess is over-confident by 0.5 there, and that is
      exactly what inflates `sum_i r_i c_i`.
    - `calib_slope` the slope of observed excess share on `r` through those
      same bins. 1 is calibrated, below 1 over-confident at the top end.

    The calibration curve itself is written to `<out>` with
    `band = "calibration"` rows, one per bin, so it can be drawn rather
    than only summarised.
    """
    import numpy as np

    blob = np.load(path, allow_pickle=False)
    arms = sorted({k[: -len("_named_gate")] for k in blob
                   if k.endswith("_named_gate")})
    if not arms:
        raise SystemExit(
            f"{path} holds no `<arm>_named_gate`. Run "
            "dump_profile_maps.py with REFL_IDS set."
        )

    # every arm must have been measured on the same reflections, or the
    # comparison is between different reflections and says nothing
    id_sets = {a: blob[f"{a}_named_refl_id"].tolist() for a in arms}
    common = sorted(set.intersection(*[set(v) for v in id_sets.values()]))
    print(f"{len(arms)} arms, {len(common)} reflections common to all")
    if len(common) < 50:
        raise SystemExit(
            f"only {len(common)} shared reflections; the arms were not "
            "measured on the same ids"
        )

    rows, curve = [], []
    for arm in arms:
        order = {int(r): i for i, r in enumerate(id_sets[arm])}
        take = np.array([order[r] for r in common])
        g = blob[f"{arm}_named_gate"][take]
        prf = blob[f"{arm}_named_profile"][take]
        c = blob[f"{arm}_named_counts"][take]
        m = blob[f"{arm}_named_mask"][take].astype(bool)
        d = (blob[f"{arm}_named_d"][take]
             if f"{arm}_named_d" in blob else np.full(len(take), np.nan))

        # the reference responsibility, from the arm's own profile and the
        # observed counts
        i0, b0 = joint_fit(prf, c)
        lam = np.maximum(i0, 0.0) * prf
        rstar = lam / (lam + np.maximum(b0, 1e-6))

        # background from the outer frame of the box, which holds no spot
        side = int(round(float(np.sqrt(c.shape[1]))))
        edge = np.zeros(c.shape[1], bool)
        if side * side == c.shape[1]:
            img = np.arange(c.shape[1]).reshape(side, side)
            edge[img[0]] = edge[img[-1]] = True
            edge[img[:, 0]] = edge[img[:, -1]] = True
        else:
            edge[:] = True
        bg = np.array([
            np.median(c[i][m[i] & edge]) if (m[i] & edge).sum() else 0.0
            for i in range(len(c))
        ])
        excess = c - bg[:, None]

        thirds = (np.digitize(d, np.nanquantile(d, [1 / 3, 2 / 3]))
                  if np.isfinite(d).all() else np.zeros(len(d), int))
        # `d` is in angstrom, so the SMALLEST d is the highest resolution
        # and `digitize` puts it in band 0. Labelling band 0 `low_res`
        # inverted every resolution trend these tables report.
        for band, label in ((slice(None), "all"), (0, "high_res"),
                            (1, "mid_res"), (2, "low_res")):
            sel = (np.ones(len(c), bool) if label == "all"
                   else thirds == band)
            if sel.sum() < 10:
                continue
            rg, rx, cap, ratio, neff, prec = [], [], [], [], [], []
            # the calibration curve is pooled over reflections rather than
            # averaged per reflection: a single shoebox has too few pixels
            # in the high-`r` bins to estimate a share from
            n_bins = 10
            cal_num = np.zeros(n_bins)   # excess counts in the bin
            cal_den = np.zeros(n_bins)   # all counts in the bin
            cal_r = np.zeros(n_bins)     # sum of r, for the bin's mean r
            cal_n = np.zeros(n_bins)     # pixels in the bin
            for i in np.flatnonzero(sel):
                mm = m[i]
                if mm.sum() < 20:
                    continue
                gi, ri, ei = g[i][mm], rstar[i][mm], excess[i][mm]
                ci = c[i][mm]
                # pooled calibration: which bin each pixel's gate falls in,
                # and how much of that pixel's counts are excess
                b_idx = np.clip((gi * n_bins).astype(int), 0, n_bins - 1)
                pos_i = np.maximum(ei, 0.0)
                np.add.at(cal_num, b_idx, pos_i)
                np.add.at(cal_den, b_idx, np.maximum(ci, 0.0))
                np.add.at(cal_r, b_idx, gi)
                np.add.at(cal_n, b_idx, 1.0)
                if gi.std() > 0 and ri.std() > 0:
                    rg.append(float(np.corrcoef(gi, ri)[0, 1]))
                if gi.std() > 0 and ei.std() > 0:
                    rx.append(float(np.corrcoef(gi, ei)[0, 1]))
                pos = np.maximum(ei, 0.0)
                if pos.sum() > 0:
                    cap.append(float((gi * pos).sum() / pos.sum()))
                denom = float((ri * ci).sum())
                if abs(denom) > 1e-9:
                    ratio.append(float((gi * ci).sum() / denom))
                sg = float(gi.sum())
                if sg > 0:
                    neff.append(sg * sg / float((gi * gi).sum()))
                admitted = float((gi * ci).sum())
                if admitted > 1e-9:
                    prec.append(float((gi * pos_i).sum() / admitted))
            med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
            ok = cal_den > 0
            share = np.where(ok, cal_num / np.maximum(cal_den, 1e-9), np.nan)
            r_bin = np.where(cal_n > 0, cal_r / np.maximum(cal_n, 1e-9), np.nan)
            use = ok & np.isfinite(share) & np.isfinite(r_bin)
            if use.sum() >= 2:
                w = cal_den[use] / cal_den[use].sum()
                calib_err = float(np.abs(share[use] - r_bin[use]) @ w)
                calib_slope = float(
                    np.polyfit(r_bin[use], share[use], 1, w=np.sqrt(w))[0])
            else:
                calib_err = calib_slope = float("nan")
            rows.append({
                "method": arm.replace("hewl1118_", "").replace(
                    "hewl9b7c_", ""),
                "band": label,
                "n": int(sel.sum()),
                "rho_gate_rstar": med(rg),
                "rho_gate_excess": med(rx),
                "captured": med(cap),
                "i_ratio": med(ratio),
                "n_eff_gate": med(neff),
                "precision": med(prec),
                "calib_err": calib_err,
                "calib_slope": calib_slope,
            })
            if label == "all":
                for k in np.flatnonzero(use):
                    curve.append({
                        "method": rows[-1]["method"],
                        "band": "calibration",
                        "bin": int(k),
                        "gate_mean": float(r_bin[k]),
                        "excess_share": float(share[k]),
                        "counts": float(cal_den[k]),
                    })

    width = max(len(r["method"]) for r in rows) + 2
    print(f"{'arm':<{width}}{'band':<10}{'rho(g,r*)':>11}"
          f"{'rho(g,c-b)':>12}{'captured':>10}{'precis':>8}"
          f"{'I/I*':>8}{'n_eff':>8}{'calib':>8}{'slope':>7}")
    for r in rows:
        print(f"{r['method']:<{width}}{r['band']:<10}"
              f"{r['rho_gate_rstar']:>11.3f}{r['rho_gate_excess']:>12.3f}"
              f"{r['captured']:>10.3f}{r['precision']:>8.3f}"
              f"{r['i_ratio']:>8.3f}"
              f"{r['n_eff_gate']:>8.1f}{r['calib_err']:>8.3f}"
              f"{r['calib_slope']:>7.2f}")
    # the calibration curve: what share of the counts under a gate of a
    # given size are actually excess. The diagonal is calibrated.
    if curve:
        print(f"\n{'arm':<{width}}{'gate':>7}{'excess share':>14}"
              f"{'counts':>12}")
        for c_row in curve:
            print(f"{c_row['method']:<{width}}{c_row['gate_mean']:>7.2f}"
                  f"{c_row['excess_share']:>14.3f}{c_row['counts']:>12.3g}")
    if out:
        import csv
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{out}: {len(rows)} rows")
        if curve:
            cpath = out.replace(".csv", "") + "_calibration.csv"
            with open(cpath, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(curve[0]))
                w.writeheader()
                w.writerows(curve)
            print(f"{cpath}: {len(curve)} rows")
    return rows


# the prior rate the conjugate update adds, from the trained runs: beta'
# comes out at 1 + TAU because the profile is normalised
TAU = 0.0097


def variance_rules(p_row, counts, tau=TAU, q_kappa=None, draws=4,
                   rng=None):
    """Every claim about the error bar, on one set of shoeboxes.

    Each rule is `(estimate, reported variance)` for ONE estimator, so a
    rule is only ever scored against the spread of the estimator it
    belongs to.

    - `conjugate`  what `_get_qi` reports: Gamma(alpha', beta') with
      alpha' = alpha_I + sum_i r_i c_i and beta' = beta_I + sum_i p_i = 1 + tau.
      Fano = 1/beta' <= 1 for every reflection.
    - `split`      the same estimator with the signal/background split
      marginalised rather than conditioned on. Each photon at pixel i is
      signal with probability r_i, so S = sum_i s_i carries a binomial
      term sum_i c_i r_i (1 - r_i) on top of the Poisson one, and the law
      of total variance gives
      Var = (alpha' + sum_i c_i r_i (1 - r_i)) / beta'^2 in closed form.
      This is the term the augmented model discards.
    - `naive`      Var[sum_i r_i c_i] = sum_i r_i^2 (I p_i + B) with r held
      FIXED, which ignores that r was computed from the same counts.
    - `fisher`     the Schur complement of the Poisson information for
      (I, B): J_II = sum p^2/lam, J_IB = sum p/lam, J_BB = sum 1/lam and
      Var = J_BB / (J_II J_BB - J_IB^2). The background is profiled out
      rather than conditioned on. This is what the Rao-Blackwellized
      export reports.
    - `fisher_marg` the same, with the PROFILE marginalised as well as the
      background. `fisher` is conditional on the one profile it was
      handed; on real data that profile is itself estimated, and its
      error moves the estimate from exposure to exposure. With a
      `q(p) = Dirichlet(q_kappa * p_hat)` the law of total variance gives
      `E_p[Var(I|p)] + Var_p[E(I|p)]`, and the second term is the
      contribution `fisher` is missing. Only built when `q_kappa` is set.
    """
    i0, b0 = plugin_estimates(p_row, counts)
    lam = i0 * p_row
    r = lam / (lam + np.maximum(b0, 1e-6))
    beta = 1.0 + tau
    alpha = (r * counts).sum(axis=1)
    gate_sum = (r * counts).sum(axis=1)
    split_term = (counts * r * (1.0 - r)).sum(axis=1)
    rate = np.maximum(i0 * p_row + b0, 1e-9)

    j_ii = (p_row ** 2 / rate).sum(axis=1)
    j_ib = (p_row / rate).sum(axis=1)
    j_bb = (1.0 / rate).sum(axis=1)
    det = np.maximum(j_ii * j_bb - j_ib ** 2, 1e-30)

    out = {
        "conjugate": (alpha / beta, alpha / beta ** 2),
        "split": (alpha / beta, (alpha + split_term) / beta ** 2),
        "naive": (gate_sum, (r ** 2 * rate).sum(axis=1)),
        "fisher": (i0.ravel(), j_bb / det),
    }
    if q_kappa:
        rng = rng or np.random.default_rng(0)
        means, vars_ = [], []
        for _ in range(draws):
            ps = rng.gamma(np.maximum(q_kappa * p_row, 1e-6))
            ps = ps / np.maximum(ps.sum(axis=1, keepdims=True), 1e-30)
            i_s, b_s = plugin_estimates(ps, counts)
            lam_s = np.maximum(i_s * ps + b_s, 1e-9)
            a_ii = (ps ** 2 / lam_s).sum(axis=1)
            a_ib = (ps / lam_s).sum(axis=1)
            a_bb = (1.0 / lam_s).sum(axis=1)
            d_s = np.maximum(a_ii * a_bb - a_ib ** 2, 1e-30)
            means.append(i_s.ravel())
            vars_.append(a_bb / d_s)
        means = np.asarray(means)
        vars_ = np.asarray(vars_)
        out["fisher_marg"] = (means.mean(axis=0),
                              vars_.mean(axis=0) + means.var(axis=0, ddof=1))
    return out


def robust_sd(z: np.ndarray) -> float:
    """The spread of the z scores, by the median absolute deviation.

    A plain standard deviation is useless here: a weak reflection whose
    gate collects almost nothing gets a reported variance near zero, and
    one such z of 1e15 sets the whole column. The MAD estimator reports
    what the bulk of the distribution does, which is the question --
    whether the typical error bar is the typical error.
    """
    v = np.asarray(z).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def calibrate_variance(n_refl: int, n_rep: int, seed: int,
                       profile_kappa: float | None = None,
                       draws: int = 4) -> list[dict]:
    """Does the reported error bar equal the spread of the estimate?

    The question merging asks is frequentist and it is not the question a
    posterior answers. careless hands `SigI` to a Student-t likelihood as
    the scale of the OBSERVATION around its expectation, so what it needs
    is the sampling variance of the reported number: hold the crystal, the
    beam and the reflection fixed, re-expose, and ask how much the
    reported intensity bounces. A posterior credible width is a different
    quantity -- it depends on the prior, and the Wilson prior is already
    inside careless, so exporting a posterior applies it twice.

    This measures the frequentist quantity directly, which simulation can
    do and real data cannot: fix `(I, B, p)`, draw `n_rep` independent
    count realisations of the SAME reflection, run each rule on each, and
    compare the mean reported variance with the empirical spread of that
    rule's own estimate.

    Reported per rule: `ratio` = mean(reported Var) / Var(estimate) over
    realisations, 1.0 being honest and below 1 over-confident; `z_sd` =
    the standard deviation of (estimate - I) / sqrt(reported Var), 1.0
    being honest; and `bias` = mean(estimate)/I - 1, because a variance is
    only meaningful about a value that is in the right place.
    """
    rng = np.random.default_rng(seed)
    p_row = streaky_profiles(n_refl, rng)
    intensity = 10.0 ** rng.uniform(1.0, 4.0, size=n_refl)
    background = 10.0 ** rng.uniform(-0.5, 1.5, size=n_refl)
    rate = intensity[:, None] * p_row + background[:, None]

    # the Cramer-Rao floor, from the TRUE (I, B, p): the information the
    # experiment contains, which no estimator of I can beat. Reported as
    # `efficiency` = floor / achieved spread, so 1.0 is an estimator that
    # extracts everything the shoebox holds and 0.5 is one throwing half
    # of it away. It bounds UNBIASED estimators, so a biased one can show
    # above 1 and that is a warning, not a win.
    t_ii = (p_row ** 2 / rate).sum(axis=1)
    t_ib = (p_row / rate).sum(axis=1)
    t_bb = (1.0 / rate).sum(axis=1)
    crb = t_bb / np.maximum(t_ii * t_bb - t_ib ** 2, 1e-30)

    est = {}
    var = {}
    for _ in range(n_rep):
        counts = rng.poisson(rate).astype(float)
        # the profile the ESTIMATOR sees. On real data it is predicted
        # from these very counts, so it is redrawn per realisation: a
        # profile fixed across exposures would show up as bias, not as
        # spread, and would leave the variance question untouched.
        seen = p_row
        if profile_kappa:
            seen = rng.gamma(np.maximum(profile_kappa * p_row, 1e-6))
            seen = seen / np.maximum(seen.sum(axis=1, keepdims=True), 1e-30)
        for name, (e, v) in variance_rules(
                seen, counts, q_kappa=profile_kappa, draws=draws,
                rng=rng).items():
            est.setdefault(name, []).append(e)
            var.setdefault(name, []).append(v)

    # a reflection is weak or bright by its TRUE intensity, never by the
    # estimate: binning on the estimator's own output selects the noise
    thirds = np.digitize(np.log10(intensity),
                         np.quantile(np.log10(intensity), [1 / 3, 2 / 3]))
    rows = []
    for name in est:
        e = np.asarray(est[name])          # (n_rep, n_refl)
        v = np.asarray(var[name])
        spread = e.var(axis=0, ddof=1)     # the thing a sigma should equal
        reported = v.mean(axis=0)
        z = (e - intensity[None, :]) / np.sqrt(np.maximum(v, 1e-30))
        for band, label in ((slice(None), "all"), (0, "weak"),
                            (1, "mid"), (2, "bright")):
            sel = (np.ones(n_refl, bool) if label == "all"
                   else thirds == band)
            if sel.sum() < 5:
                continue
            rows.append({
                "rule": name,
                "band": label,
                "n": int(sel.sum()),
                "ratio": float(np.median(reported[sel]
                                         / np.maximum(spread[sel], 1e-30))),
                "z_sd": robust_sd(z[:, sel]),
                "bias": float(np.median(e[:, sel].mean(axis=0)
                                        / intensity[sel] - 1.0)),
                "fano": float(np.median(reported[sel]
                                        / np.maximum(
                                            e[:, sel].mean(axis=0), 1e-30))),
                "efficiency": float(np.median(
                    crb[sel] / np.maximum(spread[sel], 1e-30))),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--sweep", action="store_true",
                    help="every gate across the intensity/background "
                         "regimes, with the partition diagnostic")
    ap.add_argument("--variance", action="store_true",
                    help="is the reported error bar the spread of the "
                         "estimate? repeated realisations of the same "
                         "simulated reflection")
    ap.add_argument("--profile-error", type=float, default=None,
                    metavar="KAPPA",
                    help="the estimator sees Dirichlet(KAPPA p) instead of "
                         "the true profile, redrawn per realisation; "
                         "smaller is noisier")
    ap.add_argument("--reps", type=int, default=200,
                    help="count realisations per reflection for --variance")
    ap.add_argument("--npz", default="",
                    help="measure TRAINED gates out of a "
                         "dump_profile_maps.py npz written with REFL_IDS, "
                         "instead of simulating")
    args = ap.parse_args()

    if args.npz:
        measure_trained(args.npz, args.out)
        return 0

    if args.variance:
        rows = calibrate_variance(args.n, args.reps, args.seed,
                                  args.profile_error)
        width = max(len(r["rule"]) for r in rows) + 2
        if args.profile_error:
            print(f"profile seen by the estimator: "
                  f"Dirichlet({args.profile_error} p), redrawn per "
                  f"realisation\n")
        print(f"{'rule':<{width}}{'band':<8}{'reported/spread':>16}"
              f"{'z sd':>8}{'bias':>9}{'Fano':>8}{'efficiency':>12}")
        for r in rows:
            print(f"{r['rule']:<{width}}{r['band']:<8}{r['ratio']:>16.3f}"
                  f"{r['z_sd']:>8.2f}{r['bias']:>9.3f}{r['fano']:>8.2f}"
                  f"{r['efficiency']:>12.3f}")
        if args.out:
            with open(args.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\n{args.out}: {len(rows)} rows")
        return 0

    if args.sweep:
        rows = sweep(args.n, args.seed)
        if args.out:
            with open(args.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"{args.out}: {len(rows)} rows")
        return 0

    rng = np.random.default_rng(args.seed)
    p, intensity, background, counts = simulate(args.n, rng)

    # the weak end is where a gate is actually tested
    edges = np.quantile(intensity, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    print(f"{args.n} shoeboxes, {H}x{W}, streaky Gaussians\n")
    print(f"{'gate':<14}{'prior':<8}{'bias %':>9}{'rel sd %':>10}"
          f"{'weak bias %':>13}{'E[sum r c]/I':>14}")
    for name, fn in list(GATES.items()) + [("profile_LS", None),
                                           ("joint_LS", "joint")]:
        if fn is None:
            i_hat = estimator_ls(p, counts)
        elif fn == "joint":
            i_hat = np.maximum(joint_fit(p, counts)[0].ravel(), 0.0)
        else:
            r = fn(p, intensity, background, counts)
            i_hat = (r * counts).sum(axis=1)
        ratio = i_hat / intensity
        for prior in ("flat", "wilson"):
            est = posterior_mean(i_hat, prior)
            rel = est / intensity
            weak = rel[intensity <= edges[1]]
            rows.append({
                "gate": name, "prior": prior,
                "bias_pct": round(100 * (np.mean(rel) - 1), 3),
                "rel_sd_pct": round(100 * np.std(rel), 3),
                "weak_bias_pct": round(100 * (np.mean(weak) - 1), 3),
                "mean_ratio": round(float(np.mean(ratio)), 4),
                "n": args.n,
            })
            print(f"{name:<14}{prior:<8}{rows[-1]['bias_pct']:>9.2f}"
                  f"{rows[-1]['rel_sd_pct']:>10.2f}"
                  f"{rows[-1]['weak_bias_pct']:>13.2f}"
                  f"{rows[-1]['mean_ratio']:>14.3f}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{args.out}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
