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

    rows = []
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
        for band, label in ((slice(None), "all"), (0, "low_res"),
                            (1, "mid_res"), (2, "high_res")):
            sel = (np.ones(len(c), bool) if label == "all"
                   else thirds == band)
            if sel.sum() < 10:
                continue
            rg, rx, cap, ratio, neff = [], [], [], [], []
            for i in np.flatnonzero(sel):
                mm = m[i]
                if mm.sum() < 20:
                    continue
                gi, ri, ei = g[i][mm], rstar[i][mm], excess[i][mm]
                ci = c[i][mm]
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
            med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
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
            })

    width = max(len(r["method"]) for r in rows) + 2
    print(f"{'arm':<{width}}{'band':<10}{'rho(g,r*)':>11}"
          f"{'rho(g,c-b)':>12}{'captured':>10}{'I/I*':>8}{'n_eff':>8}")
    for r in rows:
        print(f"{r['method']:<{width}}{r['band']:<10}"
              f"{r['rho_gate_rstar']:>11.3f}{r['rho_gate_excess']:>12.3f}"
              f"{r['captured']:>10.3f}{r['i_ratio']:>8.3f}"
              f"{r['n_eff_gate']:>8.1f}")
    if out:
        import csv
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{out}: {len(rows)} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--sweep", action="store_true",
                    help="every gate across the intensity/background "
                         "regimes, with the partition diagnostic")
    ap.add_argument("--npz", default="",
                    help="measure TRAINED gates out of a "
                         "dump_profile_maps.py npz written with REFL_IDS, "
                         "instead of simulating")
    args = ap.parse_args()

    if args.npz:
        measure_trained(args.npz, args.out)
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
