"""Per-image Wilson PRIOR parameters in the integrator: what recovers the oracle G_i, B_i?

Strictly the integrator's Wilson prior -- no scaling or merging model anywhere here. The
prior on each reflection's intensity is, exactly as in `MonochromaticWilsonLoss`:

    s^2  = 1 / (4 d^2)
    tau  = (1/G) * exp(2 B s^2)
    I    ~ Exp(tau)            i.e.  E[I] = G * exp(-2 B s^2)

and G, B enter the ELBO ONLY through `kl_i = KL(q(I) || Exp(tau))`. In SFX each image is
a different crystal hit by a different pulse, so the *observed* intensity scale varies
image to image and a single global G misdescribes every image at once. Two models:

    case 1   G per image, B global      (random intercept)
    case 2   G per image, B per image   (random intercept + random slope)

THE KEY FACT, which decides the architecture. Write the tau-dependent part of the KL for
the reflections j of image i:

    L(G_i, B) = sum_j [ log G_i - 2 B s_j^2 + (1/G_i) exp(2 B s_j^2) E_j ],   E_j = E_q[I_j]

Setting dL/dG_i = 0 gives a CLOSED FORM:

    G_i = (1/n_i) * sum_j exp(2 B s_j^2) * E_j

i.e. G_i is just the mean of the B-corrected posterior intensities in that image. dL/dB
= 0 gives a 1-D root in B (monotone, so bisection). So the per-image prior parameters are
SOLVED, not learned -- there is nothing for a network to amortize that arithmetic cannot
do exactly. Unlike a scaling model there is also no gauge freedom: q(I) is pinned by the
counts, so G_i is an ordinary hyperparameter of a distribution whose samples we see.

The real risk is the opposite of under-fitting. A free per-image G is fit to the very
reflections it regularizes, so with few reflections per image it tracks their noise, the
KL collapses toward zero, and the prior stops regularizing. `study_overfit` measures that.

Estimators compared:
    oracle    true G_i, B                       (ceiling)
    global    one G for all images              (what the loss does today; under-fits)
    solved    closed-form coordinate descent    (exact MLE of the hyperparameters)
    shrunk    solved + hierarchical pooling of log G_i toward the population
    amortized log G_i from image summary stats  (the restricted-function-class option)

Run:  uv run python scripts/jungfrau_sim/wilson_per_image_prior.py [--only case1]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

D_MIN, D_MAX = 1.5, 20.0  # resolution range in Angstrom


def simulate(
    n_images: int,
    n_refl_per_image: int,
    sigma_logG: float,
    b_global: float,
    sigma_B: float,
    bg: float,
    seed: int,
    g_scale: float = 200.0,
    d_min: float = D_MIN,
    d_max: float = D_MAX,
    partiality_mean: float = 1.0,
    partiality_conc: float = 2.0,
    sigma_partiality_image: float = 0.0,
) -> dict:
    """Draw per-image Wilson prior parameters and the reflections they generate.

    `sigma_B > 0` gives case 2 (per-image B); `sigma_B == 0` gives case 1 (global B).
    Resolutions are sampled uniformly in reciprocal-space VOLUME (shell volume grows as
    s^2 ds), so high-resolution reflections dominate the count as they do in real data.

    PARTIALITY. In SFX every reflection is a partial observation: the crystal never
    rotates, so the reciprocal-lattice point only clips the Ewald sphere and a fraction
    p in (0, 1] of the full intensity is recorded. The physically correct order is that
    the FULL intensity is Wilson-distributed (that is a statement about the structure)
    and the observation is p times it:

        I_full ~ Exp(mean = G_i exp(-2 B s^2)),   I_obs = p_ij * I_full

    `partiality_mean` sets the population mean of p, `partiality_conc` its Beta
    concentration (higher = tighter), and `sigma_partiality_image` lets each crystal have
    its own mean partiality (different mosaicity/size). `partiality_mean = 1` disables it.
    """
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)  # Beta draws; torch's lacks a generator arg

    # G is the scale of OBSERVED intensities in photons, so it is O(100), not O(1).
    # At G~1 every reflection past mid-resolution is buried in background and the whole
    # study measures nothing but noise.
    log_g = torch.randn(n_images, generator=g) * sigma_logG
    g_true = g_scale * log_g.exp()
    b_true = b_global + torch.randn(n_images, generator=g) * sigma_B

    s_min, s_max = 1.0 / (2.0 * d_max), 1.0 / (2.0 * d_min)
    u = torch.rand(n_images, n_refl_per_image, generator=g)
    s = (u * (s_max**3 - s_min**3) + s_min**3) ** (1.0 / 3.0)
    s_sq = s**2

    image_idx = (
        torch.arange(n_images).unsqueeze(1).expand(-1, n_refl_per_image).reshape(-1)
    )
    s_sq = s_sq.reshape(-1)

    # FULL intensity is the Wilson-distributed quantity: I_full ~ Exp(tau),
    # tau = (1/G) exp(2 B s^2)  =>  E[I_full] = G exp(-2 B s^2)
    mean_i = g_true[image_idx] * torch.exp(-2.0 * b_true[image_idx] * s_sq)
    u2 = torch.rand(mean_i.shape, generator=g).clamp_min(1e-12)
    i_full = -mean_i * torch.log(u2)

    if partiality_mean >= 1.0:
        p = torch.ones_like(i_full)
        m_image = torch.ones(n_images)
    else:
        m_image = torch.tensor(
            np.clip(
                partiality_mean
                + rng.normal(0.0, sigma_partiality_image, size=n_images),
                0.05,
                0.99,
            )
        )
        m_refl = m_image[image_idx].numpy()
        k = partiality_conc
        p = torch.tensor(rng.beta(m_refl * k, (1.0 - m_refl) * k)).clamp(1e-4, 1.0)

    i_obs = p * i_full
    counts = torch.poisson(i_obs + bg, generator=g)

    return {
        "image_idx": image_idx,
        "s_sq": s_sq,
        "i_true": i_obs,  # what the integrator's q(I) targets: the OBSERVED intensity
        "i_full": i_full,
        "partiality": p,
        "counts": counts,
        "g_true": g_true,
        # The prior on the OBSERVED intensity has mean m_i * G_i exp(-2Bs^2), so the
        # scale a correctly-specified per-image prior should recover is G_i * m_i,
        # not G_i. Per-image mean partiality is degenerate with per-image G.
        "g_effective": g_true * m_image,
        "m_image": m_image,
        "b_true": b_true,
        "b_global": b_global,
        "n_images": n_images,
        "n_refl": n_refl_per_image,
        "bg": bg,
        "per_image_B": sigma_B > 0,
        "partiality_mean": partiality_mean,
        "partiality_conc": partiality_conc,
    }


def intensities(sim: dict, oracle: bool) -> torch.Tensor:
    """E_q[I] per reflection: the true value (ceiling) or a background-subtracted estimate."""
    if oracle:
        return sim["i_true"]
    return (sim["counts"] - sim["bg"]).clamp_min(0.0)


def _solve_G(e: torch.Tensor, s_sq: torch.Tensor, image_idx, n_images, b_per_refl):
    """Closed-form G per image: mean of the B-corrected intensities (dL/dG = 0)."""
    corrected = torch.exp(2.0 * b_per_refl * s_sq) * e
    num = torch.bincount(image_idx, weights=corrected, minlength=n_images)
    den = torch.bincount(image_idx, minlength=n_images).clamp_min(1)
    return (num / den).clamp_min(1e-12)


def _solve_B(e, s_sq, g_per_refl, lo=0.0, hi=300.0, n_bisect=60):
    """Root of dL/dB = sum_j s_j^2 [ (1/G) exp(2 B s_j^2) E_j - 1 ] = 0 (monotone in B)."""

    def f(b):
        return (s_sq * ((1.0 / g_per_refl) * torch.exp(2.0 * b * s_sq) * e - 1.0)).sum()

    lo_t = torch.tensor(lo)
    hi_t = torch.tensor(hi)
    if f(lo_t) > 0:  # already over-corrected at B=0
        return lo_t
    for _ in range(n_bisect):
        mid = 0.5 * (lo_t + hi_t)
        if f(mid) < 0:
            lo_t = mid
        else:
            hi_t = mid
    return 0.5 * (lo_t + hi_t)


def fit_global(sim: dict, e: torch.Tensor, n_iter: int = 40):
    """One G and one B for the whole dataset -- what the loss does today."""
    n = sim["n_images"]
    b = torch.tensor(sim["b_global"])
    g_scalar = torch.tensor(1.0)
    for _ in range(n_iter):
        g_scalar = (torch.exp(2.0 * b * sim["s_sq"]) * e).mean().clamp_min(1e-12)
        b = _solve_B(e, sim["s_sq"], g_scalar.expand_as(e))
    return g_scalar.expand(n), b.expand(n)


def fit_solved(sim: dict, e: torch.Tensor, n_iter: int = 40):
    """Per-image G (closed form) with a GLOBAL B -- case 1's exact MLE."""
    image_idx, s_sq, n = sim["image_idx"], sim["s_sq"], sim["n_images"]
    b = torch.tensor(sim["b_global"])
    g_hat = torch.ones(n)
    for _ in range(n_iter):
        g_hat = _solve_G(e, s_sq, image_idx, n, b.expand_as(s_sq))
        b = _solve_B(e, s_sq, g_hat[image_idx])
    return g_hat, b.expand(n)


def fit_solved_per_image_B(sim: dict, e: torch.Tensor, n_iter: int = 30):
    """Per-image G AND per-image B -- case 2's exact MLE, one Wilson fit per image."""
    image_idx, s_sq, n = sim["image_idx"], sim["s_sq"], sim["n_images"]
    g_hat = torch.ones(n)
    b_hat = torch.full((n,), float(sim["b_global"]))
    for _ in range(n_iter):
        g_hat = _solve_G(e, s_sq, image_idx, n, b_hat[image_idx])
        for i in range(n):  # each image gets its own 1-D root find
            m = image_idx == i
            b_hat[i] = _solve_B(e[m], s_sq[m], g_hat[i].expand(int(m.sum())))
    return g_hat, b_hat


def fit_shrunk(sim: dict, e: torch.Tensor, prior_tau: float, n_iter: int = 40):
    """Per-image G pooled toward the population: log G_i ~ N(mu, prior_tau^2).

    The MLE of log G_i has sampling variance ~1/n_i (Exponential samples), so the
    posterior mean is the precision-weighted blend of the MLE and the population mean.
    This is the hierarchical answer to a free per-image parameter over-fitting.
    """
    g_mle, b = fit_solved(sim, e, n_iter=n_iter)
    log_mle = torch.log(g_mle.clamp_min(1e-12))
    n_i = torch.bincount(sim["image_idx"], minlength=sim["n_images"]).clamp_min(1)
    # Var(log of an Exponential mean estimated from n samples) ~ 1/n.
    w = (prior_tau**2) / (prior_tau**2 + 1.0 / n_i)
    mu = log_mle.mean()
    return torch.exp(mu + w * (log_mle - mu)), b


def fit_amortized(sim: dict, e: torch.Tensor, n_iter: int = 40):
    """log G_i from image summary statistics, by least squares on those features.

    The features are the honest 'global image stats' an amortizing network would see:
    log of the image's mean intensity, its mean s^2, and log n_refl. Because the exact
    solution is a mean of B-corrected intensities, a linear model on these stats can get
    close -- which is precisely why a network buys little over the closed form.
    """
    _, b = fit_solved(sim, e, n_iter=n_iter)
    idx, s_sq, n = sim["image_idx"], sim["s_sq"], sim["n_images"]
    cnt = torch.bincount(idx, minlength=n).clamp_min(1)

    mean_e = torch.bincount(idx, weights=e, minlength=n) / cnt
    mean_s = torch.bincount(idx, weights=s_sq, minlength=n) / cnt
    feats = torch.stack(
        [
            torch.log(mean_e.clamp_min(1e-6)),
            mean_s,
            torch.log(cnt.double()),
            torch.ones(n),
        ],
        dim=1,
    )
    target = torch.log(fit_solved(sim, e, n_iter=n_iter)[0].clamp_min(1e-12))
    coef = torch.linalg.lstsq(feats, target.unsqueeze(1)).solution
    return torch.exp((feats @ coef).squeeze(1)), b


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact Pearson correlation.

    NOT cov.mean()/(a.std()*b.std()) -- torch's .std() applies Bessel's correction while
    .mean() does not, which caps a perfect correlation at (n-1)/n and silently makes an
    oracle look imperfect.
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def score(g_hat, b_hat, sim: dict) -> dict:
    lg_hat = torch.log(g_hat.clamp_min(1e-12))
    lg_true = torch.log(sim["g_true"])
    corr = _corr(lg_hat, lg_true)
    return {
        "corr_logG": corr,
        "rmse_logG": float((lg_hat - lg_true).pow(2).mean().sqrt()),
        "rmse_B": float((b_hat - sim["b_true"]).pow(2).mean().sqrt()),
    }


def _row(name: str, m: dict) -> str:
    return (
        f"    {name:<11} corr(logG) {m['corr_logG']:6.3f}   "
        f"rmse(logG) {m['rmse_logG']:6.3f}   rmse(B) {m['rmse_B']:6.2f}"
    )


def _methods(sim, e, per_image_b: bool, prior_tau: float) -> dict:
    out = {
        "oracle": (sim["g_true"], sim["b_true"]),
        "global": fit_global(sim, e),
        "solved": fit_solved(sim, e),
        "shrunk": fit_shrunk(sim, e, prior_tau),
        "amortized": fit_amortized(sim, e),
    }
    if per_image_b:
        out["solved+Bi"] = fit_solved_per_image_B(sim, e)
    return out


def study_case1(args) -> None:
    """Case 1: G per image, B global."""
    print("\n" + "=" * 80)
    print("CASE 1 -- G per image, B global   (random intercept)")
    print("=" * 80)
    print(f"   200 images, sigma_logG={args.sigma_logg}, B={args.b_global} A^2,")
    print("   intensities estimated from counts (not oracle). Sweeping refl/image.\n")

    for n_refl in [5, 20, 100, 500]:
        sim = simulate(200, n_refl, args.sigma_logg, args.b_global, 0.0,
                       args.bg, args.seed, g_scale=args.g_scale)
        e = intensities(sim, oracle=False)
        print(f"  {n_refl:>4} refl/image")
        for name, (gh, bh) in _methods(sim, e, False, args.sigma_logg).items():
            print(_row(name, score(gh, bh, sim)))
        print()

    print("   `solved` is the exact closed form; a free per-image parameter optimized")
    print("   by SGD minimizes the SAME objective, so it can do no better. `global`")
    print("   shows the cost of the current single-G loss.")


def study_case2(args) -> None:
    """Case 2: G per image and B per image."""
    print("\n" + "=" * 80)
    print("CASE 2 -- G per image, B per image   (random intercept + random slope)")
    print("=" * 80)
    print(f"   sigma_B={args.sigma_b} A^2 about B={args.b_global}. B_i is a SLOPE against")
    print("   s^2, so it needs resolution SPREAD within an image, not just reflections.")
    print("   `solved` fits per-image G with one global B; `solved+Bi` fits both.\n")

    for n_refl in [5, 20, 100, 500]:
        sim = simulate(200, n_refl, args.sigma_logg, args.b_global, args.sigma_b,
                       args.bg, args.seed, g_scale=args.g_scale)
        e = intensities(sim, oracle=False)
        print(f"  {n_refl:>4} refl/image")
        for name, (gh, bh) in _methods(sim, e, True, args.sigma_logg).items():
            print(_row(name, score(gh, bh, sim)))
        print()

    print(f"   Note rmse(B) for `solved`: it reports the global B for every image, so its")
    print(f"   error floor is the true spread sigma_B={args.sigma_b}. `solved+Bi` beats")
    print("   that only once each image has enough reflections to fit its own slope.")


def study_ceiling(args) -> None:
    """How much of the error is the estimator vs the intensity noise feeding it?"""
    print("\n" + "=" * 80)
    print("CEILING -- estimator error vs intensity noise")
    print("=" * 80)
    print("   Same fit, but given the TRUE intensities instead of count-derived ones.")
    print("   The gap is what better intensity estimates could still buy.\n")

    print(f"    {'refl/img':>9} | {'solved (counts)':>17} {'solved (true I)':>17}")
    print("    " + "-" * 48)
    for n_refl in [5, 20, 100, 500]:
        sim = simulate(200, n_refl, args.sigma_logg, args.b_global, 0.0,
                       args.bg, args.seed, g_scale=args.g_scale)
        noisy = score(*fit_solved(sim, intensities(sim, False)), sim)
        clean = score(*fit_solved(sim, intensities(sim, True)), sim)
        print(f"    {n_refl:>9} | {noisy['corr_logG']:>17.3f} {clean['corr_logG']:>17.3f}")

    print("\n   Close together = the closed form is already extracting nearly everything")
    print("   the intensities contain, so effort belongs in the intensity model, not in")
    print("   a fancier G estimator.")


def _overdispersion(sim: dict, g_hat: torch.Tensor, b_hat: torch.Tensor) -> float:
    """Var/mean^2 of the intensities after dividing out the fitted prior mean.

    A correctly specified Exp prior gives normalized values ~ Exp(1), for which
    Var/mean^2 == 1 exactly. Anything above 1 is the prior being the wrong SHAPE, which
    is the part of partiality a per-image scale cannot absorb.
    """
    mean_fit = g_hat[sim["image_idx"]] * torch.exp(
        -2.0 * b_hat[sim["image_idx"]] * sim["s_sq"]
    )
    z = sim["i_true"] / mean_fit.clamp_min(1e-12)
    return float(z.var() / z.mean().pow(2).clamp_min(1e-12))


def study_partiality(args) -> None:
    """Does a per-observation Wilson prior survive partial reflections?"""
    print("\n" + "=" * 80)
    print("PARTIALITY -- the SFX-specific misspecification of a per-observation Wilson prior")
    print("=" * 80)
    print("   I_full ~ Exp(G_i exp(-2Bs^2));  I_obs = p * I_full,  p ~ Beta(mean m_i, conc k).")
    print("   Two separable effects:")
    print("     MEAN  E[I_obs] = m_i * G_i * exp(-2Bs^2)  -> absorbed into a per-image G.")
    print("     SHAPE p varies per reflection, so I_obs is a scale MIXTURE of")
    print("           exponentials: over-dispersed, and no per-image scalar fixes that.")
    print("   Predicted over-dispersion for Beta(m, k):  1 + 2(1-m)/(m(k+1)).\n")

    print(f"    {'E[p]':>6} {'k':>5} | {'corr vs G_true':>15} {'corr vs G_i*m_i':>16}"
          f" | {'overdisp':>9} {'predicted':>10} {'alpha':>7}")
    print("    " + "-" * 76)

    for m, k in [(1.0, 2.0), (0.8, 8.0), (0.6, 4.0), (0.5, 2.0), (0.3, 1.5)]:
        sim = simulate(
            200, 100, args.sigma_logg, args.b_global, 0.0, args.bg, args.seed,
            g_scale=args.g_scale, partiality_mean=m, partiality_conc=k,
            sigma_partiality_image=args.sigma_partiality_image,
        )
        e = intensities(sim, oracle=False)
        g_hat, b_hat = fit_solved(sim, e)

        lg = torch.log(g_hat.clamp_min(1e-12))
        c_true = _corr(lg, torch.log(sim["g_true"]))
        c_eff = _corr(lg, torch.log(sim["g_effective"]))
        od = _overdispersion(sim, g_hat, b_hat)
        pred = 1.0 if m >= 1.0 else 1.0 + 2.0 * (1.0 - m) / (m * (k + 1.0))
        print(f"    {m:>6.2f} {k:>5.1f} | {c_true:>15.3f} {c_eff:>16.3f}"
              f" | {od:>9.2f} {pred:>10.2f} {1.0 / od:>7.2f}")

    print("\n   corr-vs-G_true falling while corr-vs-(G_i*m_i) holds = the per-image G is")
    print("   absorbing mean partiality, exactly as it should: the prior describes what")
    print("   is OBSERVED, and observed intensities ARE partial. So partiality does not")
    print("   break the per-image G -- it redefines what G means (scale x mean partiality).")
    print()
    print("   The residual damage is the SHAPE, and it has a one-parameter fix. Exp(tau)")
    print("   is Gamma(1, tau); Gamma(alpha, alpha*tau) has the SAME mean but")
    print("   Var/mean^2 = 1/alpha. So setting alpha = 1/over-dispersion (last column)")
    print("   restores the second moment, and:")
    print("     - it stays conjugate to the Poisson likelihood, so nothing downstream breaks;")
    print("     - alpha is already the crystallographic knob -- acentric reflections are")
    print("       alpha=1 (Exp) and centric are alpha=1/2 (Var/mean^2 = 2). Partiality just")
    print("       makes acentrics look progressively more 'centric' in their spread.")
    print("   A single learnable alpha is therefore the cheap, principled response to")
    print("   partiality, short of modelling p per reflection.")


STUDIES = {
    "case1": study_case1,
    "case2": study_case2,
    "ceiling": study_ceiling,
    "partiality": study_partiality,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma-logg", type=float, default=0.6)
    ap.add_argument("--b-global", type=float, default=20.0)
    ap.add_argument("--sigma-b", type=float, default=5.0)
    ap.add_argument("--bg", type=float, default=1.0)
    ap.add_argument("--g-scale", type=float, default=200.0)
    ap.add_argument("--sigma-partiality-image", type=float, default=0.12,
                    help="per-crystal spread in MEAN partiality (mosaicity/size variation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=sorted(STUDIES))
    args = ap.parse_args()

    print(f"Wilson per-image PRIOR study (integrator only; no scaling model)")
    print(f"tau = (1/G) exp(2 B s^2),  I ~ Exp(tau),  fit only via KL(q(I) || Exp(tau))")
    for key in ([args.only] if args.only else list(STUDIES)):
        STUDIES[key](args)


if __name__ == "__main__":
    main()
