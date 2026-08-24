"""Per-image scale factors for SFX MERGING: what identifies G_i, and how to amortize it.

SCOPE: this is the SCALING/MERGING problem -- G_i multiplies a shared per-HKL intensity
I_h across images, so G and I are a bilinear pair and identification runs through HKL
redundancy. That is NOT the integrator's Wilson prior. For per-image parameters of the
integrator's intensity PRIOR (tau = (1/G) exp(2 B s^2), fit only by KL(q(I) || Exp(tau)),
no merging anywhere), see `wilson_per_image_prior.py` -- a different problem with a
different answer and no gauge freedom.


In SFX every image is an independent crystal in a random orientation hit by a pulse of
varying intensity, so each image carries its own scale G_i. Unlike rotation data there is
no smooth G(phi) to fit -- image i and image i+1 are unrelated, so a spline/Chebyshev has
nothing to interpolate along.

The generative model, per observation (image i, reflection h):

    C_ih ~ Poisson(G_i * I_h + bg)

with I_h ~ Exp(mean) over `n_hkl` unique reflections and log G_i ~ Normal(0, sigma_logG)
(gauge-fixed to geometric mean 1). Each image observes `n_refl_per_image` distinct HKLs
drawn at random, so the redundancy

    multiplicity = n_images * n_refl_per_image / n_hkl

is a knob. This matters more than anything else here: **G_i and I_h are only jointly
identified up to a global scale, and G_i is identified relative to other images ONLY
through HKLs they share.** With a strict partition (`disjoint=True`) nothing is shared,
G_i * I_h is a bare product of unknowns, and recovery collapses to corr 0.000 -- measured,
not asserted. But sharing turns out to be cheap: random assignment at MEAN multiplicity 1
already puts 63% of observations on a repeated HKL, and a scalar G_i needs only a few
linked reflections, so G is recovered at corr 0.997 across the whole realistic range.
What keeps improving with multiplicity is the merged INTENSITY, not the scale.

Three questions:

  1. multiplicity  How much redundancy does per-image G actually need?
  2. confound      The natural "learn G from global image stats" idea uses the image's
                   total counts. That is confounded: total counts depends on WHICH HKLs
                   the image happened to observe, not only on G. This measures the
                   confound and shows the fix.
  3. shrinkage     Images with few reflections have noisy G. A hierarchical prior pools
                   them toward the population -- exchangeability replacing smoothness as
                   the regularizer.

The punchline for amortization: given I and the profile, the sufficient statistic for
G_i is TWO numbers -- total observed counts and total expected counts at G=1 -- and the
Poisson-Gamma posterior is closed-form,

    G_i | . ~ Gamma(a + sum_h C_ih,  b + sum_h I_h),

so G is *solved*, not learned. Any network taking more than that ratio is spending
capacity to re-derive arithmetic. See README_SCALING.md.

Run:  uv run python scripts/jungfrau_sim/sfx_per_image_scale.py [--only confound]
"""

from __future__ import annotations

import argparse

import torch

torch.set_default_dtype(torch.float64)


def simulate(
    n_images: int,
    n_hkl: int,
    n_refl_per_image: int,
    sigma_logG: float,
    i_mean: float,
    bg: float,
    seed: int,
    disjoint: bool = False,
) -> dict:
    """Draw per-image scales, per-HKL intensities, and the observations linking them."""
    g = torch.Generator().manual_seed(seed)

    u = torch.rand(n_hkl, generator=g).clamp_min(1e-12)
    i_true = -i_mean * torch.log(u)  # Wilson: I ~ Exp(mean)

    log_g = torch.randn(n_images, generator=g) * sigma_logG
    log_g = log_g - log_g.mean()  # gauge: geometric mean 1
    g_true = log_g.exp()

    if disjoint:
        # Control: partition the HKLs so NO reflection is ever shared between images.
        # This is the true zero-redundancy case, and G must be unidentifiable here.
        if n_hkl < n_images * n_refl_per_image:
            raise ValueError("disjoint needs n_hkl >= n_images * n_refl_per_image")
        hkl_idx = torch.arange(n_images * n_refl_per_image)
    else:
        # Each image observes n_refl_per_image DISTINCT hkls, drawn at random. Note the
        # per-HKL observation count is then ~Poisson(mult), so even at MEAN multiplicity
        # 1 a fraction 1 - e^-1 = 63% of OBSERVATIONS land on a repeated HKL. Sharing is
        # much easier to come by than the mean multiplicity suggests.
        order = torch.rand(n_images, n_hkl, generator=g).argsort(dim=1)
        hkl_idx = order[:, :n_refl_per_image].reshape(-1)
    image_idx = (
        torch.arange(n_images).unsqueeze(1).expand(-1, n_refl_per_image).reshape(-1)
    )

    rate = g_true[image_idx] * i_true[hkl_idx] + bg
    counts = torch.poisson(rate, generator=g)

    return {
        "counts": counts,
        "image_idx": image_idx,
        "hkl_idx": hkl_idx,
        "g_true": g_true,
        "i_true": i_true,
        "n_images": n_images,
        "n_hkl": n_hkl,
        "bg": bg,
        "multiplicity": 0.0 if disjoint else n_images * n_refl_per_image / n_hkl,
        "frac_shared": float(
            (torch.bincount(hkl_idx, minlength=n_hkl)[hkl_idx] > 1).double().mean()
        ),
    }


def _oracle_a(sim: dict) -> float:
    """Gamma(a, a) prior strength matched to the TRUE spread of G: a = 1/Var(G)."""
    return float(1.0 / sim["g_true"].var().clamp_min(1e-6))


def _gauge(g: torch.Tensor) -> torch.Tensor:
    """Fix the G<->I scale degeneracy by pinning the geometric mean of G to 1."""
    return g / torch.exp(torch.log(g.clamp_min(1e-12)).mean())


def estimate_naive_total(sim: dict) -> torch.Tensor:
    """G_i proportional to the image's total counts -- 'learn it from global image stats'.

    This is the idea in its rawest form, and it is CONFOUNDED: the total is
    G_i * sum_{h in image i} I_h, so an image that happened to catch a few strong
    low-resolution reflections looks bright regardless of its actual scale. The confound
    averages out only as n_refl_per_image grows.
    """
    total = torch.bincount(
        sim["image_idx"], weights=sim["counts"], minlength=sim["n_images"]
    )
    return _gauge(total.clamp_min(1e-12))


def estimate_ratio(sim: dict, i_hat: torch.Tensor) -> torch.Tensor:
    """G_i = (total observed) / (total expected at G=1) -- the sufficient statistic.

    Identical arithmetic to the naive estimator except for dividing by what the image was
    EXPECTED to produce given which HKLs it observed. That single division is the whole
    fix, and it is also exactly the Poisson MLE for G given I.
    """
    obs = torch.bincount(
        sim["image_idx"], weights=sim["counts"], minlength=sim["n_images"]
    )
    exp = torch.bincount(
        sim["image_idx"], weights=i_hat[sim["hkl_idx"]], minlength=sim["n_images"]
    )
    return _gauge((obs / exp.clamp_min(1e-12)).clamp_min(1e-12))


def alternating(
    sim: dict,
    n_iter: int = 60,
    prior_a: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternate the closed-form Poisson updates for I and G (this is EM / CAVI).

    I_h = sum_i C_ih / sum_i G_i        (over images observing h)
    G_i = sum_h C_ih / sum_h I_h        (over reflections in image i)

    With `prior_a` set, the G step becomes the mean of the conjugate Gamma posterior
    Gamma(a + sum C, a + sum I) under a Gamma(a, a) prior -- mean 1 (the gauge), variance
    1/a. That pools sparse images toward the population: exchangeability doing the job
    smoothness does for rotation data. Passing the true 1/Var(G) makes it an ORACLE
    prior, which upper-bounds what pooling can buy. (Estimating `a` by empirical Bayes is
    a separate and genuinely fragile problem at low counts -- the naive
    moment-matching estimator collapses to total shrinkage -- so it is deliberately not
    conflated with the question of whether pooling helps at all.)
    """
    image_idx, hkl_idx = sim["image_idx"], sim["hkl_idx"]
    n_images, n_hkl = sim["n_images"], sim["n_hkl"]
    # Background is known here; subtract its contribution so the conjugacy is exact.
    counts = (sim["counts"] - sim["bg"]).clamp_min(0.0)

    obs_i = torch.bincount(image_idx, weights=counts, minlength=n_images)
    obs_h = torch.bincount(hkl_idx, weights=counts, minlength=n_hkl)

    g_hat = torch.ones(n_images)
    i_hat = torch.zeros(n_hkl)
    for _ in range(n_iter):
        den_h = torch.bincount(
            hkl_idx, weights=g_hat[image_idx], minlength=n_hkl
        )
        i_hat = obs_h / den_h.clamp_min(1e-12)

        den_i = torch.bincount(
            image_idx, weights=i_hat[hkl_idx], minlength=n_images
        )
        if prior_a is None:
            g_hat = obs_i / den_i.clamp_min(1e-12)
        else:
            g_hat = (prior_a + obs_i) / (prior_a + den_i).clamp_min(1e-12)
        g_hat = _gauge(g_hat.clamp_min(1e-12))

    return g_hat, i_hat


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


def score(g_hat: torch.Tensor, sim: dict) -> dict:
    """Accuracy of the recovered scales, and of the merged intensities they imply."""
    lg_hat = torch.log(_gauge(g_hat).clamp_min(1e-12))
    lg_true = torch.log(sim["g_true"])
    corr = _corr(lg_hat, lg_true)

    # Merged intensity under these scales, vs the truth.
    counts = (sim["counts"] - sim["bg"]).clamp_min(0.0)
    obs_h = torch.bincount(sim["hkl_idx"], weights=counts, minlength=sim["n_hkl"])
    den_h = torch.bincount(
        sim["hkl_idx"], weights=_gauge(g_hat)[sim["image_idx"]], minlength=sim["n_hkl"]
    )
    i_hat = obs_h / den_h.clamp_min(1e-12)
    seen = den_h > 0
    ih, it = i_hat[seen], sim["i_true"][seen]
    # Normalized RMSE, not the mean of per-HKL RELATIVE errors: on weak data many HKLs
    # have I_true near zero and I_hat near zero, and a ratio there is numerically wild
    # while carrying no information. Dividing by the population mean is stable.
    nrmse = 100 * float((ih - it).pow(2).mean().sqrt() / it.mean().clamp_min(1e-9))
    corr_i = _corr(ih, it)
    return {
        "corr_logG": corr,
        "rmse_logG": float((lg_hat - lg_true).pow(2).mean().sqrt()),
        "I_nrmse%": nrmse,
        "corr_I": corr_i,
    }


def _row(name: str, m: dict) -> str:
    return (
        f"    {name:<20} corr(logG) {m['corr_logG']:6.3f}   "
        f"rmse(logG) {m['rmse_logG']:6.3f}   merged I nrmse {m['I_nrmse%']:6.1f}%"
    )


def study_multiplicity(args) -> None:
    """Q1: how much HKL redundancy does per-image G need to become identifiable?"""
    print("\n" + "=" * 82)
    print("1. MULTIPLICITY -- per-image G is identified only by SHARED reflections")
    print("=" * 82)
    print("   n_images=400, n_refl_per_image=40 fixed; n_hkl shrinks -> multiplicity grows.")
    print("   `disjoint` is the control: HKLs PARTITIONED so nothing is ever shared.")
    print("   %shared = fraction of observations whose HKL is seen more than once.\n")

    print(f"    {'n_hkl':>8} {'mult':>8} {'%shared':>9} | {'corr(logG)':>11} {'merged I nrmse':>14}")
    print("    " + "-" * 56)

    sim = simulate(400, 16000, 40, args.sigma_logg, 200.0, args.bg, args.seed, disjoint=True)
    m = score(alternating(sim, prior_a=_oracle_a(sim))[0], sim)
    print(f"    {'disjoint':>8} {'0.0x':>8} {100 * sim['frac_shared']:>8.1f}%"
          f" | {m['corr_logG']:>11.3f} {m['I_nrmse%']:>12.1f}%   <- control")

    for n_hkl in [16000, 4000, 1600, 800, 400, 160]:
        sim = simulate(400, n_hkl, 40, args.sigma_logg, 200.0, args.bg, args.seed)
        m = score(alternating(sim, prior_a=_oracle_a(sim))[0], sim)
        print(f"    {n_hkl:>8} {sim['multiplicity']:>7.1f}x {100 * sim['frac_shared']:>8.1f}%"
              f" | {m['corr_logG']:>11.3f} {m['I_nrmse%']:>12.1f}%")

    print("\n   With NO sharing G is unrecoverable -- exactly the degeneracy G_i*I_h.")
    print("   But sharing arrives fast: random assignment at MEAN multiplicity 1 still")
    print("   repeats 63% of observations (1 - 1/e), and a scalar G_i needs only a few")
    print("   linked reflections. So redundancy is cheap for G; what keeps improving")
    print("   with multiplicity is the MERGED INTENSITY, not the scale.")


def study_confound(args) -> None:
    """Q2: does 'learn G from global image stats' work? Partly -- and here is the fix."""
    print("\n" + "=" * 82)
    print("2. CONFOUND -- image total counts vs the sufficient statistic")
    print("=" * 82)
    print("   naive_total   G_i ~ sum of the image's counts        (the raw 'image stats' idea)")
    print("   ratio_1pass   G_i = observed / EXPECTED, one pass    (divide by image content)")
    print("   alternating   the same ratio, iterated to convergence")
    print("   The naive estimator is confounded by WHICH hkls an image caught; that")
    print("   averages out only as reflections-per-image grows.\n")

    for n_refl in [5, 10, 25, 50, 100, 200]:
        sim = simulate(400, 800, n_refl, args.sigma_logg, 200.0, args.bg, args.seed)
        print(f"  n_refl_per_image = {n_refl:>3}   (multiplicity {sim['multiplicity']:.1f}x)")

        print(_row("naive_total", score(estimate_naive_total(sim), sim)))

        # One pass: seed I by averaging counts per hkl with no scaling at all.
        counts = (sim["counts"] - sim["bg"]).clamp_min(0.0)
        obs_h = torch.bincount(sim["hkl_idx"], weights=counts, minlength=sim["n_hkl"])
        n_h = torch.bincount(sim["hkl_idx"], minlength=sim["n_hkl"]).clamp_min(1)
        print(_row("ratio_1pass", score(estimate_ratio(sim, obs_h / n_h), sim)))

        g_hat, _ = alternating(sim, prior_a=_oracle_a(sim))
        print(_row("alternating", score(g_hat, sim)))
        print()


def study_shrinkage(args) -> None:
    """Q3: what does pooling buy when images have few reflections?"""
    print("\n" + "=" * 82)
    print("3. SHRINKAGE -- exchangeability replaces smoothness as the regularizer")
    print("=" * 82)
    print("   SFX has no smooth G(phi) to borrow strength along, but images ARE")
    print("   exchangeable, so a hierarchical prior pools them. What decides whether")
    print("   that helps is TOTAL COUNTS PER IMAGE -- the precision of G_i is ~1/sqrt(N).")
    print("   Sweeping mean reflection intensity at a fixed 5 reflections/image:\n")

    print(f"    {'I_mean':>8} {'ph/image':>9} | {'MLE corr':>9} {'oracle-prior':>12}"
          f" | {'MLE I nrmse':>12} {'oracle I nrmse':>15}")
    print("    " + "-" * 75)
    for i_mean in [200.0, 20.0, 5.0, 1.0, 0.3]:
        sim = simulate(400, 800, 5, args.sigma_logg, i_mean, args.bg, args.seed)
        mle = score(alternating(sim)[0], sim)
        shr = score(alternating(sim, prior_a=_oracle_a(sim))[0], sim)
        per_image = float(sim["counts"].sum() / sim["n_images"])
        print(f"    {i_mean:>8.1f} {per_image:>9.1f} | {mle['corr_logG']:>9.3f}"
              f" {shr['corr_logG']:>12.3f} | {mle['I_nrmse%']:>9.1f}%"
              f" {shr['I_nrmse%']:>12.1f}%")

    print("\n   Shrinkage is inert on bright data -- even 5 reflections x 200 photons")
    print("   pin G to ~1/sqrt(1000) = 3%, so the likelihood swamps any prior. It only")
    print("   pays once an image holds ~tens of photons total, i.e. the weak-data limit.")
    print("   Budget the complexity accordingly: pooling is insurance for sparse images,")
    print("   not a general-purpose win.")


STUDIES = {
    "multiplicity": study_multiplicity,
    "confound": study_confound,
    "shrinkage": study_shrinkage,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigma-logg", type=float, default=0.6,
                    help="spread of log G across images (0.6 ~ a 2x spread, realistic)")
    ap.add_argument("--bg", type=float, default=0.0,
                    help="flat background per observation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=sorted(STUDIES))
    args = ap.parse_args()

    print(f"SFX per-image scale study  (sigma_logG={args.sigma_logg}, seed={args.seed})")
    for key in ([args.only] if args.only else list(STUDIES)):
        STUDIES[key](args)


if __name__ == "__main__":
    main()
