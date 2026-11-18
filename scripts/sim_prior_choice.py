"""Which prior recovers I_h best when the Wilson parameters are NOT known?

`sim_joint_vs_staged.py` showed that learning the Wilson prior costs real
accuracy: `joint mean`, which refits (G, B) inside the loop, loses to
`staged + FW` at every count level. That leaves the obvious question this script
answers -- is there a prior that needs no fitting at all and still matches the
oracle?

Everything here holds the estimator fixed at the reflection-level conjugate
update and varies ONLY the prior, so the comparison is about the prior and
nothing else:

    I_h  ~  Gamma(alpha, tau_h)                     the prior
    I_h | data  ~  Gamma(alpha + A_h, tau_h + E_h)  the posterior

with `A_h = sum_i sum_p r c` the responsibility-weighted photons and
`E_h = sum_i s_i` the exposure. Choosing a prior is choosing two numbers, so the
whole question is a two-column table:

    arm              alpha    tau_h                        fitted params
    oracle           alpha_W  exp(2 B s^2)/G, true G,B     0 (cheats)
    plug-in Wilson   alpha_W  exp(2 B s^2)/G, fitted       2
    Wilson shape     alpha_W  0                            0
    Jeffreys         1/2      0                            0
    flat             1        0                            0
    ML (Haldane)     0        0                            0
    hierarchical G   alpha_W  G integrated out, B fitted   1 + uncertainty
    Clevenson-Zidek  --       --                           0 (not a prior)

THE CLAIM UNDER TEST, which is a statement about identifiability rather than
about statistics: the Wilson scale is a GAUGE of the joint likelihood, so the
oracle's advantage over a scale-free prior can only be as large as the part of
the Wilson curve the scale model cannot already express.

The pixel likelihood is invariant under (I_h, s_i) -> (c I_h, s_i / c); the only
term that breaks it is the prior rate tau_h, and under the collapsed objective
the change is exactly -alpha_W log c. So G is pure convention. B is convention
too AS SOON AS the scale model can represent a smooth function of resolution,
because then exp(-2 B s^2) is absorbable into s -- and careless's scale does
take d as an input (7.7% of its log-scale variance), while our own pre-test B
found the model's d-trend IS a B-gauge at R^2 = 0.98.

That is why the gauge group is a scoring choice here, not an estimator choice.
Every arm is scored after projecting out, in log space, exactly what the scale
model could have absorbed:

    --gauge const    scale = per-image scalar.          Remove 1 parameter.
                     B is identified, so the oracle should genuinely win.
    --gauge res      scale = per-image scalar x f(s^2).  Remove 2 parameters.
                     B is a gauge direction, so the scale-free priors should tie
                     the oracle.

Projecting at scoring time is generous to every arm equally and keeps a scale
fit of my own from becoming the thing that fails. It is not identical to the
estimator having fitted the resolution term, because a wrong one also distorts
the E-step through r = s I prf / (s I prf + b); at a 96.6% background share that
is second order, but it is the one approximation in the setup.

Resolution terciles are fixed from the simulated s^2 before any estimator runs,
so no arm selects its own range.

Usage:
    uv run python scripts/sim_prior_choice.py
    uv run python scripts/sim_prior_choice.py --gauge res
    uv run python scripts/sim_prior_choice.py --hkls 400 --seeds 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim_joint_vs_staged import (  # noqa: E402
    ALPHA_W,
    fit_scales_staged,
    fit_wilson,
    simulate,
    staged,
    tau_of,
)


# ------------------------------------------------------- the common estimator
def e_step_photons(d, s_hat, i_h):
    """Responsibility-weighted photons per reflection, `A_h`.

    `r = s I prf / (s I prf + b)` is the probability that a photon in that pixel
    came from the Bragg peak rather than the background, so `sum_p r c` is the
    peak's share of the observed counts. Never clip `c - b` at zero instead:
    that biased the first version of this simulator 19x high.
    """
    sig = (s_hat * i_h[d["hkl_of"]])[:, None] * d["prf"]
    r = sig / np.maximum(sig + d["bg"][:, None], 1e-12)
    return np.bincount(d["hkl_of"], weights=(r * d["counts"]).sum(1),
                       minlength=d["hkls"])


def conjugate(d, s_hat, *, alpha, tau, n_iter=40, refit=None):
    """Alternate the E-step with the Gamma posterior mean.

    Args:
        alpha: prior shape, held fixed.
        tau: prior rate, an array over reflections or a scalar.
        refit: optional `i_h -> tau` callable, for the arms that learn the prior.
    """
    e_h = np.bincount(d["hkl_of"], weights=s_hat, minlength=d["hkls"])
    i_h = np.maximum(staged(d, s_hat)[0], 1e-9)
    tau_h = np.broadcast_to(np.asarray(tau, float), (d["hkls"],)).copy()
    for _ in range(n_iter):
        a_h = e_step_photons(d, s_hat, i_h)
        if refit is not None:
            tau_h = refit(i_h)
        i_h = np.maximum((alpha + a_h) / np.maximum(tau_h + e_h, 1e-12), 1e-12)
    return i_h


def hierarchical_g(d, s_hat, n_iter=12, n_grid=160):
    """Integrate G out instead of plugging it in, with B from the Wilson plot.

    The textbook diagnosis of "learning the prior hurts" is that plug-in
    empirical Bayes ignores the hyperparameter's own uncertainty, paying a risk
    penalty of order 1/H_bin. The fix is to marginalize, not to estimate better.

    `A_h | G ~ NegBinomial(alpha_W, E_h / (E_h + tau_h(G)))` because the Gamma
    prior is conjugate to the Poisson count, so the marginal posterior of G is a
    one-dimensional quadrature with the scale-invariant hyperprior
    `pi(G) ~ 1/G`, and the intensity estimate averages the conditional posterior
    mean over it.
    """
    e_h = np.bincount(d["hkl_of"], weights=s_hat, minlength=d["hkls"])
    i_h = np.maximum(staged(d, s_hat)[0], 1e-9)
    for _ in range(n_iter):
        a_h = e_step_photons(d, s_hat, i_h)
        _, b = fit_wilson(i_h, d["s2"])
        scale = np.exp(2.0 * b * d["s2"])          # tau_h = scale / G

        g0 = max(float(np.mean(np.maximum(i_h, 1e-12) * scale)), 1e-9)
        log_g = np.linspace(np.log(g0) - 4.0, np.log(g0) + 4.0, n_grid)
        tau = scale[None, :] / np.exp(log_g)[:, None]        # (grid, hkls)

        frac = e_h[None, :] / (tau + e_h[None, :])
        logp = (ALPHA_W * np.log1p(-frac) + a_h[None, :] * np.log(frac)).sum(1)
        logp = logp - log_g                                  # pi(G) ~ 1/G
        w = np.exp(logp - logp.max())
        w = w / w.sum()

        post = (ALPHA_W + a_h[None, :]) / (tau + e_h[None, :])
        i_h = np.maximum((w[:, None] * post).sum(0), 1e-12)
    return i_h


def clevenson_zidek(d, s_hat, a_h):
    """Hyperparameter-free shrinkage that provably dominates the MLE.

    Clevenson and Zidek (1975): for `X_h ~ Poisson(lambda_h)` under the loss
    `sum_h (delta_h - lambda_h)^2 / lambda_h`, the estimator

        delta_h = (1 - (H - 1) / (H - 1 + sum_h X_h)) X_h

    beats `X_h` for EVERY true lambda once `H >= 2`. It is the Poisson
    James-Stein: the shrinkage factor reads only the grand total, so there is
    nothing to fit and nothing to get wrong. Included because if any
    hyperparameter-free rule can be shown to beat the no-prior estimator, this
    is the one with the theorem attached.

    The derivation assumes a common exposure, which holds here only because
    every reflection appears on every image; `E_h` is checked rather than
    assumed.
    """
    e_h = np.bincount(d["hkl_of"], weights=s_hat, minlength=d["hkls"])
    if e_h.std() > 1e-9 * max(e_h.mean(), 1e-12):
        raise ValueError(
            "Clevenson-Zidek needs a common exposure; E_h varies by "
            f"{e_h.std() / e_h.mean():.3g} relative"
        )
    h = len(a_h)
    z = float(a_h.sum())
    shrink = 1.0 - (h - 1.0) / (h - 1.0 + z)
    return np.maximum(shrink * a_h / np.maximum(e_h, 1e-12), 1e-12)


# ------------------------------------------------------------------- scoring
def project_gauge(log_ratio, s2, mode):
    """Remove whatever the scale model could have absorbed, in log space.

    `const` removes a single multiplicative constant, the gauge of a per-image
    scalar scale. `res` also removes a slope in `s^2`, the gauge of a scale that
    sees resolution -- which is the one that makes the Wilson B unidentifiable.
    """
    if mode == "const":
        return log_ratio - log_ratio.mean()
    if mode == "res":
        design = np.stack([np.ones_like(s2), s2], 1)
        coef, *_ = np.linalg.lstsq(design, log_ratio, rcond=None)
        return log_ratio - design @ coef
    raise ValueError(f"unknown gauge {mode!r}")


def score(i_hat, d, gauge, terciles):
    """rms of the gauge-projected log error, overall and per resolution tercile."""
    ok = (i_hat > 0) & (d["i_true"] > 0)
    lr = np.full(len(i_hat), np.nan)
    lr[ok] = np.log(i_hat[ok]) - np.log(d["i_true"][ok])
    lr[ok] = project_gauge(lr[ok], d["s2"][ok], gauge)
    out = {"rms_all": float(np.nanstd(lr)), "bias": float(np.nanmean(lr))}
    for k in (0, 1, 2):
        sel = terciles == k
        out[f"rms_t{k}"] = float(np.nanstd(lr[sel]))
    return out


ARMS = (
    "oracle",
    "plug-in Wilson",
    "Wilson shape",
    "Jeffreys",
    "flat",
    "ML (Haldane)",
    "hierarchical G",
    "Clevenson-Zidek",
)


def run_arms(d, s_hat):
    """Every arm on one simulated dataset, as a name -> I_hat mapping."""
    out = {}
    out["oracle"] = conjugate(
        d, s_hat, alpha=ALPHA_W,
        tau=tau_of(d["g_true"], d["b_true"], d["s2"]),
    )
    out["plug-in Wilson"] = conjugate(
        d, s_hat, alpha=ALPHA_W, tau=0.0,
        refit=lambda i: tau_of(*fit_wilson(i, d["s2"]), d["s2"]),
    )
    out["Wilson shape"] = conjugate(d, s_hat, alpha=ALPHA_W, tau=0.0)
    out["Jeffreys"] = conjugate(d, s_hat, alpha=0.5, tau=0.0)
    out["flat"] = conjugate(d, s_hat, alpha=1.0, tau=0.0)
    out["ML (Haldane)"] = conjugate(d, s_hat, alpha=0.0, tau=0.0)
    out["hierarchical G"] = hierarchical_g(d, s_hat)

    # The E-step needs SOME prior to keep its fixed point off zero, so the
    # Clevenson-Zidek arm reads the responsibilities of the Jeffreys run and
    # shrinks only the final point estimate. The shrinkage itself stays
    # hyperparameter-free; the E-step is shared, not tuned.
    a_h = e_step_photons(d, s_hat, out["Jeffreys"])
    out["Clevenson-Zidek"] = clevenson_zidek(d, s_hat, a_h)
    return out


def main():
    p = argparse.ArgumentParser(
        description="which prior recovers I_h when (G, B) are unknown"
    )
    p.add_argument("--images", type=int, default=10)
    p.add_argument("--hkls", type=int, default=50)
    p.add_argument("--nx", type=int, default=15)
    p.add_argument("--ny", type=int, default=15)
    p.add_argument("--photons", type=float, default=None)
    p.add_argument("--b-true", type=float, default=20.0, help="Wilson B, A^2")
    p.add_argument("--bg-share", type=float, default=0.966)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--gauge", choices=("const", "res"), default="const",
                   help="what the scale model can absorb; see the module docstring")
    args = p.parse_args()

    ladder = [args.photons] if args.photons else [10.0, 30.0, 100.0, 1000.0]
    print(f"{args.images} images x {args.hkls} hkls, {args.nx}x{args.ny} px, "
          f"background share {args.bg_share}, Wilson B {args.b_true} A^2, "
          f"{args.seeds} seeds")
    print(f"gauge projected out at scoring: {args.gauge} "
          f"({'constant only' if args.gauge == 'const' else 'constant + s^2 slope'})")
    print("rms = rms of the projected log(I_hat/I_true); t0 low res -> t2 high "
          "res, terciles fixed from s^2.\n")
    hdr = (f"{'photons':>7}  {'prior':<17} {'rms_all':>8} {'rms_t0':>7} "
           f"{'rms_t1':>7} {'rms_t2':>7} {'bias':>7}  vs oracle (paired)")
    print(hdr)
    print("-" * len(hdr))

    for photons in ladder:
        acc: dict[str, list] = {}
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            d = simulate(rng, images=args.images, hkls=args.hkls, nx=args.nx,
                         ny=args.ny, photons=photons, bg_share=args.bg_share,
                         b_true=args.b_true)
            cuts = np.quantile(d["s2"], [1 / 3, 2 / 3])
            terciles = np.searchsorted(cuts, d["s2"])
            s_hat = fit_scales_staged(d)
            for name, i_hat in run_arms(d, s_hat).items():
                acc.setdefault(name, []).append(
                    score(i_hat, d, args.gauge, terciles)
                )

        # Paired by seed against the oracle: the seed-to-seed spread of rms is
        # several times the differences between arms, so the unpaired columns
        # cannot resolve them. The same simulated data goes to every arm, so the
        # per-seed difference is the statistic with the noise removed.
        base = np.array([x["rms_all"] for x in acc["oracle"]])
        for name in ARMS:
            es = acc[name]
            a = {k: float(np.nanmean([x[k] for x in es])) for k in es[0]}
            diff = np.array([x["rms_all"] for x in es]) - base
            sem = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
            tag = "" if name == "oracle" else f"{diff.mean():+.4f} +/- {sem:.4f}"
            print(f"{photons:>7.0f}  {name:<17} {a['rms_all']:>8.4f} "
                  f"{a['rms_t0']:>7.4f} {a['rms_t1']:>7.4f} "
                  f"{a['rms_t2']:>7.4f} {a['bias']:>7.3f}  {tag}")
        print()


if __name__ == "__main__":
    main()
