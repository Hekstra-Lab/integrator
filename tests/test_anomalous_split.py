"""The Bijvoet pair as a Poisson split: a total and a Binomial, solved apart.

`update_merged_anomalous` solves `(J, Delta)` by a 2x2 Newton step whose
Hessian is rank-1 when a mate is missing; guarding its determinant the wrong
way once flipped the step, drove `J` to zero and corrupted every scale and
background in the sweep. `anomalous_split` replaces it with the exact
factorization Poisson thinning gives -- a Gamma posterior for the total and a
one-dimensional strictly concave Binomial for the split -- so there is no
determinant to guard and no rank deficiency to detect.

What these tests pin:

  - a planted fractional difference comes back, and comes back at least as well
    as the existing block gets it on the same data;
  - a one-sided pair keeps its intensity and returns the prior's difference,
    with no NaN and no collapse, which is the failure that cost a sweep;
  - the unknown really is the intensity fraction and not the observed split:
    with unequal exposures the two differ, and a correct block recovers the
    intensity fraction;
  - a centric gives an exact Binomial null rather than a heuristic spread.
"""

import math

import pytest
import torch

from integrator.fit.merge import anomalous_split, update_merged_anomalous

DT = torch.float64


def plant(
    n=4000, j_mean=800.0, frac=0.06, e_plus=40.0, e_minus=40.0, seed=0,
    beta_w=None, centric_frac=0.0,
):
    """Pairs with a known common intensity and fractional difference.

    `frac` is `(I+ - I-)/(I+ + I-)`, the quantity the prior belongs on.
    """
    g = torch.Generator().manual_seed(seed)
    j = torch.distributions.Exponential(1.0 / j_mean).sample((n,), ).to(DT) \
        if False else (-j_mean * torch.rand(n, generator=g, dtype=DT).clamp(1e-9, 1).log())
    sign = torch.where(
        torch.rand(n, generator=g, dtype=DT) < 0.5, -torch.ones(n, dtype=DT),
        torch.ones(n, dtype=DT),
    )
    f = torch.full((n,), float(frac), dtype=DT) * sign
    centric = torch.rand(n, generator=g, dtype=DT) < centric_frac
    f = torch.where(centric, torch.zeros_like(f), f)

    jp = j * (1.0 + f)
    jm = j * (1.0 - f)
    ep = torch.full((n,), float(e_plus), dtype=DT)
    em = torch.full((n,), float(e_minus), dtype=DT)
    kp = torch.poisson((jp * ep).float(), generator=g).to(DT)
    km = torch.poisson((jm * em).float(), generator=g).to(DT)
    bw = torch.full((n,), 1.0 / j_mean if beta_w is None else beta_w, dtype=DT)
    return dict(
        kp=kp, km=km, ep=ep, em=em, bw=bw, j=j, frac=f, centric=centric, n=n
    )


def run_split(d, tau=0.2):
    return anomalous_split(
        d["kp"], d["km"], d["ep"], d["em"], d["bw"],
        torch.full_like(d["kp"], float(tau)),
        centric=d["centric"] if d["centric"].any() else None,
    )


# ------------------------------------------------------------------ recovery


def test_the_planted_fractional_difference_comes_back():
    d = plant(frac=0.06, seed=1)
    out = run_split(d)
    r = float(torch.corrcoef(torch.stack([out["frac"], d["frac"]]))[0, 1])
    assert r > 0.6, f"corr(frac, truth) {r:.3f}"
    # and unbiased in magnitude, which an over-shrunk block would fail
    ratio = float(out["frac"].abs().mean() / d["frac"].abs().mean())
    assert 0.7 < ratio < 1.3, f"|frac| recovered at {ratio:.3f} of truth"


def test_the_common_intensity_comes_back():
    d = plant(frac=0.06, seed=2)
    out = run_split(d)
    rel = float(((out["j"] - d["j"]).abs() / d["j"]).median())
    assert rel < 0.08, f"median relative error on J {rel:.4f}"


def test_no_planted_difference_gives_none_back():
    d = plant(frac=0.0, seed=3)
    out = run_split(d, tau=0.2)
    assert float(out["frac"].abs().median()) < 0.03, (
        f"spurious fractional difference {float(out['frac'].abs().median()):.4f}"
    )


def test_it_is_at_least_as_good_as_the_newton_block():
    """Same planted data, same prior scale, compared on the fractional scale."""
    d = plant(frac=0.06, seed=4)
    new = run_split(d, tau=0.2)

    # the Newton block's tau is on the ABSOLUTE difference, so give it the
    # matching scale: sd(Delta) = 2 J tau_frac at the planted mean
    tau_abs = torch.full_like(d["kp"], 2.0 * float(d["j"].mean()) * 0.2)
    j_old, delta_old = update_merged_anomalous(
        d["kp"], d["km"], d["ep"], d["em"], d["bw"], tau_abs
    )
    frac_old = delta_old / (2.0 * j_old).clamp_min(1e-12)

    r_new = float(torch.corrcoef(torch.stack([new["frac"], d["frac"]]))[0, 1])
    r_old = float(torch.corrcoef(torch.stack([frac_old.to(DT), d["frac"]]))[0, 1])
    assert r_new >= r_old - 0.02, (
        f"split corr {r_new:.3f} against Newton {r_old:.3f}"
    )


# ------------------------------------------------------- the failure it fixes


def test_a_one_sided_pair_keeps_its_intensity_and_returns_the_prior():
    """No mate: the Hessian the Newton block forms is rank-1 here."""
    d = plant(n=800, frac=0.06, seed=5)
    d["em"] = torch.zeros_like(d["em"])
    d["km"] = torch.zeros_like(d["km"])
    out = run_split(d)

    assert torch.isfinite(out["j"]).all()
    assert torch.isfinite(out["frac"]).all()
    assert torch.isfinite(out["var_delta"]).all()
    # the data carry nothing about the split, so the prior returns rho = 1/2
    assert float(out["rho"].sub(0.5).abs().max()) < 1e-6
    assert float(out["frac"].abs().max()) < 1e-6
    # and the intensity is still the one-sided estimate, not zero
    expect = d["kp"] / d["ep"]
    rel = float(((out["j"] - expect).abs() / expect).median())
    assert rel < 0.10, f"one-sided J off by {rel:.3f}"


def test_nothing_goes_nan_at_zero_counts():
    d = plant(n=500, j_mean=800.0, seed=6)
    d["kp"] = torch.zeros_like(d["kp"])
    d["km"] = torch.zeros_like(d["km"])
    out = run_split(d)
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} is not finite"
    assert float(out["frac"].abs().max()) < 1e-6


# ------------------------------------------------- the offset does real work


def test_unequal_exposure_is_an_offset_not_a_difference():
    """`E+ != E-` changes the observed split but not the intensity fraction.

    This is the whole reason the unknown is `rho` and not `theta`: with twice
    the exposure on one mate the counts split 2:1 with no anomalous signal at
    all, and a block that read the split directly would report a 33% Bijvoet
    difference.
    """
    d = plant(n=4000, frac=0.0, e_plus=80.0, e_minus=40.0, seed=7)
    out = run_split(d, tau=0.2)

    # the raw split is far from even
    raw = float((d["kp"] / (d["kp"] + d["km"])).median())
    assert raw == pytest.approx(2.0 / 3.0, abs=0.02), f"raw split {raw:.3f}"
    # the recovered intensity fraction is not
    assert float(out["frac"].abs().median()) < 0.05, (
        f"exposure leaked into the difference: {float(out['frac'].abs().median()):.4f}"
    )
    assert float(out["theta_null"].median()) == pytest.approx(2 / 3, abs=1e-6)


def test_a_planted_difference_survives_unequal_exposure():
    d = plant(n=4000, frac=0.08, e_plus=70.0, e_minus=35.0, seed=8)
    out = run_split(d, tau=0.3)
    r = float(torch.corrcoef(torch.stack([out["frac"], d["frac"]]))[0, 1])
    assert r > 0.5, f"corr with unequal exposure {r:.3f}"
    ratio = float(out["frac"].abs().mean() / d["frac"].abs().mean())
    assert 0.6 < ratio < 1.4, f"|frac| at {ratio:.3f} of truth"


# ------------------------------------------------------------ exact anchor


def test_the_centric_null_is_exact_and_calibrated():
    """A centric must split at `E+/(E+ + E-)`, with a known Binomial null."""
    d = plant(n=6000, frac=0.0, e_plus=55.0, e_minus=40.0, seed=9)
    out = run_split(d)
    theta0 = out["theta_null"]
    t = out["total"]
    # standardized deviation of the observed split from the exact null
    z = (d["kp"] - t * theta0) / (t * theta0 * (1 - theta0)).clamp_min(1e-9).sqrt()
    sd = float(z.std())
    assert 0.85 < sd < 1.15, f"standardized spread {sd:.3f}, expected 1"


def test_a_mate_discriminating_scale_error_shows_in_the_null():
    """A 5% scale error between mates must break the exact null."""
    clean = plant(n=6000, frac=0.0, e_plus=50.0, e_minus=50.0, seed=10)
    dirty = plant(n=6000, frac=0.0, e_plus=50.0, e_minus=50.0, seed=10)
    # the counts were generated at the true exposure; the merge is told a wrong
    # one, which is what an unmodelled detector-position term does
    dirty["ep"] = dirty["ep"] * 1.05

    zs = []
    for d in (clean, dirty):
        out = run_split(d)
        t, th = out["total"], out["theta_null"]
        z = (d["kp"] - t * th) / (t * th * (1 - th)).clamp_min(1e-9).sqrt()
        zs.append(float(z.std()))
    assert zs[1] > 1.5 * zs[0], f"clean {zs[0]:.2f}, 5% mate error {zs[1]:.2f}"


def test_centrics_are_held_at_no_difference():
    d = plant(n=3000, frac=0.07, centric_frac=0.3, seed=11)
    out = run_split(d)
    c = d["centric"]
    assert float(out["frac"][c].abs().max()) == 0.0
    assert float(out["var_delta"][c].abs().max()) == 0.0
    assert float(out["frac"][~c].abs().median()) > 0.01


# ------------------------------------------------------------- the variance


def test_the_reported_variance_matches_the_scatter():
    d = plant(n=8000, frac=0.0, seed=12)
    out = run_split(d, tau=10.0)  # a wide prior, so the data set the spread
    z = out["frac"] / out["var_frac"].clamp_min(1e-18).sqrt()
    sd = float(z.std())
    assert 0.8 < sd < 1.25, f"standardized fractional difference sd {sd:.3f}"


def test_the_intensity_variance_is_the_gamma_posterior():
    d = plant(n=2000, frac=0.0, seed=13)
    out = run_split(d)
    # Gamma(alpha, beta) has mean alpha/beta and variance alpha/beta^2, so
    # var = mean / beta and the implied beta must be the exposure plus the
    # Wilson rate
    beta_implied = out["j"] / out["var_j"]
    expect = d["bw"] + 2.0 * (0.5 * d["ep"] + 0.5 * d["em"])
    assert torch.allclose(beta_implied, expect, rtol=1e-6)


def test_more_newton_steps_do_not_move_the_answer():
    """Strict concavity in one dimension: it is converged well before 12."""
    d = plant(n=2000, frac=0.06, seed=14)
    a = anomalous_split(
        d["kp"], d["km"], d["ep"], d["em"], d["bw"],
        torch.full_like(d["kp"], 0.2), n_iter=6,
    )
    b = anomalous_split(
        d["kp"], d["km"], d["ep"], d["em"], d["bw"],
        torch.full_like(d["kp"], 0.2), n_iter=60,
    )
    assert torch.allclose(a["frac"], b["frac"], atol=1e-9)
    assert torch.allclose(a["j"], b["j"], rtol=1e-9)


def test_the_fractional_scale_is_what_tau_controls():
    """Tightening tau shrinks the fractional difference, not the intensity.

    In the WEAK regime. The prior competes with the Binomial information
    `T theta (1 - theta)`, so on a bright pair it barely acts: at `J = 800` and
    `E = 40` the total is 64,000 counts, the information is about 16,000, and
    even `tau = 0.01` moves the difference by 18%. That is correct behaviour --
    the data have measured the split to a fractional sd of 0.004 and the prior
    should not override them -- so the shrinkage has to be tested where it is
    supposed to bite.
    """
    d = plant(n=4000, j_mean=4.0, frac=0.05, e_plus=8.0, e_minus=8.0, seed=15)
    wide = run_split(d, tau=1.0)
    tight = run_split(d, tau=0.01)
    assert float(tight["frac"].abs().median()) < 0.4 * float(
        wide["frac"].abs().median()
    )
    rel = float(((tight["j"] - wide["j"]).abs() / wide["j"]).median())
    assert rel < 0.02, f"tau moved the intensity by {rel:.4f}"


def test_the_prior_does_not_override_a_well_measured_split():
    """The converse, which is why the test above had to change regime."""
    d = plant(n=2000, j_mean=800.0, frac=0.05, e_plus=40.0, e_minus=40.0, seed=17)
    wide = run_split(d, tau=1.0)
    tight = run_split(d, tau=0.01)
    # 64,000 counts per pair measure the fraction to sd ~0.004; a prior of 0.01
    # is entitled to move it a little and no more
    ratio = float(
        tight["frac"].abs().median() / wide["frac"].abs().median()
    )
    assert ratio > 0.6, f"a tight prior overrode a 15-sigma measurement: {ratio:.3f}"


def test_math_identity_two_rho_minus_one_is_the_bijvoet_fraction():
    d = plant(n=200, frac=0.09, seed=16)
    out = run_split(d, tau=1.0)
    jp = 2.0 * out["j"] * out["rho"]
    jm = 2.0 * out["j"] * (1.0 - out["rho"])
    lhs = (jp - jm) / (jp + jm)
    assert torch.allclose(lhs, out["frac"], atol=1e-9)
    assert math.isclose(float((jp - jm - out["delta"]).abs().max()), 0.0, abs_tol=1e-6)
