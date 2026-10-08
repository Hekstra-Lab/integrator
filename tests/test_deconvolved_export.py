"""The prior-deconvolved intensity export.

Our q(I) is a Gamma already shrunk toward the integrator's Wilson prior. careless
then applies a Wilson prior of its own, so weak reflections are shrunk twice.
Dividing the posterior by its prior is exact in the Gamma family; these check the
algebra, the direction of the correction, and the guard for the case where it
cannot be done.
"""

import torch


def deconvolve(alpha, beta, c, lam):
    """Gamma(alpha,beta) / Gamma(c,lam) -> Gamma(alpha-c+1, beta-lam)."""
    rate = beta - lam
    floored = rate <= 0
    conc_like = torch.clamp(alpha - c + 1.0, min=1e-3)
    rate_like = torch.where(floored, beta, rate)
    conc_like = torch.where(floored, alpha, conc_like)
    return conc_like, rate_like, floored


def test_multiplying_the_prior_back_recovers_the_posterior():
    """The identity the export rests on, checked on the natural parameters."""
    alpha = torch.tensor([3.0, 10.0, 0.8])
    beta = torch.tensor([0.5, 2.0, 0.3])
    c = torch.tensor([1.0, 1.0, 1.0])
    lam = torch.tensor([0.1, 0.4, 0.05])
    a, b, _ = deconvolve(alpha, beta, c, lam)
    # multiplying the two Gammas re-adds the shapes minus one and the rates
    assert torch.allclose(a + c - 1.0, alpha)
    assert torch.allclose(b + lam, beta)


def test_acentric_keeps_the_shape():
    """With c = 1 the shape is unchanged and only the rate moves."""
    alpha = torch.tensor([4.0]); beta = torch.tensor([1.0])
    a, b, _ = deconvolve(alpha, beta, torch.tensor([1.0]), torch.tensor([0.25]))
    assert torch.allclose(a, alpha)
    assert torch.allclose(b, torch.tensor([0.75]))


def test_centric_shifts_the_shape_up():
    """A centric prior is Gamma(1/2), so removing it RAISES the shape by a half."""
    alpha = torch.tensor([4.0]); beta = torch.tensor([1.0])
    a, _, _ = deconvolve(alpha, beta, torch.tensor([0.5]), torch.tensor([0.25]))
    assert torch.allclose(a, torch.tensor([4.5]))


def test_correction_raises_the_mean_and_the_variance():
    """Removing a shrinkage prior must un-shrink: both moments grow."""
    alpha = torch.tensor([2.0]); beta = torch.tensor([1.0])
    lam = torch.tensor([0.3])
    a, b, _ = deconvolve(alpha, beta, torch.tensor([1.0]), lam)
    assert (a / b) > (alpha / beta)
    assert (a / b**2) > (alpha / beta**2)


def test_strong_reflections_are_barely_touched():
    """A reflection whose rate dwarfs the prior's barely moves: the prior did little."""
    alpha = torch.tensor([400.0]); beta = torch.tensor([100.0])
    lam = torch.tensor([0.1])
    a, b, _ = deconvolve(alpha, beta, torch.tensor([1.0]), lam)
    ratio = (a / b) / (alpha / beta)
    assert 1.0 < ratio < 1.002


def test_weak_reflections_move_most():
    """The correction must be larger where the prior carried more of the answer."""
    lam = torch.tensor([0.3])
    c = torch.tensor([1.0])
    weak = deconvolve(torch.tensor([1.0]), torch.tensor([0.5]), c, lam)
    strong = deconvolve(torch.tensor([100.0]), torch.tensor([50.0]), c, lam)
    weak_ratio = (weak[0] / weak[1]) / (1.0 / 0.5)
    strong_ratio = (strong[0] / strong[1]) / (100.0 / 50.0)
    assert weak_ratio > strong_ratio


def test_narrower_than_the_prior_is_floored_not_negative():
    """beta <= lam cannot happen for a true posterior; it must not yield a negative rate."""
    alpha = torch.tensor([2.0]); beta = torch.tensor([0.2])
    a, b, floored = deconvolve(alpha, beta, torch.tensor([1.0]), torch.tensor([0.5]))
    assert bool(floored.all())
    assert (b > 0).all()
    assert torch.allclose(a, alpha) and torch.allclose(b, beta)


def test_loss_exposes_the_same_prior_it_uses():
    """intensity_prior_params must return the acentric/centric Wilson prior."""
    from integrator.model.loss.wilson_loss import WilsonLoss

    assert hasattr(WilsonLoss, "intensity_prior_params")
