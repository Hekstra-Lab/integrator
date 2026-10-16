"""The per-observation Wilson rate in the merging model.

Wilson's prior is a statement about a measurement, so on Laue data its rate
depends on the wavelength of each observation. A merged latent takes one
prior, so the rates have to be averaged -- and averaging the rates is not the
same as evaluating the rate once at the mean wavelength, because `1/G` does not
commute with a mean. The default path does the latter, which on hewl_1118 left
the prior anti-correlated with the beam.
"""

import math

import pytest
import torch

from integrator.model.loss.learned_spectrum import ChebyshevSpectrum


class _PolyTau:
    """The rate `PolychromaticWilsonLoss._get_tau` computes, standalone."""

    def __init__(self, c, B=20.0, lam=(0.96, 1.26)):
        self.spectrum = ChebyshevSpectrum(degree=len(c) - 1, lambda_min=lam[0],
                                          lambda_max=lam[1])
        with torch.no_grad():
            self.spectrum.c.copy_(torch.as_tensor(c))
        self.B = B

    def _get_tau(self, meta, s_sq, device):
        log_tau = 2.0 * self.B * s_sq - self.spectrum.get_log_G(meta["wavelength"])
        return torch.exp(log_tau.clamp(-30.0, 30.0))


def _scatter_mean(src, index, n):
    out = torch.zeros(n, dtype=src.dtype)
    out.scatter_add_(0, index, src)
    cnt = torch.zeros(n, dtype=src.dtype)
    cnt.scatter_add_(0, index, torch.ones_like(src))
    return out / cnt.clamp(min=1.0)


# the identified hewl_1118 spectrum, from the seven ablation arms
C_1118 = [6.871, 0.254, -1.923, 1.012, -0.333, -0.225]


def test_mean_of_rates_differs_from_rate_at_the_mean_wavelength():
    """The two orders of operations are not interchangeable, by a lot."""
    loss = _PolyTau(C_1118)
    lam = torch.tensor([0.98, 1.05, 1.22])          # one reflection, three shots
    d = torch.full_like(lam, 2.0)
    s_sq = 1.0 / (4.0 * d.pow(2))

    per_obs = loss._get_tau({"wavelength": lam}, s_sq, None)
    mean_of_rates = float(per_obs.mean())
    rate_at_mean = float(
        loss._get_tau({"wavelength": lam.mean().reshape(1)}, s_sq[:1], None)
    )
    assert mean_of_rates / rate_at_mean > 1.5, (mean_of_rates, rate_at_mean)


def test_the_rate_is_dominated_by_the_dimmest_observation():
    """A mean of rates is a harmonic mean of prior means, which is the point.

    `tau ~ 1/G`, so the average rate is set by the observation with the least
    beam behind it -- the conservative choice for a prior that has to cover
    every measurement of the reflection.
    """
    loss = _PolyTau(C_1118)
    lam = torch.tensor([1.05, 1.05, 1.24])
    s_sq = torch.full_like(lam, 1.0 / 16.0)
    tau = loss._get_tau({"wavelength": lam}, s_sq, None)
    assert tau[2] > 5 * tau[0]
    assert float(tau.mean()) > float(tau[:2].mean())


def test_a_spectral_scale_cancels_the_wavelength_out_of_the_rate():
    """The self-correcting property the construction rests on.

    The implied prior on `I_h` has rate `s_i tau_i`. When the scale carries the
    spectrum, `s_i` is proportional to `G(lambda_i)` and the wavelength drops
    out, leaving the lambda-free rate a crystal-level latent should have.
    """
    loss = _PolyTau(C_1118)
    lam = torch.linspace(0.98, 1.24, 12)
    s_sq = torch.full_like(lam, 1.0 / 16.0)
    tau = loss._get_tau({"wavelength": lam}, s_sq, None)

    with torch.no_grad():
        G = loss.spectrum.get_log_G(lam).exp()
    scale_spectral = G / G.mean()                  # a scale that got it right
    scale_blind = torch.ones_like(lam)             # a scale that did not

    spread = lambda v: float(v.max() / v.min())
    assert spread(tau) > 10
    assert spread(scale_blind * tau) > 10          # nothing cancels
    assert spread(scale_spectral * tau) < 1.001    # lambda is gone
    print()


def test_group_aggregation_keeps_one_prior_per_latent():
    """N observations of a reflection contribute one rate, not N."""
    loss = _PolyTau(C_1118)
    lam = torch.tensor([1.00, 1.10, 1.20, 1.05, 1.15])
    asu = torch.tensor([0, 0, 0, 1, 1])
    s_sq = torch.full_like(lam, 1.0 / 16.0)
    scale = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8])

    tau = loss._get_tau({"wavelength": lam}, s_sq, None)
    tau_h = _scatter_mean(scale * tau, asu, 2)

    assert tau_h.shape == (2,)
    assert float(tau_h[0]) == pytest.approx(float((scale[:3] * tau[:3]).mean()))
    assert float(tau_h[1]) == pytest.approx(float((scale[3:] * tau[3:]).mean()))
    # the rate does not grow with multiplicity, which is what applying the
    # prior once per latent means
    doubled = _scatter_mean(
        torch.cat([scale[:3], scale[:3]]),
        torch.zeros(6, dtype=torch.long),
        1,
    )
    assert float(doubled[0]) == pytest.approx(float(scale[:3].mean()))


def test_resolution_still_enters_per_observation():
    """s^2 comes from each observation's own d, not the group mean."""
    loss = _PolyTau(C_1118)
    lam = torch.full((3,), 1.05)
    d = torch.tensor([1.5, 2.5, 8.0])
    s_sq = 1.0 / (4.0 * d.pow(2))
    tau = loss._get_tau({"wavelength": lam}, s_sq, None)
    assert tau[0] > tau[1] > tau[2]
    assert math.isfinite(float(tau.sum()))
