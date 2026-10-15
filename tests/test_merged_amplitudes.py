"""The amplitude moments the merged MTZ is written from.

A defect here is invisible in the MTZ and shows up only as a worse
refinement, which is then blamed on the model.  `SigF` was computed by a
subtraction that cancels in float32, which inflated it roughly tenfold on the
brightest reflections and clamped others to zero, where the writer deleted
the row.  That presented as a 6.5x anomalous-peak deficit.
"""

import numpy as np
import pytest
import torch
from scipy.special import gammaln

from integrator.cli.utils.merged_mtz_writer import _gamma_amplitude


@pytest.mark.parametrize("k", [1.0, 10.0, 100.0, 1e3, 1e4])
@pytest.mark.parametrize("rate", [1.0, 137.0])
def test_amplitude_sigma_matches_the_exact_form_where_that_is_safe(k, rate):
    """Below the crossover the subtraction still has digits left to check against."""
    log_r = gammaln(k + 0.5) - gammaln(k)
    want = np.sqrt(max(k - np.exp(2.0 * log_r), 0.0) / rate)
    _, sig_F = _gamma_amplitude(np.array([k]), np.array([rate]))
    assert sig_F[0] == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("k", [1e5, 1e6, 1e8])
@pytest.mark.parametrize("rate", [1e-3, 1.0, 137.0])
def test_amplitude_sigma_holds_its_asymptote(k, rate):
    """`sd[sqrt(I)] -> 1 / (2 sqrt(rate))` however large the shape gets.

    Past `k ~ 1e5` even float64 loses the subtraction, so this is the regime
    the series branch exists for.
    """
    _, sig_F = _gamma_amplitude(np.array([k]), np.array([rate]))
    assert sig_F[0] == pytest.approx(0.5 / np.sqrt(rate), rel=1e-5)


def test_amplitude_sigma_is_continuous_across_the_crossover():
    """A visible step here would be a discontinuity in every exported SigF.

    The true kink is 1.3e-7; a wrong coefficient in the series shows up as a
    step of order 1e-6, which this catches.
    """
    k = np.array([9.99e3, 1.0e4, 1.001e4])
    _, sig_F = _gamma_amplitude(k, np.ones_like(k))
    assert np.all(np.abs(np.diff(sig_F)) < 5e-7)


def test_amplitude_mean_is_the_gamma_half_moment():
    """`E[sqrt(I)] = Gamma(k + 1/2) / Gamma(k) / sqrt(rate)`, no cancellation."""
    k = np.array([0.5, 1.0, 7.0, 1e3])
    rate = np.full_like(k, 3.0)
    F_mean, _ = _gamma_amplitude(k, rate)
    want = np.exp(gammaln(k + 0.5) - gammaln(k)) / np.sqrt(rate)
    assert np.allclose(F_mean, want, rtol=1e-12)


@pytest.mark.parametrize("k,ratio", [(1e4, 10.0), (1e6, 100.0), (1e8, 1000.0)])
def test_the_float32_path_is_the_defect_this_replaced(k, ratio):
    """Guards the reason for the helper, in the arithmetic that produced it.

    A float32 `lgamma` cannot represent `log Gamma(k)` finely enough for its
    difference to survive, and the error is then multiplied by `k`.
    """
    kt = torch.tensor([k], dtype=torch.float32)
    log_r = torch.lgamma(kt + 0.5) - torch.lgamma(kt)
    naive = torch.sqrt(torch.clamp(kt - torch.exp(log_r).pow(2), min=0.0))
    assert naive.item() > ratio * 0.5

    _, sig_F = _gamma_amplitude(np.array([k]), np.array([1.0]))
    assert sig_F[0] == pytest.approx(0.5, rel=1e-5)
