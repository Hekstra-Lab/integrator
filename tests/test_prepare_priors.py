"""Robustness tests for the empirical-prior fitting in prepare_priors."""

from __future__ import annotations

import torch

from integrator.utils.prepare_priors import _fit_gamma_mle, _fit_per_bin_gamma


def test_gamma_mle_constant_input_stays_finite():
    """Zero-variance input must not blow up (alpha = mean^2 / var -> inf)."""
    x = torch.full((100,), 5.0)
    alpha, beta = _fit_gamma_mle(x)
    assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
    assert alpha > 0 and beta > 0


def test_gamma_mle_recovers_reasonable_fit():
    """On dispersed positive data the fit is finite and sensibly scaled."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(5000, generator=g) * 8.0 + 1.0  # ~Uniform(1, 9)
    alpha, beta = _fit_gamma_mle(x)
    assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
    # mean of Gamma(alpha, beta) is alpha/beta; should track the sample mean
    assert torch.isclose(alpha / beta, x.mean(), rtol=0.1)


def test_fit_per_bin_gamma_returns_finite_pair_per_bin():
    """Per-bin fit yields n_bins finite (alpha, rate) pairs; sparse bins fall back."""
    g = torch.Generator().manual_seed(0)
    n_bins = 4
    bg = torch.rand(2000, generator=g) * 8.0 + 1.0
    labels = torch.randint(0, n_bins, (2000,), generator=g)
    labels[bg.argsort()[:5]] = n_bins - 1  # make one bin tiny to hit the fallback
    alphas, rates = _fit_per_bin_gamma(bg, labels, n_bins)
    assert len(alphas) == n_bins and len(rates) == n_bins
    assert all(a > 0 and r > 0 for a, r in zip(alphas, rates))
