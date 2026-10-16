"""Fitting the scale by conditional likelihood, and not absorbing the signal.

The scale is where the polychromatic merge fails: replacing ours with
careless's, and changing nothing else, took the mean anomalous peak from 3.35
to 6.17. Fitting it directly means profiling out the intensities, which the
conditional (multinomial) Poisson likelihood does exactly.

Two properties have to hold. The fit must recover a planted scale, and it must
NOT be able to absorb a planted Bijvoet difference -- a scale with enough
freedom to represent a per-(reflection, Friedel sign) factor can manufacture
or eat anomalous signal, since that factor is in the gauge of the likelihood.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_physical_scale.py"
    spec = importlib.util.spec_from_file_location("fit_physical_scale", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FP = _module()


def _synthetic(n_refl=400, n_obs=40, planted_log_rho=0.0, seed=0, intensity=800.0):
    """Observations of Friedel mates under a smooth scale in one covariate."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_refl), n_obs)
    t = rng.uniform(0.0, 1.0, size=n_refl * n_obs)      # the covariate
    e = rng.uniform(0.6, 1.4, size=n_refl * n_obs)      # base exposure
    truth = 0.8 * t - 0.5 * t**2                        # the scale, in log
    is_minus = (np.arange(n_refl * n_obs) % 2 == 1).astype(float)
    I = intensity * np.exp(planted_log_rho * is_minus)  # the planted contrast
    mu = e * np.exp(truth) * I
    S = rng.poisson(mu).astype(float)
    X = np.stack([t, t**2, t**3], axis=1)
    return X, S, e, group, truth, is_minus


def test_the_fit_recovers_a_planted_scale():
    X, S, e, group, truth, _ = _synthetic(seed=1)
    c, iters, _ = FP.fit_conditional(X, S, e, group, pen=np.zeros(X.shape[1]))
    got = X @ c
    # the overall level is a gauge, so compare after centring
    err = (got - got.mean()) - (truth - truth.mean())
    assert float(np.abs(err).max()) < 0.02, f"max error {np.abs(err).max():.4f}"
    assert iters < 20


def test_the_likelihood_is_blind_to_a_per_reflection_rescaling():
    """I_h is profiled out, so a per-group factor must not move the fit."""
    X, S, e, group, _, _ = _synthetic(n_refl=120, seed=2)
    pen = np.zeros(X.shape[1])
    c_a, _, _ = FP.fit_conditional(X, S, e, group, pen=pen)
    rng = np.random.default_rng(3)
    factor = np.exp(rng.normal(0, 1.5, size=group.max() + 1))[group]
    c_b, _, _ = FP.fit_conditional(X, S, e * factor, group, pen=pen)
    assert np.allclose(c_a, c_b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("planted", [0.03, -0.03])
def test_the_scale_cannot_absorb_a_planted_bijvoet_difference(planted):
    """The leak test: plant log rho, fit the scale, measure what survives.

    The planted contrast alternates between mates and is uncorrelated with the
    scale's covariate, so a correctly specified scale leaves it untouched. A
    scale that could represent a per-(reflection, sign) factor would eat it.
    """
    X, S, e, group, _, is_minus = _synthetic(
        n_refl=600, n_obs=40, planted_log_rho=planted, seed=4
    )
    c, _, _ = FP.fit_conditional(X, S, e, group, pen=np.zeros(X.shape[1]))
    s_fit = np.exp(X @ c)
    # recover rho per reflection from the fitted scale, as the merge would
    n_h = group.max() + 1
    num = np.zeros(n_h); den = np.zeros(n_h)
    np.add.at(num, group, S * is_minus)
    np.add.at(den, group, S * (1 - is_minus))
    enum = np.zeros(n_h); eden = np.zeros(n_h)
    np.add.at(enum, group, e * s_fit * is_minus)
    np.add.at(eden, group, e * s_fit * (1 - is_minus))
    rho = (num / np.maximum(enum, 1e-12)) / np.maximum(den / np.maximum(eden, 1e-12), 1e-12)
    recovered = float(np.median(np.log(rho[np.isfinite(rho) & (rho > 0)])))
    assert recovered == pytest.approx(planted, abs=0.01), (
        f"planted {planted:+.3f}, recovered {recovered:+.3f}: the scale "
        "absorbed the contrast"
    )


def test_polarisation_is_a_known_form_with_the_rate_multiplied_by_P():
    """P = [1 + cos^2(2t) - tau cos(2phi) sin^2(2t)]/2 multiplies the rate.

    The tabulated quantity is usually the correction 1/P; applying that as the
    scale inverts the azimuthal dependence, which is what the scale module
    currently does.
    """
    two_t = np.full(4, np.deg2rad(49.25))      # 1.5 A at 1.25 A wavelength
    phi = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    logP = FP.polarisation_offset(two_t, phi, tau=0.9)
    P = np.exp(logP)
    # in the polarisation plane the scattered intensity is suppressed
    assert P[0] < P[1]
    assert P[0] == pytest.approx(P[2], rel=1e-12)
    assert P[1] == pytest.approx(P[3], rel=1e-12)
    swing = P.max() / P.min()
    assert swing > 2.0, f"azimuthal swing only {swing:.2f}x"
