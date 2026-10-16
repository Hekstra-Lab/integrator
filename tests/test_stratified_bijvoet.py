"""Conditioning on wavelength strata, instead of tying the mates' scales.

In a rotation experiment both Friedel mates ride nearly the same scale, so its
error cancels in `I(+) - I(-)`.  On a still they cannot: `-h` is excited at a
different wavelength from `h`, and their true scales differ by as much as the
spectrum spans.  Tying them would assert something false.

What is true is that every factor shared inside a (reflection, wavelength bin)
stratum cancels from the conditional distribution of the counts given their
total.  These tests check that this is what the estimator actually delivers: a
scale error that destroys the naive ratio must leave the conditional one alone.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stratified_bijvoet.py"
    spec = importlib.util.spec_from_file_location("stratified_bijvoet", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SB = _module()


def _dataset(
    rho_true=0.96,
    n_groups=300,
    n_bins=8,
    per_bin=8,
    i_true=400.0,
    bin_bias=0.0,
    seed=0,
):
    """Observations of Friedel mates spread over wavelength bins.

    `bin_bias` tilts how the two mates populate the bins: at 0 both mates are
    spread evenly, at 1 the plus mate sits in the low bins and the minus mate in
    the high ones.  Combined with a per-bin scale error that is what breaks a
    naive ratio on Laue data.
    """
    rng = np.random.default_rng(seed)
    # the nuisance: a per-(group, bin) factor, i.e. the spectrum and its error,
    # the resolution gauge, anything shared inside a stratum
    nuisance = np.exp(rng.normal(0.0, 0.8, size=(n_groups, n_bins)))

    s, e, group, strat, minus = [], [], [], [], []
    for g in range(n_groups):
        for b in range(n_bins):
            for sign in (0, 1):
                share = 1.0 + bin_bias * (1 if sign == 0 else -1) * (
                    1 - 2 * b / max(n_bins - 1, 1)
                )
                n = max(int(round(per_bin * max(share, 0.05))), 1)
                exposure = rng.uniform(0.6, 1.4, size=n)
                intensity = i_true * (rho_true if sign else 1.0)
                # what the detector sees: nuisance x exposure x intensity
                mu = nuisance[g, b] * exposure * intensity
                counts = rng.poisson(mu).astype(float)
                # the MODEL's exposure knows nothing of the nuisance
                s.append(counts)
                e.append(exposure)
                group.append(np.full(n, g))
                strat.append(np.full(n, g * n_bins + b))
                minus.append(np.full(n, float(sign)))
    cat = lambda xs: np.concatenate(xs)
    return (cat(s), cat(e), cat(group), cat(strat), cat(minus))


def _conditional(s, e, group, strat, minus):
    n_g = int(group.max()) + 1
    strat_u = np.unique(strat, return_inverse=True)[1]
    n_s = int(strat_u.max()) + 1
    g_of_s = np.zeros(n_s, dtype=np.int64)
    g_of_s[strat_u] = group
    sp = SB._sums(strat_u, s * (1 - minus), n_s)
    sm = SB._sums(strat_u, s * minus, n_s)
    ep = SB._sums(strat_u, e * (1 - minus), n_s)
    em = SB._sums(strat_u, e * minus, n_s)
    rho, info = SB.conditional_ratio(g_of_s, sp, sm, ep, em, n_g)
    return rho, info


def _naive(s, e, group, minus):
    """What the merge does now: ratio of scale-weighted sums, per mate."""
    n_g = int(group.max()) + 1
    sp = SB._sums(group, s * (1 - minus), n_g)
    sm = SB._sums(group, s * minus, n_g)
    ep = SB._sums(group, e * (1 - minus), n_g)
    em = SB._sums(group, e * minus, n_g)
    return (sm / np.maximum(em, 1e-30)) / np.maximum(sp / np.maximum(ep, 1e-30), 1e-30)


def test_conditioning_recovers_the_ratio_through_a_large_nuisance():
    """A nuisance of 0.8 in log, which is twice the measured scale error."""
    data = _dataset(rho_true=0.96, seed=1)
    rho, _ = _conditional(*data)
    assert np.median(rho) == pytest.approx(0.96, abs=0.01)


def test_the_naive_ratio_breaks_where_the_conditional_one_does_not():
    """The mates populate the band differently, as Laue geometry makes them."""
    data = _dataset(rho_true=0.96, bin_bias=0.9, seed=2)
    s, e, group, _, minus = data
    rho_cond, _ = _conditional(*data)
    rho_naive = _naive(s, e, group, minus)
    err_cond = abs(float(np.median(rho_cond)) - 0.96)
    err_naive = abs(float(np.median(rho_naive)) - 0.96)
    assert err_cond < 0.02, f"conditional off by {err_cond:.4f}"
    assert err_naive > 3 * max(err_cond, 1e-3), (
        f"naive {err_naive:.4f} vs conditional {err_cond:.4f}: the test did "
        "not reproduce the bias it is meant to exhibit"
    )


def test_a_single_sign_stratum_contributes_nothing():
    """Common support is automatic: no mate, no information, no bias."""
    s, e, group, strat, minus = _dataset(rho_true=1.0, n_groups=40, seed=3)
    keep = ~((strat % 8 == 0) & (minus > 0))      # delete one mate in one bin
    _, info_full = _conditional(s, e, group, strat, minus)
    rho_cut, info_cut = _conditional(*(v[keep] for v in (s, e, group, strat, minus)))
    assert np.median(rho_cut) == pytest.approx(1.0, abs=0.02)
    # the deleted stratum carried information, so the error bar must grow
    assert float(np.median(info_cut)) < float(np.median(info_full))


def test_the_estimate_is_invariant_to_rescaling_a_whole_stratum():
    """The exact invariance the construction is for."""
    s, e, group, strat, minus = _dataset(rho_true=0.9, n_groups=60, seed=4)
    rho_a, _ = _conditional(s, e, group, strat, minus)
    rng = np.random.default_rng(5)
    factor = np.exp(rng.normal(0, 1.5, size=int(strat.max()) + 1))[strat]
    rho_b, _ = _conditional(s, e * factor, group, strat, minus)
    assert np.allclose(rho_a, rho_b, rtol=1e-6)


def test_rescaling_one_mate_inside_a_stratum_does_move_it():
    """A sanity check in the other direction: it is not invariant to everything."""
    s, e, group, strat, minus = _dataset(rho_true=0.9, n_groups=60, seed=6)
    rho_a, _ = _conditional(s, e, group, strat, minus)
    rho_b, _ = _conditional(s, e * np.where(minus > 0, 1.5, 1.0), group, strat, minus)
    assert not np.allclose(rho_a, rho_b, rtol=1e-3)


def test_the_information_is_the_fisher_information_of_the_conditional():
    """sigma(log rho) = J^-1/2, so the spread of the estimates must match it."""
    rho_true = 1.0
    rho, info = _conditional(*_dataset(rho_true=rho_true, n_groups=600, seed=7))
    ok = info > 0
    z = np.log(rho[ok]) * np.sqrt(info[ok])
    assert abs(float(z.mean())) < 0.15
    assert float(z.std()) == pytest.approx(1.0, abs=0.15)


def test_the_dispersion_estimator_agrees_with_the_model_s():
    """The script's eps must be the same quantity the model inflates by."""
    rng = np.random.default_rng(8)
    n_g, n_obs, eps = 200, 100, 0.3
    s, e, group, d = [], [], [], []
    for g in range(n_g):
        exposure = rng.uniform(0.6, 1.4, size=n_obs)
        jitter = np.clip(1.0 + eps * rng.normal(size=n_obs), 0.05, None)
        s.append(rng.poisson(exposure * jitter * 500.0).astype(float))
        e.append(exposure)
        group.append(np.full(n_obs, g))
        d.append(np.full(n_obs, 2.0))
    eps_sq, _ = SB.dispersion(
        np.concatenate(group), np.concatenate(s), np.concatenate(e),
        np.concatenate(d), shells=1,
    )
    assert float(eps_sq[0]) ** 0.5 == pytest.approx(eps, rel=0.15)
