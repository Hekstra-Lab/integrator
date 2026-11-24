"""`scripts/sim_truth/sbc_simulate.py` draws from the same model the inference fits."""

from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sbc_simulate", ROOT / "scripts" / "sim_truth" / "sbc_simulate.py")
sbc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sbc)


def _model():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / "configs" / "sim_truth" / "mos010_default_kmixv2_post.yaml").read_text())
    torch.manual_seed(0)
    m = construct_integrator(cfg).eval()
    with torch.no_grad():  # move the metadata prior off its initialisation
        for p in m.surrogates["qp"].shape_head.parameters():
            p.normal_(0, 0.1)
    return m


def test_at_zero_spread_the_prior_profile_is_the_forwards_mean_profile():
    m = _model()
    qp = m.surrogates["qp"]
    counts, shoebox, mask, md = _batch()
    mask = mask.squeeze(-1)
    mask[0, :40] = 0  # a masked corner, so the renormalisation is exercised
    with torch.no_grad():
        qp.eta_delta.weight.zero_()
        qp.eta_delta.bias.zero_()
        qp.eta_sd_raw.fill_(-1e4)  # sigma = 0: the draw is the prior mean
        x = torch.randn(counts.shape[0], qp.eta_delta.in_features)
        ref = qp(x, mc_samples=1, metadata=md, mask=mask).mean_profile
        eta, eta_mean, sig, p = sbc.prior_profile(qp, md, mask, torch.Generator().manual_seed(0))
    torch.testing.assert_close(p, ref)
    assert torch.equal(eta, eta_mean)
    assert torch.all(p[0, :40] == 0)


def test_the_draw_spreads_by_sigma():
    m = _model()
    qp = m.surrogates["qp"]
    _, _, mask, md = _batch()
    with torch.no_grad():
        reps = [sbc.prior_profile(qp, md, mask.squeeze(-1), torch.Generator().manual_seed(s)) for s in range(400)]
    eta = torch.stack([r[0] for r in reps])
    z = (eta - reps[0][1]) / reps[0][2]
    assert abs(float(z.mean())) < 0.05
    assert abs(float(z.std()) - 1) < 0.05


def test_the_background_prior_recovers_a_known_gamma():
    rng = np.random.default_rng(0)
    n, P = 40000, 625
    alpha, beta = np.array([20.0, 60.0]), np.array([1.0, 2.0])  # means 20 and 30 photons/px
    groups = rng.integers(0, 2, n)
    b = rng.gamma(alpha[groups], 1 / beta[groups])
    counts = rng.poisson(b[:, None], size=(n, P)).astype(np.float32)
    masks = np.ones((n, P), bool)
    a_hat, b_hat, mu = sbc.empirical_bg_prior(counts, masks, groups, 2)
    np.testing.assert_allclose(mu, alpha / beta, rtol=0.01)
    # the median of integer counts is coarse at ~20/px; the shape is recovered to 15%
    np.testing.assert_allclose(a_hat, alpha, rtol=0.15)
    assert math.isfinite(float(b_hat.sum()))


def test_ranks_are_uniform_under_an_exact_posterior():
    spec = importlib.util.spec_from_file_location("sbc_ranks", ROOT / "scripts" / "sim_truth" / "sbc_ranks.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    rng = np.random.default_rng(1)
    a, rate = 1.0, 0.02  # Wilson-like: exponential, mean 50
    I = rng.gamma(a, 1 / rate, 200000)
    y = rng.poisson(I)
    pa, pr = a + y, rate + 1.0  # the exact Gamma posterior
    u = r.gamma_pit(I, pa / pr, pa / pr**2)
    s = r.summarise(u)
    assert abs(s["mean_minus_half"]) < 0.005 and abs(s["var12"] - 1) < 0.01 and s["ks"] < 1.36
    # a posterior twice too narrow is flagged
    s = r.summarise(r.gamma_pit(I, pa / pr, pa / pr**2 / 4))
    assert s["var12"] > 1.3 and s["ks"] > 1.36


def test_q_eta_reproduces_the_forwards_mean_profile():
    m = _model()
    qp = m.surrogates["qp"]
    counts, shoebox, mask, md = _batch()
    with torch.no_grad():
        qp.eta_delta.weight.normal_(0, 0.05)
        qp.eta_scale.weight.normal_(0, 0.05)
        x = torch.randn(counts.shape[0], qp.eta_delta.in_features)
        qp._refine_capture = []
        ref = qp(x, mc_samples=1, metadata=md, mask=mask.squeeze(-1)).mean_profile
        raw_delta, raw_scale = qp._refine_capture
        qp._refine_capture = None
        phi, fxy, eta_mean = sbc.eta_prior_mean(qp, md, counts)
        loc, sd = sbc.q_eta(qp, eta_mean, raw_delta, raw_scale)
        p = sbc.profile_at(qp, phi, fxy, loc, mask)
    torch.testing.assert_close(p, ref)
    assert torch.all(sd > 0)


def test_rank_pit_is_uniform_when_the_truth_is_one_more_draw():
    spec = importlib.util.spec_from_file_location("sbc_ranks", ROOT / "scripts" / "sim_truth" / "sbc_ranks.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    g = torch.Generator().manual_seed(0)
    draws = torch.randn(64, 100000, generator=g)
    truth = torch.randn(100000, generator=g)
    s = r.summarise(r.rank_pit(truth, draws, g).numpy())
    assert abs(s["mean_minus_half"]) < 0.005 and abs(s["var12"] - 1) < 0.01 and s["ks"] < 1.36
