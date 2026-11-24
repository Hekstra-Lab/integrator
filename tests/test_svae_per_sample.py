"""`svae_per_sample`: q(I) and q(b) conditioned on each profile sample."""

from __future__ import annotations

import copy
import pathlib

import pytest
import torch
from torch.distributions import Gamma, kl_divergence

from integrator.model.integrators.svae_per_sample_integrator import PerSampleGamma
from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth"


def test_mixture_moments_and_kl():
    g = torch.Generator().manual_seed(0)
    c = 1 + 5 * torch.rand(7, 4, generator=g)
    r = 0.1 + torch.rand(7, 4, generator=g)
    q = PerSampleGamma(c, r)
    m = c / r
    torch.testing.assert_close(q.mean, m.mean(0))
    torch.testing.assert_close(q.variance, (c / r**2).mean(0) + m.var(0, unbiased=False))
    p = Gamma(torch.ones(4), torch.full((4,), 0.01))
    torch.testing.assert_close(kl_divergence(q, p), kl_divergence(Gamma(c, r), p).mean(0))
    assert q.rsample([7]).shape == (7, 4)
    with pytest.raises(ValueError):
        q.rsample([3])


def test_draws_follow_their_component():
    c = torch.tensor([[1e4], [4e4]])  # component 0 near 100, component 1 near 400
    r = torch.tensor([[100.0], [100.0]])
    x = PerSampleGamma(c, r).rsample([2])
    assert abs(float(x[0, 0]) - 100) < 5 and abs(float(x[1, 0]) - 400) < 10


def _pair():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((CFG / "mos010_default_kmixv2_post.yaml").read_text())
    torch.manual_seed(0)
    base = construct_integrator(cfg).eval()
    cfg2 = copy.deepcopy(cfg)
    cfg2["integrator"]["name"] = "svae_per_sample"
    per = construct_integrator(cfg2).eval()
    per.load_state_dict(base.state_dict(), strict=True)
    return base, per


def test_with_no_profile_spread_it_reproduces_svae():
    base, per = _pair()
    with torch.no_grad():
        for m in (base, per):
            m.surrogates["qp"].eta_sd_raw.fill_(-1e4)  # sigma = 0: every profile sample is the mean profile
        counts, shoebox, mask, md = _batch()
        torch.manual_seed(3)
        a = base(counts, shoebox, mask, md)
        torch.manual_seed(3)
        b = per(counts, shoebox, mask, md)
    torch.testing.assert_close(b["qi"].mean, a["qi"].mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b["qbg"].mean, a["qbg"].mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b["qi"].variance, a["qi"].variance, rtol=1e-4, atol=1e-4)
    assert b["forward_out"]["signal_prob"].shape == a["forward_out"]["signal_prob"].shape


def test_profile_spread_widens_q_i():
    base, per = _pair()
    with torch.no_grad():
        counts, shoebox, mask, md = _batch()
        torch.manual_seed(3)
        a = base(counts, shoebox, mask, md)
        torch.manual_seed(3)
        b = per(counts, shoebox, mask, md)
    assert torch.all(b["qi"].variance >= 0.999 * b["qi"]._components.variance.mean(0))
    assert torch.isfinite(b["forward_out"]["rates"]).all()
    assert b["forward_out"]["rates"].shape == a["forward_out"]["rates"].shape


def test_the_rb_export_runs():
    _, per = _pair()
    per.predict_keys = ["i_rb", "var_rb", "qi_mean", "qi_var", "qbg_mean", "signal_prob"]
    with torch.inference_mode():
        out = per.predict_step(_batch(), 0)
    for k in per.predict_keys:
        assert torch.isfinite(out[k]).all(), k
    assert out["i_rb"].shape == out["qi_mean"].shape == (6,)
