"""`polychromatic_wilson_eb`: the intensity prior mean set in closed form, separable in (lambda bin, d bin)."""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from integrator.model.loss.wilson_eb_loss import PolychromaticWilsonEBLoss
from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _loss(**kw):
    args = dict(degree=5, lambda_min=1.0, lambda_max=1.2, init_G=500.0, init_B=20.0, n_bins=4,
                eb_lambda_bins=5, bg_rate=1.0, bg_concentration=1.0)
    args.update(kw)
    return PolychromaticWilsonEBLoss(**args)


class _Q:
    def __init__(self, mean):
        self.mean = mean


def _md(n, gen):
    lam = 1.0 + 0.2 * torch.rand(n, generator=gen)
    k = torch.randint(0, 4, (n,), generator=gen)
    d = 1.8 + k.float()  # any d; the bins come from group_label
    return {"wavelength": lam, "group_label": k, "d": d}


def test_before_the_first_m_step_the_prior_is_the_wilson_prior():
    L = _loss()
    md = _md(100, torch.Generator().manual_seed(0))
    s_sq = 1 / (4 * md["d"] ** 2)
    ref = torch.exp(2 * L.get_B() * s_sq - L.spectrum.get_log_G(md["wavelength"]))
    torch.testing.assert_close(L._get_tau(md, s_sq, torch.device("cpu")), ref)
    assert not any(p.requires_grad for p in L.parameters())


def test_the_m_step_recovers_a_separable_mean():
    L = _loss()
    gen = torch.Generator().manual_seed(1)
    md = _md(40000, gen)
    g_true = torch.tensor([6.0, 6.3, 6.1, 5.8, 5.5], dtype=torch.float64)
    h_true = torch.tensor([0.8, 0.1, -0.3, -0.6], dtype=torch.float64)
    a, k = L._cells(md, torch.device("cpu"))
    mean = torch.exp(g_true[a] + h_true[k]).float()
    L._accumulate_fit(None, None, None, _Q(mean), md, torch.device("cpu"))
    L.refit_prior(0)
    fitted = L.eb_g[a] + L.eb_h[k]
    torch.testing.assert_close(fitted, torch.log(mean.double()), rtol=0, atol=1e-6)
    tau = L._get_tau(md, 1 / (4 * md["d"] ** 2), torch.device("cpu"))
    torch.testing.assert_close(tau, 1 / mean, rtol=1e-5, atol=0)
    wk = torch.bincount(k, minlength=4).double()
    assert abs(float((L.eb_h * wk).sum() / wk.sum())) < 1e-9  # h centred, g carries the scale


def test_damping_moves_part_way_and_centrics_keep_the_mean():
    L = _loss(eb_damping=0.5)
    gen = torch.Generator().manual_seed(2)
    md = _md(20000, gen)
    for target in (100.0, 400.0):
        L._accumulate_fit(None, None, None, _Q(torch.full((20000,), target)), md, torch.device("cpu"))
        L.refit_prior(0)
    # first M-step is undamped (log 100), the second halfway to log 400
    expect = 0.5 * math.log(100.0) + 0.5 * math.log(400.0)
    a, k = L._cells(md, torch.device("cpu"))
    torch.testing.assert_close(L.eb_g[a] + L.eb_h[k], torch.full((20000,), expect, dtype=torch.float64), rtol=0, atol=1e-6)
    md["centric"] = torch.arange(20000) % 2 == 0
    c, r = L.intensity_prior_params(md, torch.device("cpu"))
    assert torch.all(c[md["centric"]] == 0.5) and torch.all(c[~md["centric"]] == 1.0)
    torch.testing.assert_close(c / r, torch.full((20000,), math.exp(expect)), rtol=1e-5, atol=0)


def test_the_config_builds_and_trains_through_the_factory():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / "configs" / "sim_truth" / "mos010_default_kmixv2_post_eb.yaml").read_text())
    torch.manual_seed(0)
    m = construct_integrator(cfg).train()
    assert type(m.loss).__name__ == "PolychromaticWilsonEBLoss"
    assert not any(p.requires_grad for n, p in m.named_parameters() if n.startswith("loss."))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        out = m._step(_batch(), step="train")
        loss = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
    assert float(m.loss._eb_n.sum()) == 3 * 6
    m.loss.refit_prior(0)
    assert bool(m.loss.eb_ready) and torch.isfinite(m.loss.eb_g).all()


def test_a_fixed_prior_loads_and_is_never_refitted(tmp_path):
    g = torch.tensor([6.0, 6.3, 6.1, 5.8, 5.5], dtype=torch.float64)
    h = torch.tensor([0.8, 0.1, -0.3, -0.6], dtype=torch.float64)
    f = tmp_path / "eb.pt"
    torch.save({"eb_g": g, "eb_h": h}, f)
    L = _loss(eb_init_from=str(f), eb_refit=False)
    md = _md(1000, torch.Generator().manual_seed(4))
    a, k = L._cells(md, torch.device("cpu"))
    tau = L._get_tau(md, 1 / (4 * md["d"] ** 2), torch.device("cpu"))
    torch.testing.assert_close(tau.double(), torch.exp(-(g[a] + h[k])), rtol=1e-6, atol=0)
    assert not L.refit_enabled
    with pytest.raises(ValueError, match="eb_init_from"):
        _loss(eb_refit=False)
    with pytest.raises(ValueError, match="do not match"):
        _loss(eb_init_from=str(f), eb_lambda_bins=7)
