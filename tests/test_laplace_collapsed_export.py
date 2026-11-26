"""`export_posterior: laplace_collapsed`: prediction-only, default off, weights and state_dict untouched."""

from __future__ import annotations

import copy
import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth" / "mos010_default_kmixv2_post.yaml"


def _model(**kw):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CFG.read_text())
    cfg["integrator"]["args"].update(kw)
    cfg["integrator"]["args"]["predict_keys"] = ["qi_mean", "qi_var", "qbg_mean", "qbg_var", "i_rb", "var_rb"]
    torch.manual_seed(0)
    m = construct_integrator(cfg).eval()
    with torch.no_grad():
        m.surrogates["qp"].eta_delta.weight.normal_(0, 0.05)
    return m


def _batch_bp():
    counts, shoebox, mask, md = _batch()
    md["group_label"] = torch.zeros(6, dtype=torch.long)
    return counts, shoebox.squeeze(-1), mask.squeeze(-1), md


def test_default_is_off():
    assert _model().export_posterior == "none"


def test_export_adds_finite_columns_and_leaves_the_model_alone():
    m = _model(export_posterior="laplace_collapsed", export_posterior_eta_samples=4, export_posterior_steps=10,
               export_posterior_grid=64)
    before = copy.deepcopy(m.state_dict())
    with torch.inference_mode():
        out = m.predict_step(_batch_bp(), 0)
    for k in ("i_lc", "var_lc", "bg_lc", "var_bg_lc"):
        assert out[k].shape == (6,) and torch.isfinite(out[k]).all(), k
    assert (out["var_lc"] > 0).all() and (out["bg_lc"] > 0).all()
    # the collapsed variance includes the background's uncertainty, which the conjugate q(I) omits
    assert float(out["var_lc"].median()) > float(out["qi_var"].median())
    assert set(m.state_dict()) == set(before)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in m.parameters())


def _lc(**kw):
    m = _model(export_posterior="laplace_collapsed", export_posterior_eta_samples=4, export_posterior_steps=10,
               export_posterior_grid=64, **kw)
    torch.manual_seed(1)  # the eta draws
    with torch.inference_mode():
        return m, m.predict_step(_batch_bp(), 0)


def test_intensity_prior_default_is_the_model_prior():
    m, base = _lc()
    assert m.export_posterior_intensity_prior == "model"
    _, same = _lc(export_posterior_intensity_prior="model")
    for k in ("i_lc", "var_lc", "bg_lc", "var_bg_lc"):
        assert torch.equal(base[k], same[k]), k


def test_flat_intensity_prior_changes_only_the_exported_posterior():
    _, base = _lc()
    _, flat = _lc(export_posterior_intensity_prior="flat")
    for k in ("i_lc", "var_lc", "bg_lc", "var_bg_lc"):
        assert torch.isfinite(flat[k]).all() and (flat[k] > 0).all(), k
    assert not torch.allclose(flat["i_lc"], base["i_lc"])
    # the amortized columns do not see the export's prior
    for k in ("qi_mean", "qi_var", "i_rb", "var_rb"):
        assert torch.equal(flat[k], base[k]), k


def test_flat_intensity_prior_is_the_normalised_likelihood():
    """With Gamma(1, 1e-10) on I, the collapsed posterior mean of I is the mean of
    likelihood(I, b) * p(b) integrated numerically over a 2-D grid."""
    from integrator.model.posterior.collapsed import collapsed

    torch.manual_seed(0)
    P = 49
    p = torch.softmax(torch.randn(3, P), -1).double()
    c = torch.poisson(torch.tensor([[5.0], [60.0], [800.0]]) * p + 2.0).double()
    a_b, r_b = 4.0, 2.0
    q = collapsed(c, torch.ones_like(c), p, torch.ones(3), torch.full((3,), 1e-10),
                  torch.full((3,), a_b), torch.full((3,), r_b), grid=512, newton_b=30)
    # newton_b=30: at the default 6 the inner b-mode is unconverged at I ~ 800 and the
    # mean is 0.8 % low (764.0 vs 770.0); that is a property of `collapsed`, not the prior
    got = q.mean_var_I()[0]
    bs = torch.linspace(0.01, 8.0, 800, dtype=torch.float64)
    for row, hi in zip(range(3), (60.0, 250.0, 1500.0)):
        xs = torch.linspace(1e-6, hi, 3000, dtype=torch.float64)
        lam = xs[:, None, None] * p[row] + bs[None, :, None]
        lp = (c[row] * torch.log(lam) - lam).sum(-1) + (a_b - 1) * torch.log(bs) - r_b * bs
        w = torch.exp(lp - lp.max()).sum(1)
        ref = (w * xs).sum() / w.sum()
        assert abs(float(got[row]) / float(ref) - 1) < 5e-3, (row, float(got[row]), float(ref))
