"""`svae_collapsed`: full-covariance q(eta) and the collapsed bound."""

from __future__ import annotations

import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth" / "sbc_prior_b1_kmixv2_post_collapsed.yaml"


def _model():
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CFG.read_text())
    cfg["integrator"]["args"]["predict_keys"] = ["qi_mean", "qi_var", "qbg_mean", "qbg_var", "eta_loc", "eta_chol",
                                                 "i_rb", "var_rb"]
    torch.manual_seed(0)
    m = construct_integrator(cfg)
    with torch.no_grad():
        m.surrogates["qp"].eta_delta.weight.normal_(0, 0.05)
        m.surrogates["qp"].eta_scale.weight.normal_(0, 0.05)
    return m


def _bp():
    counts, shoebox, mask, md = _batch()
    return counts, shoebox.squeeze(-1), mask.squeeze(-1), md


def test_at_initialisation_q_eta_and_its_kl_are_the_surrogates_diagonal_ones():
    m = _model().eval()
    counts, shoebox, mask, md = _bp()
    qp = m.surrogates["qp"]
    with torch.no_grad():
        x = m.encoders["profile"](m._profile_encoder_input((shoebox * mask).reshape(6, 1, 25, 25), md))
        phi, fxy, eta_mean, loc, L, sig, mm = m._q_eta(x, md, mask)
        loc0, sd0 = qp.q_eta_from_raw(eta_mean, qp.eta_delta(x), qp.eta_scale(x))
        _, kl, *_ = m._bound(counts, shoebox, mask, md)
    torch.testing.assert_close(L, torch.diag_embed(sd0))
    torch.testing.assert_close(loc, loc0)
    delta, s = (loc0 - eta_mean) / sig, sd0 / sig
    kl_diag = 0.5 * (delta**2 + s**2 - 1 - 2 * torch.log(s)).sum(-1)  # the surrogate's own KL
    torch.testing.assert_close(kl, kl_diag, rtol=1e-5, atol=1e-5)


def test_training_steps_are_finite_and_the_offdiagonal_head_learns():
    m = _model().train()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    w0 = m.eta_offdiag.weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        out = m._step(_bp(), step="train")
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        opt.step()
    assert not torch.equal(m.eta_offdiag.weight, w0)


def test_prediction_exports_the_posterior_and_q_eta():
    m = _model().eval()
    with torch.inference_mode():
        out = m.predict_step(_bp(), 0)
    for k in ("qi_mean", "qi_var", "qbg_mean", "qbg_var", "i_rb", "var_rb"):
        assert out[k].shape == (6,) and torch.isfinite(out[k]).all(), k
    assert out["eta_loc"].shape == (6, 12) and out["eta_chol"].shape == (6, 12, 12)
    assert (out["qi_var"] > 0).all()
