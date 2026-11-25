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
