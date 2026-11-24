"""`export_refine_steps`: per-reflection refinement of the shape posterior at prediction, trained weights untouched."""

from __future__ import annotations

import copy
import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth"


def _load(name: str, **integrator_args) -> dict:
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((CFG / f"{name}.yaml").read_text())
    cfg["integrator"]["args"].update(integrator_args)
    cfg["integrator"]["args"]["predict_keys"] = ["i_rb", "var_rb", "qp_mean", "qbg_mean"]
    return cfg


def _model(**kw):
    torch.manual_seed(0)
    m = construct_integrator(_load("mos010_default_kmixv2_post", **kw)).eval()
    # move the shape heads off their 1e-3 initialisation so delta(x) is not ~0
    with torch.no_grad():
        qp = m.surrogates["qp"]
        qp.eta_delta.weight.normal_(0, 0.05)
        qp.eta_scale.weight.normal_(0, 0.05)
    return m


def test_the_keys_default_off_and_build_through_the_factory():
    m = construct_integrator(_load("mos010_default_kmixv2_post"))
    assert m.export_refine_steps == 0
    m = construct_integrator(_load("mos010_default_kmixv2_post", export_refine_steps=5, export_refine_lr=0.1))
    assert (m.export_refine_steps, m.export_refine_lr) == (5, 0.1)


def test_overriding_with_the_encoders_own_values_changes_nothing():
    m = _model()
    counts, shoebox, mask, md = _batch()
    qp = m.surrogates["qp"]
    raw = qp.refine_init(lambda: m(counts, shoebox, mask, md))
    torch.manual_seed(1)
    a = m(counts, shoebox, mask, md)["qp"].mean_profile
    qp.refine_override = raw
    try:
        torch.manual_seed(1)
        b = m(counts, shoebox, mask, md)["qp"].mean_profile
    finally:
        qp.refine_override = None
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)  # raw captured under no_grad: float-level differences only


def test_steps_zero_predicts_as_before():
    m = _model()
    batch = _batch()
    torch.manual_seed(2)
    with torch.inference_mode():
        a = m.predict_step(batch, 0)
    m.export_refine_steps = 0
    torch.manual_seed(2)
    with torch.inference_mode():
        b = m.predict_step(batch, 0)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_refinement_lowers_the_batch_loss_and_leaves_the_weights_alone():
    m = _model(export_refine_steps=30, export_refine_lr=0.05)
    before = copy.deepcopy(m.state_dict())
    batch = _batch()
    counts, shoebox, mask, md = batch
    qp = m.surrogates["qp"]

    def loss_at(override):
        losses = []
        for seed in range(8):  # the shape posterior is sampled; average the noise out
            torch.manual_seed(seed)
            qp.refine_override = override
            try:
                with torch.no_grad():
                    losses.append(float(m._batch_loss(m(counts, shoebox, mask, md), md)))
            finally:
                qp.refine_override = None
        return sum(losses) / len(losses)

    raw0 = qp.refine_init(lambda: m(counts, shoebox, mask, md))
    captured = {}
    orig = torch.optim.Adam

    class Spy(orig):
        def __init__(self, params, **kw):
            captured["free"] = params
            super().__init__(params, **kw)
    torch.optim.Adam = Spy
    try:
        with torch.inference_mode():
            out = m.predict_step(batch, 0)
    finally:
        torch.optim.Adam = orig

    refined = [t.detach() for t in captured["free"]]
    assert loss_at(refined) < loss_at(raw0)
    assert not torch.allclose(refined[0], raw0[0])
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed during refinement"
    assert all(p.grad is None for p in m.parameters())
    assert qp.refine_override is None
    m.export_refine_steps = 0
    with torch.inference_mode():
        plain = m.predict_step(batch, 0)
    assert not torch.allclose(out["qp_mean"], plain["qp_mean"])
    assert torch.isfinite(out["i_rb"]).all()


def test_a_surrogate_without_per_reflection_parameters_refuses():
    torch.manual_seed(0)
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((CFG / "mos010_default_kmix_base.yaml").read_text())
    cfg["integrator"]["args"]["export_refine_steps"] = 3
    m = construct_integrator(cfg).eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="per-reflection variational parameters"):
        m.predict_step(_batch(), 0)
