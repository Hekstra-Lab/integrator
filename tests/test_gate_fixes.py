"""`gate_i0_floor` and `gate_background`: the responsibility gate's two starting estimates, defaults unchanged."""

from __future__ import annotations

import copy
import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth" / "mos010_default_kmixv2_post.yaml"


def _pair(**kw):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CFG.read_text())
    torch.manual_seed(0)
    base = construct_integrator(cfg).eval()
    c2 = copy.deepcopy(cfg)
    c2["integrator"]["args"].update(kw)
    m = construct_integrator(c2).eval()
    m.load_state_dict(base.state_dict())
    return base, m


def _faint_batch():
    counts, shoebox, mask, md = _batch()
    torch.manual_seed(5)
    counts = torch.poisson(torch.full_like(counts, 20.0))  # background only: least-squares i0 is often negative
    return counts, counts.clone().unsqueeze(-1), mask, md


def _qi(m, batch):
    torch.manual_seed(1)
    with torch.no_grad():
        return m(*batch)["qi"].mean


def test_defaults_are_bit_identical():
    base, m = _pair(gate_i0_floor=0.0, gate_background="median")
    b = _faint_batch()
    assert torch.equal(_qi(base, b), _qi(m, b))


def test_the_floor_keeps_the_gate_open_where_i0_would_clamp_to_zero():
    base, m = _pair(gate_i0_floor=0.5)
    b = _faint_batch()
    a_I, r_I = base._intensity_prior(b[3], torch.device("cpu"))
    q0, q1 = _qi(base, b), _qi(m, b)
    # with the gate closed q(I) = Gamma(alpha_I, tau + 1): mean alpha_I / (tau + 1), about 1 photon
    closed = (q0 - a_I / (r_I + 1)).abs() < 1e-3
    assert closed.any(), "the test batch should contain closed gates"
    assert torch.all(q1[closed] > q0[closed])


def test_least_squares_background_is_continuous_and_changes_q():
    base, m = _pair(gate_background="least_squares")
    counts, shoebox, mask, md = _batch()
    q0, q1 = _qi(base, (counts, shoebox, mask, md)), _qi(m, (counts, shoebox, mask, md))
    assert torch.isfinite(q1).all() and not torch.allclose(q0, q1)
