"""`streak_kmix_v2_geom`: the kmix-v2 profile with a geometric prior mean in t = tan(2 theta)."""

from __future__ import annotations

import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import _batch

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth" / "mos010_default_kmixv2_post_collapsed_geom.yaml"


def _qp():
    yaml = pytest.importorskip("yaml")
    torch.manual_seed(0)
    return construct_integrator(yaml.safe_load(CFG.read_text())).surrogates["qp"]


def test_prior_widths_round_trip_through_the_raw_coordinates():
    qp = _qp()
    md = _batch()[3]
    with torch.no_grad():
        _, mu, _ = qp.prior_mean(md, torch.float64, torch.device("cpu"))
        along, across = qp.geom_widths(md, torch.float64, torch.device("cpu"))
        a, c, w, sh = qp.unpack(mu)
    torch.testing.assert_close(a, along, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(c, across, rtol=1e-6, atol=1e-6)


def test_widths_grow_with_radius_and_the_network_is_gone():
    qp = _qp()
    assert qp.shape_head is None
    with torch.no_grad():
        qp.beta_raw.fill_(1.0)
        qp.eps_raw.fill_(1.0)
        md = {"xyzcal.px.0": torch.tensor([2074.5, 2074.5 + 500, 2074.5 + 1500]),
              "xyzcal.px.1": torch.tensor([2181.5 + 1, 2181.5, 2181.5])}
        along, across = qp.geom_widths(md, torch.float64, torch.device("cpu"))
    assert torch.all(along[1:] > along[:-1]) and torch.all(across[1:] > across[:-1])


def test_the_forward_and_the_eta_helpers_run():
    qp = _qp().eval()
    counts, shoebox, mask, md = _batch()
    with torch.no_grad():
        phi, fxy, eta_mean = qp.eta_prior(md, mask.squeeze(-1))
        p = qp.profile_from_eta(phi, fxy, eta_mean, mask.squeeze(-1))
    assert p.shape == (6, 625) and torch.allclose(p.sum(-1), torch.ones(6))
