"""`polychromatic_wilson_bgmap`: a learned detector map as the background prior's mean."""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from integrator.registry import REGISTRY
from integrator.utils.factory_utils import construct_integrator

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth"
B, SIDE = 6, 25
P = SIDE * SIDE


def _load(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((CFG / f"{name}.yaml").read_text())


@pytest.fixture
def batch():
    torch.manual_seed(0)
    x = torch.tensor([600.0, 3500.0, 2074.5, 900.0, 3000.0, 1500.0])
    y = torch.tensor([700.0, 900.0, 3600.0, 3400.0, 2181.5, 1200.0])
    counts = torch.poisson(torch.full((B, P), 19.0))
    counts[:, P // 2] += 400.0
    metadata = {
        "group_label": torch.zeros(B, dtype=torch.long),
        "xyzcal.px.0": x, "xyzcal.px.1": y,
        "bbox.0": torch.floor(x) - SIDE // 2, "bbox.2": torch.floor(y) - SIDE // 2,
        "d": torch.full((B,), 2.0), "wavelength": torch.full((B,), 1.1),
        "image_num": torch.tensor([1.0, 2.0, 3.0, 90.0, 179.0, 180.0]),
    }
    return counts, counts.clone().unsqueeze(-1), torch.ones(B, P, 1), metadata


def test_is_registered():
    assert "polychromatic_wilson_bgmap" in REGISTRY["loss"]


def test_prior_mean_starts_at_bg_init_everywhere(batch):
    loss = construct_integrator(_load("mos010_default_kmix_base_cheb40_bgmap")).loss
    conc, rate = loss._background_prior(None, batch[3], "cpu")
    assert torch.allclose(conc / rate, torch.full((B,), 19.0), rtol=1e-5)
    assert torch.allclose(conc, torch.full((B,), 1.0), atol=1e-3)


def test_image_offset_cannot_trade_with_the_constant(batch):
    loss = construct_integrator(_load("mos010_default_kmix_base_cheb40_bgmap")).loss
    before = loss.bg_log_mean(batch[3], "cpu")
    with torch.no_grad():
        loss.bg_image_offset.add_(0.7)
    assert torch.allclose(loss.bg_log_mean(batch[3], "cpu"), before, atol=1e-6)


def test_the_map_parameters_receive_gradient_through_the_background_kl(batch):
    model = construct_integrator(_load("mos010_default_kmix_base_cheb40_bgmap")).train()
    counts, shoebox, mask, metadata = batch
    out = model._step((counts, shoebox, mask, metadata), step="train")
    loss = out["loss"] if isinstance(out, dict) else out
    assert torch.isfinite(loss)
    loss.backward()
    L = model.loss
    for name, p in (("bg_coef", L.bg_coef), ("bg_kappa_raw", L.bg_kappa_raw), ("bg_image_offset", L.bg_image_offset)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, name
    # only the radial and azimuthal terms the six reflections can see are nonzero, but c_00 always is
    assert float(L.bg_coef.grad[0, 0].abs()) > 0


def test_the_parent_prior_is_unchanged():
    cfg = _load("mos010_default_kmix_base_cheb40")
    loss = construct_integrator(cfg).loss
    conc, rate = loss._background_prior(None, {}, "cpu")
    assert float(conc) == 1.0 and float(rate) == 1.0
    assert math.isclose(float(conc / rate), 1.0)
