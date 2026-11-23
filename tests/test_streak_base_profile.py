"""`streak_base_profile`: built through the factory, trained everywhere, and the base is what it claims.

The class exists because `geometric_basis_profile` on mos010 reached a state
with no likelihood gradient (a one-hot canonical softmax) and a stretch
pinned on a hard clamp. These tests pin the properties the replacement was
written to have: every trainable parameter gets gradient at initialisation,
a closed gate returns the streak base for any encoder output, the base's
second moments are its extents, and the base rotates exactly with the
detector azimuth.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from integrator.registry import REGISTRY
from integrator.utils.factory_utils import construct_integrator

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "sim_truth"
B, SIDE = 6, 25
P = SIDE * SIDE
BEAM = (2074.5, 2181.5)


def _load(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    path = CFG / f"{name}.yaml"
    if not path.is_file():
        pytest.skip(f"no {path}")
    return yaml.safe_load(path.read_text())


@pytest.fixture
def batch():
    torch.manual_seed(0)
    x = torch.tensor([600.0, 3500.0, 2074.5, 900.0, 3000.0, 1500.0])
    y = torch.tensor([700.0, 900.0, 3600.0, 3400.0, 2181.5, 1200.0])
    counts = torch.poisson(torch.full((B, P), 20.0))
    counts[:, P // 2] += 400.0  # a bright centre pixel, so the gate opens on some rows
    metadata = {
        "group_label": torch.zeros(B, dtype=torch.long),
        "xyzcal.px.0": x,
        "xyzcal.px.1": y,
        "bbox.0": torch.floor(x) - SIDE // 2,
        "bbox.2": torch.floor(y) - SIDE // 2,
        "d": torch.full((B,), 2.0),
        "wavelength": torch.full((B,), 1.1),
    }
    return counts, counts.clone().unsqueeze(-1), torch.ones(B, P, 1), metadata


def _surrogate(**kw):
    cls = REGISTRY["surrogates"]["streak_base_profile"]
    args = dict(input_dim=32, latent_dim=12, output_dim=P, beam_center=BEAM,
                profile_shape=(1, SIDE, SIDE), lambda_min=1.01, lambda_max=1.19)
    args.update(kw)
    return cls(**args)


def test_is_registered():
    assert "streak_base_profile" in REGISTRY["surrogates"]


@pytest.mark.parametrize("name", ["mos010_default_streak", "mos010_default_streak_nogate"])
def test_every_profile_parameter_receives_gradient(name, batch):
    """Through `construct_integrator`, a forward and a backward reach every
    trainable parameter of the profile path at initialisation."""
    model = construct_integrator(_load(name)).train()
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    out["forward_out"]["rates"].sum().backward()
    dead = {
        n for n, p in model.named_parameters()
        if p.requires_grad and (n.startswith("surrogates.qp") or n.startswith("encoders.profile"))
        and (p.grad is None or float(p.grad.abs().sum()) == 0)
    }
    assert not dead, sorted(dead)


def test_without_residual_the_residual_path_is_frozen(batch):
    model = construct_integrator(_load("mos010_default_streak_nores")).train()
    qp = model.surrogates["qp"]
    assert not qp.residual
    assert not any(p.requires_grad for p in qp.loc_head.parameters())
    assert not qp.decoder.weight.requires_grad
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    out["forward_out"]["rates"].sum().backward()
    for n in ("log_extent0", "beta_raw"):
        assert getattr(qp, n).grad is not None and float(getattr(qp, n).grad.abs().sum()) > 0


def test_closed_gate_returns_the_base_for_any_encoder_output(batch):
    """gamma = 0 (flat counts, no signal) makes the profile the streak base,
    whatever the encoder says."""
    q = _surrogate().eval()
    _, _, _, metadata = batch
    flat = torch.full((B, P), 20.0)
    phi, log_s, shift = q.geometry(metadata, torch.float32, "cpu")
    g = q.base(phi, log_s, shift, None)
    gamma, _ = q.gate(g, flat, None)
    assert torch.all(gamma == 0)
    ref = torch.softmax(torch.log(g + q.base_floor / P), -1)
    for _ in range(3):
        out = q(torch.randn(B, 32) * 10, metadata=metadata, counts=flat)
        torch.testing.assert_close(out.mean_profile, ref, atol=1e-6, rtol=0)


@pytest.mark.parametrize("beta", [1.2, 2.0, 3.0])
def test_base_second_moments_are_the_extents(beta):
    """On a box large enough to hold it, the base's sds along and across the
    streak equal exp(log s): no other quantity carries its size."""
    side = 61
    q = _surrogate(output_dim=side * side, profile_shape=(1, side, side), init_beta=(beta, beta),
                   quad=5)
    phi = torch.tensor([0.0])
    log_s = torch.log(torch.tensor([[3.0, 1.5]]))
    g = q.base(phi, log_s, torch.zeros(1, 2), None).reshape(side, side)
    yy, xx = torch.meshgrid(torch.arange(side) - (side - 1) / 2, torch.arange(side) - (side - 1) / 2,
                            indexing="ij")
    sd_u = math.sqrt(float((g * xx**2).sum()))  # phi = 0: along = +x
    sd_v = math.sqrt(float((g * yy**2).sum()))
    # the pixel integral adds 1/12 px^2 of variance to each axis
    assert abs(math.sqrt(sd_u**2 - 1 / 12) - 3.0) < 0.03
    assert abs(math.sqrt(sd_v**2 - 1 / 12) - 1.5) < 0.03


def test_base_rotates_exactly_with_the_azimuth():
    """A quarter turn of the azimuth is a quarter turn of the image."""
    q = _surrogate()
    log_s = torch.log(torch.tensor([[4.0, 1.2]]))
    g0 = q.base(torch.tensor([0.3]), log_s, torch.zeros(1, 2), None).reshape(SIDE, SIDE)
    g1 = q.base(torch.tensor([0.3 + math.pi / 2]), log_s, torch.zeros(1, 2), None).reshape(SIDE, SIDE)
    # +x -> +y in (column, row): a counter-clockwise turn in image coordinates
    torch.testing.assert_close(g1, torch.rot90(g0, k=-1, dims=(0, 1)), atol=1e-6, rtol=0)


def test_extents_stay_inside_their_bound():
    q = _surrogate(max_log_extent=1.2)
    with torch.no_grad():
        q.extent_head[-1].bias.fill_(1e6)
    md = {"xyzcal.px.0": torch.tensor([100.0]), "xyzcal.px.1": torch.tensor([100.0]),
          "wavelength": torch.tensor([1.1])}
    _, log_s, _ = q.geometry(md, torch.float32, "cpu")
    assert torch.all(log_s <= q.log_extent0 + 1.2 + 1e-5)
