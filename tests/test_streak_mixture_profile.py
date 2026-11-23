"""`streak_mixture_profile`: built through the factory, trained everywhere it should be, and the base is what it claims.

The class replaces the one-scale base of `streak_base_profile` with a core and
a halo whose seven shape parameters come from the detector geometry. These
tests pin: every trainable profile parameter receives gradient in the M1
configuration; in M0 (no residual) the profile encoder is cut out and the
shape head still trains; the base matches the exact pixel integral of the
core and halo, with along-streak variance w a1^2 + (1-w) a2^2 + 1/12; a zero shape offset returns the metadata
base for any encoder output; the shape KL is zero at initialisation.
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
    counts[:, P // 2] += 400.0
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
    cls = REGISTRY["surrogates"]["streak_mixture_profile"]
    args = dict(input_dim=32, latent_dim=12, output_dim=P, beam_center=BEAM,
                profile_shape=(1, SIDE, SIDE), lambda_min=1.01, lambda_max=1.19)
    args.update(kw)
    return cls(**args)


def _zvec(a1, a2, c1, c2, w, du=0.0, dv=0.0):
    isp = lambda y: math.log(math.expm1(y))  # noqa: E731
    return torch.tensor([[math.log(a1), isp(a2 / a1 - 1), math.log(c1), isp(c2 / c1 - 1),
                          math.log((w - 0.02) / (0.98 - w)), du, dv]])


def _dead(model, prefixes):
    return sorted(
        n for n, p in model.named_parameters()
        if p.requires_grad and n.startswith(prefixes)
        and (p.grad is None or float(p.grad.abs().sum()) == 0)
    )


def test_is_registered():
    assert "streak_mixture_profile" in REGISTRY["surrogates"]


def test_m1_every_profile_parameter_receives_gradient(batch):
    model = construct_integrator(_load("mos010_default_mixture")).train()
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    out["forward_out"]["rates"].sum().backward()
    assert not _dead(model, ("surrogates.qp", "encoders.profile"))


def test_m0_shape_head_trains_and_the_encoder_is_cut_out(batch):
    model = construct_integrator(_load("mos010_default_mixture_base")).train()
    qp = model.surrogates["qp"]
    assert qp.residual == "none" and qp.shape_posterior == "none"
    assert not qp.decoder.weight.requires_grad
    assert not any(p.requires_grad for p in qp.loc_head.parameters())
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    out["forward_out"]["rates"].sum().backward()
    assert not _dead(model, ("surrogates.qp",))
    enc = [p.grad for p in model.encoders["profile"].parameters() if p.requires_grad]
    assert all(g is None or float(g.abs().sum()) == 0 for g in enc)


@pytest.mark.parametrize("shape,phi,offset", [((1.9, 5.7, 0.20, 1.05, 0.38), 0.0, (0.0, 0.0)),
                                              ((2.5, 5.5, 0.18, 0.73, 0.52), 2.2, (0.3, -0.2)),
                                              ((1.6, 5.7, 0.30, 1.20, 0.30), 0.9, (0.0, 0.4))])
def test_base_matches_the_pixel_integral(shape, phi, offset):
    """Against a 15 x 15 sub-pixel quadrature of the same core and halo: cos >= 0.999,
    and the along-streak variance (every component wider than a pixel) equal to
    w a1^2 + (1-w) a2^2 + 1/12. Across, the 0.2 px core has no pixel-centre moment
    equal to its continuous one, so only the quadrature is the reference."""
    side = 61
    q = _surrogate(output_dim=side * side, profile_shape=(1, side, side),
                   shape_posterior="none", residual="none")
    a1, a2, c1, c2, w = shape
    z = _zvec(*shape, *offset)
    ph = torch.tensor([phi])
    with torch.no_grad():
        g = q.base(ph, z, torch.zeros(1, 2), None)[0]
        u, v = q.streak_coords(ph, z[:, 5:7])
        u, v = u[0], v[0]
        sub = (torch.arange(15.0) + 0.5) / 15 - 0.5
        oy, ox = torch.meshgrid(sub, sub, indexing="ij")
        c, s_ = math.cos(phi), math.sin(phi)
        X, Y = q._px[:, None] + ox.reshape(1, -1), q._py[:, None] + oy.reshape(1, -1)
        U = c * X + s_ * Y - offset[0]
        V = -s_ * X + c * Y - offset[1]

        def n(x, sd):
            return torch.exp(-0.5 * (x / sd) ** 2) / sd

        def part(a, cc):
            e = (n(U, a) * n(V, cc)).mean(1)
            return e / e.sum()

        exact = w * part(a1, c1) + (1 - w) * part(a2, c2)
    cos = float((g * exact).sum() / (g.norm() * exact.norm()))
    assert cos >= 0.999, cos
    mu = (g * u).sum()
    vu = float((g * (u - mu) ** 2).sum())
    assert abs(vu - (w * a1**2 + (1 - w) * a2**2 + 1 / 12)) < 0.02


def test_zero_offset_returns_the_metadata_base_and_zero_kl(batch):
    q = _surrogate(shape_posterior="amortized", residual="none").eval()
    _, _, mask, metadata = batch
    m = mask.squeeze(-1)
    with torch.no_grad():
        q.delta_head.weight.zero_()
        q.zscale_head.weight.zero_()
        p1 = q(torch.randn(B, 32), metadata=metadata, mask=m)
        p2 = q(10 * torch.randn(B, 32), metadata=metadata, mask=m)
    torch.testing.assert_close(p1.mean_profile, p2.mean_profile, atol=1e-6, rtol=0)
    assert float(p1.kl.abs().max()) < 1e-4
    torch.testing.assert_close(p1.mean_profile.sum(-1), torch.ones(B), atol=1e-5, rtol=0)
