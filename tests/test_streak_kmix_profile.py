"""`streak_kmix_profile`: built through the factory, every setting from the window, trained where it should be.

The class exists to carry no number taken from a true profile. These tests pin:
the T0 configs build and train their shape parameters with the profile encoder
cut out; with a residual every profile parameter receives gradient; the
starting widths are the window grid times `init_grid_scale`, below the cap;
each component is normalised on the window, so the weights are the shares; and
the base matches the exact pixel integral of the same mixture.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
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
    cls = REGISTRY["surrogates"]["streak_kmix_profile"]
    args = dict(input_dim=32, latent_dim=12, output_dim=P, beam_center=BEAM,
                profile_shape=(1, SIDE, SIDE), lambda_min=1.01, lambda_max=1.19)
    args.update(kw)
    return cls(**args)


def _dead(model, prefixes):
    return sorted(
        n for n, p in model.named_parameters()
        if p.requires_grad and n.startswith(prefixes)
        and (p.grad is None or float(p.grad.abs().sum()) == 0)
    )


def test_is_registered():
    assert "streak_kmix_profile" in REGISTRY["surrogates"]


@pytest.mark.parametrize("name", ["mos010_default_kmix_base", "mos010_default_kmix_base_grid05",
                                  "mos010_default_kmix_base_grid2"])
def test_t0_trains_the_shape_and_cuts_out_the_encoder(name, batch):
    model = construct_integrator(_load(name)).train()
    qp = model.surrogates["qp"]
    assert qp.residual == "none"
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    out["forward_out"]["rates"].sum().backward()
    assert not _dead(model, ("surrogates.qp",))
    assert qp.theta0.grad is not None and float(qp.theta0.grad.abs().sum()) > 0
    enc = [p.grad for p in model.encoders["profile"].parameters() if p.requires_grad]
    assert all(g is None or float(g.abs().sum()) == 0 for g in enc)


@pytest.mark.parametrize("over", [dict(residual="detector"), dict(residual="detector", shape_posterior="amortized"),
                                  dict(residual="detector", n_components=5)])
def test_with_residual_every_profile_parameter_receives_gradient(over, batch):
    cfg = _load("mos010_default_kmix_base")
    cfg["surrogates"]["qp"]["args"].update(over)
    model = construct_integrator(cfg).train()
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    out["forward_out"]["rates"].sum().backward()
    assert not _dead(model, ("surrogates.qp", "encoders.profile"))


@pytest.mark.parametrize("gs", [0.5, 1.0, 2.0])
def test_starting_widths_are_the_window_grid(gs):
    q = _surrogate(residual="none", init_grid_scale=gs)
    a, c, w, sh = q.unpack(q.theta0.detach())
    exp_a = np.minimum(np.geomspace(1.0, SIDE / 4, 3) * gs, 0.9 * q.cap)
    exp_c = np.minimum(np.geomspace(0.3, SIDE / 8, 3) * gs, 0.9 * q.cap)
    np.testing.assert_allclose(a.numpy(), exp_a, rtol=1e-4)
    np.testing.assert_allclose(c.numpy(), exp_c, rtol=1e-4)
    np.testing.assert_allclose(w.numpy(), np.full(3, 1 / 3), rtol=1e-6)
    assert float(sh.abs().max()) == 0.0
    assert float(a.max()) < q.cap and float(c.max()) < q.cap


@pytest.mark.parametrize("phi,offset", [(0.0, (0.0, 0.0)), (2.2, (0.3, -0.2))])
def test_base_matches_the_pixel_integral(phi, offset):
    side = 61
    q = _surrogate(output_dim=side * side, profile_shape=(1, side, side), residual="none")
    K = q.K
    a = torch.tensor([1.9, 3.0, 5.7])
    c = torch.tensor([0.2, 0.6, 1.1])
    wts = torch.tensor([0.4, 0.25, 0.35])

    def inv(y):
        return torch.log(torch.expm1(q.cap * torch.atanh(y / q.cap)))

    th = torch.cat([inv(a), inv(c), wts.log(), torch.atanh(torch.tensor(offset) / q.max_shift) * q.max_shift])[None]
    a_, c_, w_, sh_ = q.unpack(th)
    torch.testing.assert_close(w_[0], wts, atol=1e-6, rtol=0)
    torch.testing.assert_close(sh_[0], torch.tensor(offset), atol=1e-5, rtol=0)
    ph = torch.tensor([phi])
    with torch.no_grad():
        g = q.base(ph, th, torch.zeros(1, 2), None)[0]
        sub = (torch.arange(15.0) + 0.5) / 15 - 0.5
        oy, ox = torch.meshgrid(sub, sub, indexing="ij")
        cs, sn = math.cos(phi), math.sin(phi)
        X, Y = q._px[:, None] + ox.reshape(1, -1), q._py[:, None] + oy.reshape(1, -1)
        U = cs * X + sn * Y - offset[0]
        V = -sn * X + cs * Y - offset[1]
        exact = torch.zeros_like(g)
        for k in range(K):
            e = (torch.exp(-0.5 * (U / a[k]) ** 2 - 0.5 * (V / c[k]) ** 2)).mean(1)
            exact = exact + wts[k] * e / e.sum()
    cos = float((g * exact).sum() / (g.norm() * exact.norm()))
    assert cos >= 0.999, cos
    torch.testing.assert_close(g.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)
