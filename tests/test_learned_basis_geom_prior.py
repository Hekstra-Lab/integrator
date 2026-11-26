"""learned_basis_geom_prior_profile and gmm_geom_prior_profile: strictly learned profiles with a geometry-conditioned latent prior."""

import pathlib

import pytest
import torch

from integrator.registry import REGISTRY
from integrator.utils.factory_utils import construct_integrator

ROOT = pathlib.Path(__file__).resolve().parents[1]
B, SIDE = 6, 25
P = SIDE * SIDE


@pytest.fixture
def batch():
    torch.manual_seed(0)
    x = torch.tensor([600.0, 3500.0, 2000.0, 900.0, 3000.0, 1500.0])
    y = torch.tensor([700.0, 900.0, 3600.0, 3400.0, 2100.0, 1200.0])
    counts = torch.poisson(torch.full((B, P), 10.0))
    counts[:, P // 2] += 300.0
    md = {"group_label": torch.zeros(B, dtype=torch.long), "xyzcal.px.0": x, "xyzcal.px.1": y,
          "bbox.0": torch.floor(x) - SIDE // 2, "bbox.2": torch.floor(y) - SIDE // 2,
          "d": torch.full((B,), 2.0), "wavelength": torch.full((B,), 1.1)}
    return counts, counts.clone().unsqueeze(-1), torch.ones(B, P, 1), md


def test_starts_as_the_learned_basis(batch):
    torch.manual_seed(1)
    geo = REGISTRY["surrogates"]["learned_basis_geom_prior_profile"](input_dim=32, latent_dim=12, output_dim=P,
                                                                      beam_center=(2074.5, 2181.5), lambda_min=1.01, lambda_max=1.19)
    base = REGISTRY["surrogates"]["learned_basis_profile"](input_dim=32, latent_dim=12, output_dim=P)
    base.load_state_dict({k: v for k, v in geo.state_dict().items() if not k.startswith("prior_mean_head")})
    x = torch.randn(B, 32)
    md = batch[3]
    torch.manual_seed(2)
    a = geo(x, mc_samples=3, metadata=md)
    torch.manual_seed(2)
    b = base(x, mc_samples=3)
    assert torch.allclose(a.mean_profile, b.mean_profile) and torch.allclose(a.zp, b.zp)
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(b.loc, b.scale),
                                            torch.distributions.Normal(torch.zeros_like(b.loc), 3.0)).sum(-1)
    assert torch.allclose(a.kl, ref, atol=1e-5)


@pytest.mark.parametrize("path", ["configs/sim_truth/mos010_default_lbgeom.yaml", "configs/poly/hewl1118_lbgeom.yaml"])
def test_configs_build_and_train_the_prior_mean(path, batch):
    yaml = pytest.importorskip("yaml")
    model = construct_integrator(yaml.safe_load((ROOT / path).read_text())).train()
    counts, shoebox, mask, md = batch
    out = model(counts, shoebox, mask, md)
    loss = out["forward_out"]["rates"].sum() + out["qp"].kl.sum()
    loss.backward()
    qp = model.surrogates["qp"]
    dead = [n for n, p in qp.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    # at initialisation the MLP's last layer is zero, so only it (and not the layers before) receives gradient
    assert all(n.startswith("prior_mean_head.net.0") or n.startswith("prior_mean_head.net.2") for n in dead), dead
    assert float(qp.prior_mean_head.net[-1].weight.grad.abs().sum()) > 0


def test_gmm_starts_isotropic_and_normalised(batch):
    q = REGISTRY["surrogates"]["gmm_geom_prior_profile"](input_dim=32, output_dim=P, beam_center=(2074.5, 2181.5),
                                                          lambda_min=1.01, lambda_max=1.19)
    out = q(torch.randn(B, 32), mc_samples=2, metadata=batch[3], mask=torch.ones(B, P, 1))
    assert out.zp.shape == (2, B, P) and torch.allclose(out.mean_profile.sum(-1), torch.ones(B))
    _, sds, _, w = q.components(out.loc)
    assert torch.allclose(sds[..., 0], sds[..., 1]) and torch.allclose(w, torch.full_like(w, 1 / 3))


@pytest.mark.parametrize("path", ["configs/sim_truth/mos010_default_gmmgeom.yaml", "configs/poly/hewl1118_gmmgeom.yaml"])
def test_gmm_configs_build_and_every_profile_parameter_trains(path, batch):
    yaml = pytest.importorskip("yaml")
    model = construct_integrator(yaml.safe_load((ROOT / path).read_text())).train()
    counts, shoebox, mask, md = batch
    opt = torch.optim.Adam(model.surrogates["qp"].parameters(), lr=1e-2)
    # two zero-initialised layers in series (the decoder's last, then the prior MLP's last): the
    # gradient reaches the prior MLP's first layers on the third step
    for _ in range(3):
        opt.zero_grad()
        out = model(counts, shoebox, mask, md)
        (out["forward_out"]["rates"].sum() + out["qp"].kl.sum()).backward()
        opt.step()
    qp = model.surrogates["qp"]
    dead = [n for n, p in qp.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    assert not dead, dead
