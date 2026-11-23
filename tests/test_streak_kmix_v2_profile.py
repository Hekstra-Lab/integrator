"""`streak_kmix_v2_profile`: tilt and per-reflection shape posterior on top of `streak_kmix_profile`.

Pins: every arm builds through the factory and trains every profile parameter
it should (the shape posterior reaches the profile encoder even with no
residual); at default keys the class is the parent exactly; a zero posterior
offset is the metadata base at zero KL for any encoder output; the learned
tilt rotates the base exactly; the shape sd stays inside its bound; and the
data module's `split_seed` reproduces its train/validation split.
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
KW = dict(input_dim=32, latent_dim=12, output_dim=P, beam_center=BEAM, profile_shape=(1, SIDE, SIDE),
          lambda_min=1.01, lambda_max=1.19)


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


def _dead(model, prefixes):
    return sorted(n for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith(prefixes)
                  and (p.grad is None or float(p.grad.abs().sum()) == 0))


def test_is_registered():
    assert "streak_kmix_v2_profile" in REGISTRY["surrogates"]


@pytest.mark.parametrize("name,encoder_trains", [
    ("mos010_default_kmix_h128", False), ("mos010_default_kmix_h128_grid05", False),
    ("mos010_default_kmixv2_post", True), ("mos010_default_kmixv2_post_fixed", True),
    ("mos010_default_kmixv2_post_det", True), ("mos010_default_kmixv2_tilt", False),
    ("mos010_default_kmixv2_post_grid05", True), ("mos010_default_kmixv2_post_grid05_k7", True)])
def test_arm_builds_and_trains(name, encoder_trains, batch):
    model = construct_integrator(_load(name)).train()
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    (out["forward_out"]["rates"].sum() + out["qp"].kl.sum()).backward()
    assert not _dead(model, ("surrogates.qp",))
    enc = [p.grad for p in model.encoders["profile"].parameters() if p.requires_grad]
    live = any(g is not None and float(g.abs().sum()) > 0 for g in enc)
    assert live == encoder_trains


def test_defaults_are_the_parent_exactly(batch):
    torch.manual_seed(3)
    v1 = REGISTRY["surrogates"]["streak_kmix_profile"](**KW, residual="none").eval()
    v2 = REGISTRY["surrogates"]["streak_kmix_v2_profile"](**KW).eval()
    v2.load_state_dict(v1.state_dict())
    _, _, mask, md = batch
    x = torch.randn(B, 32)
    with torch.no_grad():
        a = v1(x, metadata=md, mask=mask.squeeze(-1)).mean_profile
        b = v2(x, metadata=md, mask=mask.squeeze(-1)).mean_profile
    assert torch.equal(a, b)


def test_zero_offset_is_the_metadata_base_at_zero_kl(batch):
    q = REGISTRY["surrogates"]["streak_kmix_v2_profile"](**KW, shape_posterior="amortized").eval()
    ref = REGISTRY["surrogates"]["streak_kmix_profile"](**KW, residual="none").eval()
    sd = {k: v for k, v in q.state_dict().items() if k in ref.state_dict()}
    ref.load_state_dict(sd)
    _, _, mask, md = batch
    m = mask.squeeze(-1)
    with torch.no_grad():
        q.eta_delta.weight.zero_()
        q.eta_scale.weight.zero_()
        p1 = q(torch.randn(B, 32), metadata=md, mask=m)
        p2 = q(10 * torch.randn(B, 32), metadata=md, mask=m)
        p0 = ref(torch.randn(B, 32), metadata=md, mask=m)
    torch.testing.assert_close(p1.mean_profile, p2.mean_profile, atol=1e-6, rtol=0)
    torch.testing.assert_close(p1.mean_profile, p0.mean_profile, atol=1e-5, rtol=0)
    assert float(p1.kl.abs().max()) < 1e-4


def test_tilt_rotates_the_base(batch):
    """A tilt of pi/2 on a centred base is a quarter turn of the image."""
    q = REGISTRY["surrogates"]["streak_kmix_v2_profile"](**KW, tilt="metadata").eval()
    th = q.theta0.detach()[None]
    fxy = torch.zeros(1, 2)
    phi = torch.tensor([0.3])
    u0, v0 = q.frame(phi, torch.zeros(1), th, fxy)
    u1, v1 = q.frame(phi, torch.full((1,), math.pi / 2), th, fxy)
    g0 = q.base_uv(u0, v0, th, None).reshape(SIDE, SIDE)
    g1 = q.base_uv(u1, v1, th, None).reshape(SIDE, SIDE)
    torch.testing.assert_close(g1, torch.rot90(g0, k=-1, dims=(0, 1)), atol=1e-6, rtol=0)


def test_shape_sd_stays_below_its_bound():
    q = REGISTRY["surrogates"]["streak_kmix_v2_profile"](**KW, shape_posterior="amortized")
    with torch.no_grad():
        q.eta_sd_raw.fill_(50.0)
    assert torch.all(q.eta_sd <= q.eta_sd_max)
    fixed = REGISTRY["surrogates"]["streak_kmix_v2_profile"](**KW, shape_posterior="amortized", shape_sd="fixed")
    assert not fixed.eta_sd_raw.requires_grad
    torch.testing.assert_close(fixed.eta_sd[:-1], torch.full((fixed.NE - 1,), 0.3), atol=1e-6, rtol=0)


def test_split_seed_draw_is_reproducible_and_none_is_unchanged():
    """The data module draws its permutation with torch.randperm(n, generator=gen): a seeded
    generator repeats the split, and gen = None draws from the global generator as before."""
    import inspect

    from integrator.data_loaders.poly_data_module import PolychromaticDataModule

    src = inspect.getsource(PolychromaticDataModule)
    assert "split_seed" in inspect.signature(PolychromaticDataModule.__init__).parameters
    assert "torch.randperm(len(train_val_idx), generator=gen)" in src

    def draw(seed):
        gen = None if seed is None else torch.Generator().manual_seed(seed)
        return torch.randperm(1000, generator=gen)[:200]

    assert torch.equal(draw(0), draw(0))
    assert not torch.equal(draw(0), draw(1))
    torch.manual_seed(5)
    a = torch.randperm(1000)[:200]
    torch.manual_seed(5)
    assert torch.equal(a, draw(None))


def test_rb_profile_mean_fits_i_rb_with_the_mean_profile(batch):
    from integrator.model.integrators.integrator_utils import _samples_first, rao_blackwellized_intensity

    cfg = _load("mos010_default_kmixv2_post_fixed")
    cfg["integrator"]["args"]["predict_keys"] = ["i_rb", "var_rb", "var_rb_cond", "var_rb_profile", "rb_gate"]
    torch.manual_seed(0)
    base = construct_integrator(cfg).eval()
    assert base.rb_profile == "samples"
    cfg["integrator"]["args"]["rb_profile"] = "mean"
    model = construct_integrator(cfg).eval()
    assert model.rb_profile == "mean"
    model.load_state_dict(base.state_dict())

    counts, shoebox, mask, metadata = batch
    torch.manual_seed(1)
    out_s = base.predict_step(batch, 0)
    torch.manual_seed(1)
    out_m = model.predict_step(batch, 0)
    torch.manual_seed(1)
    fo = model(counts, shoebox, mask, metadata)["forward_out"]
    zbg = _samples_first(fo["zbg"], counts.shape[0])
    pm = fo["qp_mean"].reshape(counts.shape[0], -1).unsqueeze(0).expand(zbg.shape[0], -1, -1)
    ref = rao_blackwellized_intensity(fo["counts"], fo["mask"], pm, zbg)
    assert torch.allclose(out_m["i_rb"], ref["i_rb"], rtol=1e-5, atol=1e-4)
    assert torch.allclose(out_m["var_rb_profile"], out_s["var_rb_profile"])
    assert torch.allclose(out_m["var_rb"], ref["var_rb_cond"] + out_s["rb_gate"] * out_s["var_rb_profile"], rtol=1e-5)
    assert not torch.allclose(out_m["i_rb"], out_s["i_rb"])
