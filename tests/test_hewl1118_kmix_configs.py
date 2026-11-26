"""The hewl1118 configs with the mos010 streak profile models build through the factory and train."""

import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

ROOT = pathlib.Path(__file__).resolve().parents[1]
B, SIDE = 6, 25
P = SIDE * SIDE


@pytest.fixture
def batch():
    torch.manual_seed(0)
    x = torch.tensor([600.0, 3500.0, 1990.0, 900.0, 3000.0, 1500.0])
    y = torch.tensor([700.0, 900.0, 3600.0, 3400.0, 1973.0, 1200.0])
    counts = torch.poisson(torch.full((B, P), 10.0))
    counts[:, P // 2] += 300.0
    metadata = {"group_label": torch.zeros(B, dtype=torch.long), "xyzcal.px.0": x, "xyzcal.px.1": y,
                "bbox.0": torch.floor(x) - SIDE // 2, "bbox.2": torch.floor(y) - SIDE // 2,
                "d": torch.full((B,), 2.0), "wavelength": torch.full((B,), 1.1)}
    return counts, counts.clone().unsqueeze(-1), torch.ones(B, P, 1), metadata


@pytest.mark.parametrize("name", ["hewl1118_kmix_base", "hewl1118_kmix_h128_grid05", "hewl1118_kmixv2_post"])
def test_hewl1118_streak_configs_forward_and_train_the_profile(name, batch):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((ROOT / "configs" / "poly" / f"{name}.yaml").read_text())
    assert cfg["integrator"]["args"]["gate"] == "responsibility"
    model = construct_integrator(cfg).train()
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    rates = out["forward_out"]["rates"]
    assert torch.isfinite(rates).all()
    rates.sum().backward()
    qp = model.surrogates["qp"]
    dead = [n for n, p in qp.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    assert not dead, dead
