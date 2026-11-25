"""`optimizer.frozen_params`: named parameters frozen after the warm start, default off."""

from __future__ import annotations

import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth" / "mos010_default_kmixv2_post.yaml"
PRIOR = ["surrogates.qp.theta0", "surrogates.qp.eta_sd_raw", "surrogates.qp.shape_head", "loss."]


def _cfg(**opt):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CFG.read_text())
    cfg.setdefault("optimizer", {}).update(opt)
    return cfg


def test_default_freezes_nothing():
    torch.manual_seed(0)
    m = construct_integrator(_cfg())
    assert m.surrogates["qp"].theta0.requires_grad and m.loss.raw_B.requires_grad


def test_frozen_keep_the_checkpoint_values_and_the_rest_trains(tmp_path):
    torch.manual_seed(0)
    src = construct_integrator(_cfg())
    with torch.no_grad():
        src.surrogates["qp"].theta0.add_(0.5)
    ck = tmp_path / "src.ckpt"
    torch.save({"state_dict": src.state_dict()}, ck)
    cfg = _cfg(frozen_params=PRIOR)
    cfg["integrator"]["args"]["init_from_checkpoint"] = str(ck)
    torch.manual_seed(1)
    m = construct_integrator(cfg)
    assert torch.equal(m.surrogates["qp"].theta0, src.surrogates["qp"].theta0)
    for n, p in m.named_parameters():
        frozen = any(n.startswith(x) for x in PRIOR)
        assert p.requires_grad != frozen or n.startswith("surrogates.qp.") and not p.requires_grad, n
    names = {id(p) for g in m._build_optimizer().param_groups for p in g["params"]}
    assert not any(id(p) in names for n, p in m.named_parameters() if any(n.startswith(x) for x in PRIOR))
    assert m.surrogates["qp"].eta_delta.weight.requires_grad


def test_an_unknown_prefix_raises():
    with pytest.raises(ValueError, match="frozen_params"):
        construct_integrator(_cfg(frozen_params=["surrogates.qp.no_such"]))
