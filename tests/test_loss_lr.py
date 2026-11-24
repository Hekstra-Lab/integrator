"""`optimizer.loss_lr`: the loss's parameters in a group of their own, default off."""

from __future__ import annotations

import pathlib

import pytest
import torch

from integrator.utils.factory_utils import construct_integrator

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth" / "mos010_default_kmixv2_post.yaml"


def _groups(**opt):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(CFG.read_text())
    cfg.setdefault("optimizer", {}).update(opt)
    torch.manual_seed(0)
    m = construct_integrator(cfg)
    names = {id(p): n for n, p in m.named_parameters()}
    return [(g["lr"], g["weight_decay"], sorted(names[id(p)] for p in g["params"]))
            for g in m._build_optimizer().param_groups]


def test_default_groups_are_unchanged():
    g = _groups()
    assert all(lr == 1e-3 for lr, _, _ in g)
    assert any(n.startswith("loss.") for _, _, ns in g for n in ns)


def test_loss_lr_moves_every_loss_parameter_and_nothing_else():
    base = _groups()
    g = _groups(loss_lr=1e-2)
    loss_group = [x for x in g if x[0] == 1e-2]
    assert len(loss_group) == 1
    _, wd, names = loss_group[0]
    assert wd == 0.0 and names and all(n.startswith("loss.") for n in names)
    every_loss = sorted(n for _, _, ns in base for n in ns if n.startswith("loss."))
    assert names == every_loss
    rest = sorted(n for lr, _, ns in g if lr == 1e-3 for n in ns)
    assert not any(n.startswith("loss.") for n in rest)
    assert len(rest) + len(names) == sum(len(ns) for _, _, ns in base)


def test_loss_lr_must_be_positive():
    with pytest.raises(ValueError, match="loss_lr"):
        _groups(loss_lr=0.0)
