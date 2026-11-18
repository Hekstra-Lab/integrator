"""The profile encoder's three reduction heads.

`flatten` is the original: one dense layer over the whole conv map, which on a
41x41 shoebox is 32 x 23,104 = 739,328 parameters, 97.5% of the encoder.
`pool` reduces once more first; `moments` replaces the dense layer with the
spatial moments of each feature map.
"""

import copy

import pytest
import torch
import yaml

from integrator.utils.factory_utils import construct_integrator

CFG = "paper_figures/profile_model/data/config_log.yaml"


def _build(head):
    cfg = yaml.safe_load(open(CFG))
    for e in cfg["encoders"]:
        if e["name"] == "profile_encoder":
            e.setdefault("args", {})["head"] = head
    return construct_integrator(cfg)


@pytest.mark.parametrize("head,expected_fc_in", [("flatten", 23104),
                                                 ("pool", 6400),
                                                 ("moments", 384)])
def test_head_shapes_and_size(head, expected_fc_in):
    m = _build(head)
    e = m.encoders["profile"]
    assert e.fc.in_features == expected_fc_in
    out = e(torch.randn(3, 1, 41, 41))
    assert out.shape == (3, e.encoder_out)
    assert torch.isfinite(out).all()


def test_flatten_head_is_the_default():
    """Omitting `head` must leave the encoder bit-identical to before."""
    a = _build("flatten")
    cfg = yaml.safe_load(open(CFG))          # no `head` key at all
    b = construct_integrator(cfg)
    assert a.encoders["profile"].fc.in_features == \
        b.encoders["profile"].fc.in_features
    assert b.encoders["profile"].head == "flatten"


def test_moments_survive_a_degenerate_shoebox():
    """A feature map with all its mass on one pixel has zero spatial spread.

    `sqrt` has infinite derivative at zero, so taking the spread as
    `clamp_min(0).sqrt()` puts a NaN in the backward pass the first time a
    shoebox does this -- which is what killed the first `moments` run at step
    one. The epsilon goes inside the sqrt.
    """
    m = _build("moments")
    e = m.encoders["profile"]
    x = torch.zeros(2, 1, 41, 41)
    x[:, :, 20, 20] = 50.0                   # all mass on a single pixel
    e.zero_grad()
    out = e(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    grads = torch.cat([p.grad.reshape(-1) for p in e.parameters()
                       if p.grad is not None])
    assert torch.isfinite(grads).all(), "zero spread must not give a NaN gradient"


def test_coord_channels_run_under_svae():
    """`profile_coord_channels` widens conv1, so the forward must widen too.

    The factory sets `conv1.in_channels = 3` from the config, but the SVAE
    forward reshaped the shoebox to one channel unconditionally, so the config
    built a model that raised at the first batch.
    """
    cfg = yaml.safe_load(open(CFG))
    cfg["integrator"]["args"]["profile_coord_channels"] = True
    cfg["integrator"]["args"]["beam_center"] = \
        cfg["surrogates"]["qp"]["args"]["beam_center"]
    m = construct_integrator(cfg)
    assert m.encoders["profile"].conv1.in_channels == 3
    b, p = 3, 1681
    counts = torch.rand(b, p) * 50
    sb = torch.randn(b, p, 1)
    mask = torch.ones(b, p, 1)
    md = {"xyzcal.px.0": torch.rand(b) * 4000,
          "xyzcal.px.1": torch.rand(b) * 4000,
          "wavelength": torch.rand(b) * 0.2 + 1.0,
          "d": torch.rand(b) * 8 + 2,
          "group_label": torch.zeros(b, dtype=torch.long)}
    m.eval()
    with torch.no_grad():
        out = m(counts, sb, mask, md)["forward_out"]
    assert torch.isfinite(out["qi_mean"]).all()
