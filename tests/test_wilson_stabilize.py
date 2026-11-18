"""Tests for the DIALS-free Wilson G/B scale stabilization."""

import math

import torch
from torch.distributions import Dirichlet, Gamma

from integrator.model.loss.monochromatic_wilson_loss import (
    MonochromaticWilsonLoss,
)

G_TRUE, B_TRUE = 150.0, 20.0


def _synthetic_counts(b=240, p=25, seed=0):
    """Reflections whose summed counts encode a known Wilson plot."""
    g = torch.Generator().manual_seed(seed)
    d = 1.5 + 4.5 * torch.rand(b, generator=g)
    s2 = 1.0 / (4.0 * d.pow(2))
    i_true = G_TRUE * torch.exp(-2.0 * B_TRUE * s2)
    counts = torch.ones(b, p)  # unit background per pixel
    counts[:, 0] = (counts[:, 0] + i_true).round()  # integer signal in one pixel
    mask = torch.ones(b, p, 1)
    metadata = {"d": d}
    return counts, mask, metadata, s2


def _forward(loss, counts, mask, metadata, seed=0):
    torch.manual_seed(seed)
    b, p = counts.shape
    rate = torch.rand(b, 4, p) * 5 + 0.1
    qp = Dirichlet(torch.ones(b, p))
    qi = Gamma(torch.rand(b) + 1.0, torch.rand(b) + 1.0)
    qbg = Gamma(torch.rand(b) + 1.0, torch.rand(b) + 1.0)
    group_labels = torch.zeros(b, dtype=torch.long)
    return loss(
        rate=rate, counts=counts, qp=qp, qi=qi, qbg=qbg, mask=mask,
        group_labels=group_labels, metadata=metadata,
    )


def test_legacy_default_unchanged_and_no_new_state():
    """stabilize_scale off: softplus params, and zero added state_dict keys."""
    loss = MonochromaticWilsonLoss(bg_rate=1.0, bg_concentration=1.0)
    assert not loss.stabilize_scale
    assert not hasattr(loss, "scale_initialized")
    keys = set(loss.state_dict())
    # Asserts what this test is about: stabilize_scale adds no state. It used
    # to require the state_dict be EXACTLY {raw_B, raw_G}, which also forbade
    # state from unrelated features -- intensity_kl_bins belongs to the
    # intensity-KL calibration knob and is persistent so existing checkpoints
    # keep loading. Making that buffer conditional to satisfy the old wording
    # broke every checkpoint already written.
    assert {"raw_B", "raw_G"} <= keys
    assert keys - {"raw_B", "raw_G", "intensity_kl_bins"} == set()
    torch.testing.assert_close(loss.get_G(), torch.nn.functional.softplus(loss.raw_G))
    torch.testing.assert_close(
        loss.get_B(), torch.nn.functional.softplus(loss.raw_B) + loss.b_min
    )


def test_stabilize_parameterization():
    loss = MonochromaticWilsonLoss(
        stabilize_scale=True, b_min=1.0, init_B=30.0,
        bg_rate=1.0, bg_concentration=1.0,
    )
    # G is log-space
    torch.testing.assert_close(loss.get_G(), torch.exp(loss.raw_G))
    # B = b_min + softplus(raw_B): floored below, unbounded above, init at init_B
    B = loss.get_B()
    assert float(B) > loss.b_min
    assert abs(float(B) - 30.0) < 1e-3
    assert not bool(loss.scale_initialized)


def test_B_floored_at_b_min_no_upper_cap():
    loss = MonochromaticWilsonLoss(
        stabilize_scale=True, b_min=1.0, bg_rate=1.0, bg_concentration=1.0,
    )
    with torch.no_grad():
        loss.raw_B.copy_(torch.tensor(-1e4))  # drive toward the floor
    B = loss.get_B()
    assert torch.isfinite(B)
    assert abs(float(B) - loss.b_min) < 1e-3  # softplus floor at b_min, not 0
    with torch.no_grad():
        loss.raw_B.copy_(torch.tensor(500.0))  # large B: no upper cap
    assert float(loss.get_B()) > 400.0


def test_lazy_count_init_recovers_scale():
    """First forward fits G from raw counts; recovered scale is near truth."""
    loss = MonochromaticWilsonLoss(
        stabilize_scale=True, b_min=1.0, init_B=30.0,
        wilson_init_bins=15, bg_rate=1.0, bg_concentration=1.0,
    )
    counts, mask, metadata, _ = _synthetic_counts()
    assert not bool(loss.scale_initialized)
    _forward(loss, counts, mask, metadata)
    assert bool(loss.scale_initialized)
    G = float(loss.get_G())
    # crude proxy, but must land within a small factor of the true scale
    # (the collapse trap is ~300x off; this must be far tighter)
    assert G_TRUE / 3.0 < G < G_TRUE * 3.0


def test_init_B_is_physical_in_both_parameterizations():
    """One knob: init_B is the B factor itself, floor included, either way."""
    for stabilize in (False, True):
        for b_min in (0.0, 1.0):
            loss = MonochromaticWilsonLoss(
                stabilize_scale=stabilize, init_B=30.0, b_min=b_min,
                bg_rate=1.0, bg_concentration=1.0,
            )
            assert abs(float(loss.get_B()) - 30.0) < 1e-3


def test_init_G_is_physical_and_pins_the_scale():
    """init_G sets G exactly and disables the count-based fit."""
    for stabilize in (False, True):
        loss = MonochromaticWilsonLoss(
            stabilize_scale=stabilize, init_G=250.0,
            bg_rate=1.0, bg_concentration=1.0,
        )
        assert abs(float(loss.get_G()) - 250.0) < 1e-2
        assert not loss.init_scale_from_counts


def test_init_G_survives_the_first_forward():
    """With init_G set, the Wilson fit must not overwrite the pinned scale."""
    loss = MonochromaticWilsonLoss(
        stabilize_scale=True, init_G=250.0, b_min=1.0,
        bg_rate=1.0, bg_concentration=1.0,
    )
    counts, mask, metadata, _ = _synthetic_counts()
    _forward(loss, counts, mask, metadata)
    assert abs(float(loss.get_G()) - 250.0) < 1e-2  # not refit to G_TRUE=150
    assert not bool(loss.scale_initialized)


def test_default_G_without_init_G_is_one():
    for stabilize in (False, True):
        loss = MonochromaticWilsonLoss(
            stabilize_scale=stabilize, bg_rate=1.0, bg_concentration=1.0,
        )
        assert abs(float(loss.get_G()) - 1.0) < 1e-5
        assert loss.init_scale_from_counts  # fit still enabled


def test_init_runs_once():
    loss = MonochromaticWilsonLoss(
        stabilize_scale=True, bg_rate=1.0, bg_concentration=1.0,
    )
    counts, mask, metadata, _ = _synthetic_counts()
    _forward(loss, counts, mask, metadata)
    G_after_first = float(loss.get_G())
    with torch.no_grad():
        loss.raw_G.copy_(torch.tensor(math.log(999.0)))  # pretend training moved it
    _forward(loss, counts, mask, metadata)  # must NOT re-init
    assert abs(float(loss.get_G()) - 999.0) < 1e-2
    assert G_after_first != 999.0


def test_stabilize_forward_backward():
    loss = MonochromaticWilsonLoss(
        stabilize_scale=True, bg_rate=1.0, bg_concentration=1.0,
    )
    counts, mask, metadata, _ = _synthetic_counts()
    out = _forward(loss, counts, mask, metadata)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert loss.raw_G.grad is not None and torch.isfinite(loss.raw_G.grad).all()
    assert loss.raw_B.grad is not None and torch.isfinite(loss.raw_B.grad).all()


def test_stabilize_state_dict_roundtrip():
    a = MonochromaticWilsonLoss(stabilize_scale=True, bg_rate=1.0, bg_concentration=1.0)
    counts, mask, metadata, _ = _synthetic_counts()
    _forward(a, counts, mask, metadata)  # marks scale_initialized True
    b = MonochromaticWilsonLoss(stabilize_scale=True, bg_rate=1.0, bg_concentration=1.0)
    b.load_state_dict(a.state_dict())  # resume must carry the init flag
    assert bool(b.scale_initialized)
