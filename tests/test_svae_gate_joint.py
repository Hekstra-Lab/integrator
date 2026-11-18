"""`gate: joint` -- the responsibility built on a joint (I, B) fit.

The gate's job is the Poisson split. For `c_i = s_i + b_i` with
`s_i ~ Poisson(I p_i)` and `b_i ~ Poisson(B)`, the exact posterior share of
a count belonging to signal is `r_i = I p_i / (I p_i + B)`, and with that
share `E[sum_i r_i c_i] = I` identically. Every gate here is scored on that
sum against the intensity the counts were drawn with.

What these pin is the failure `scripts/gate_study.py` found: the form is
right and the ESTIMATOR is what loses it. `responsibility` takes the
background from the box median, and a Poisson median is exactly 0 whenever
the rate is below about 0.7, which saturates the gate to 1 wherever the
profile is non-zero and sends the intensity to the box total. `joint` fits
`I` and `B` together against `[p, 1]`, which neither quantises nor is
robust by accident.
"""

from __future__ import annotations

import math

import pytest
import torch

from integrator.model.integrators.svae_integrator import SVAEIntegrator


class _Gate:
    """The gate computation alone, with a config and nothing else.

    `_get_pixel_probs` needs only `self.cfg`, so the integrator is not
    constructed: building one drags in encoders, surrogates and a loss that
    have no bearing on the arithmetic under test, and the point here is the
    arithmetic.
    """

    def __init__(self, gate: str):
        self.cfg = type("cfg", (), {"gate": gate, "gate_detach": False})()

    _get_pixel_probs = SVAEIntegrator._get_pixel_probs


class _QP:
    def __init__(self, profile):
        self.mean_profile = profile


def _shoeboxes(n=512, side=21, intensity=50.0, background=0.3, seed=0):
    """Streaky Gaussians, and counts drawn from `Poisson(I p + B)`.

    The defaults are the regime this matters in: `background=0.3` is below
    the rate at which a Poisson median is 0, and `intensity=50` against a
    background total of 0.3 * 441 = 132 is background-dominated, as a Laue
    shoebox is -- 96.6% background, measured. A bright reflection hides the
    defect: at I=1500 the median gate is only 1.5% high, and at I=20 it is
    372% high.
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.arange(side) - (side - 1) / 2.0,
        torch.arange(side) - (side - 1) / 2.0,
        indexing="ij",
    )
    profiles = []
    for _ in range(n):
        theta = torch.rand(1, generator=g).item() * math.pi
        a = 1.6 + 2.9 * torch.rand(1, generator=g).item()
        b = a / (1.5 + 2.5 * torch.rand(1, generator=g).item())
        ct, st = math.cos(theta), math.sin(theta)
        u = xx * ct + yy * st
        v = -xx * st + yy * ct
        p = torch.exp(-0.5 * ((u / a) ** 2 + (v / b) ** 2))
        profiles.append((p / p.sum()).reshape(-1))
    prf = torch.stack(profiles)
    rate = intensity * prf + background
    counts = torch.poisson(rate, generator=g)
    return prf, counts, intensity


def _recovered(gate: str, prf, counts):
    """`sum_i r_i c_i` for one gate, averaged over the batch."""
    mask = torch.ones_like(prf).unsqueeze(-1)
    r, _bg, _p = _Gate(gate)._get_pixel_probs(
        _QP(prf), None, raw_counts=counts, mask=mask
    )
    return (r * counts).sum(-1)


@pytest.fixture(scope="module")
def faint():
    """A background of 0.3 per pixel: the median of these counts is 0."""
    return _shoeboxes(background=0.3)


def test_the_median_background_really_is_zero_here(faint):
    """The premise. Without it the rest of the file tests nothing."""
    _prf, counts, _i = faint
    b0 = counts.median(dim=-1).values
    assert (b0 == 0).float().mean() > 0.9


def test_joint_beats_the_median_background_on_a_faint_shoebox(faint):
    prf, counts, intensity = faint
    joint = _recovered("joint", prf, counts).mean().item() / intensity
    plugin = _recovered("responsibility", prf, counts).mean().item() / intensity
    # the median collapses to zero, the gate saturates, and the intensity
    # becomes the box total -- several times the truth
    # measured at these settings: joint 0.989, responsibility 2.507
    assert plugin > 1.8, plugin
    assert abs(joint - 1.0) < 0.10, joint
    assert abs(joint - 1.0) < abs(plugin - 1.0)


def test_both_agree_once_the_reflection_is_bright(faint):
    """The defect closes at the bright end, which is why it went unseen."""
    prf, counts, intensity = _shoeboxes(intensity=1500.0, background=0.3,
                                        seed=3)
    joint = _recovered("joint", prf, counts).mean().item() / intensity
    plugin = _recovered("responsibility", prf, counts).mean().item() / intensity
    assert abs(joint - 1.0) < 0.1, joint
    assert abs(plugin - 1.0) < 0.1, plugin


def test_the_gate_is_a_probability(faint):
    prf, counts, _i = faint
    for gate in ("responsibility", "joint"):
        mask = torch.ones_like(prf).unsqueeze(-1)
        probs, bg, _p = _Gate(gate)._get_pixel_probs(
            _QP(prf), None, raw_counts=counts, mask=mask
        )
        assert torch.isfinite(probs).all()
        assert (probs >= 0).all() and (probs <= 1).all()
        # signal and background shares partition the pixel
        assert torch.allclose(probs + bg, torch.ones_like(probs), atol=1e-5)


def test_a_computed_gate_needs_the_counts():
    """`joint` derives the gate from data, so it must say when it has none."""
    prf = torch.full((2, 9), 1 / 9)
    with pytest.raises(ValueError, match="needs raw_counts"):
        _Gate("joint")._get_pixel_probs(_QP(prf), None, raw_counts=None,
                                        mask=None)
