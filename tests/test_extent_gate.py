"""Tests for the information-gated profile extent.

The geometry-predicted extent rescues weak reflections but fits bright spots worse than a
freely inferred profile does, so the gate lets a shoebox bend the extent toward its own
data in proportion to how many signal photons it has. The properties worth pinning are
that it is off by default and byte-identical when off, that it saturates correctly, and
that a bright shoebox actually moves while a weak one does not.
"""

import pytest
import torch

from integrator.model.distributions.profile_surrogates import GeometricProfileSurrogate

BEAM = [1984.99, 1967.21]
SHAPE = [1, 21, 21]
IN = 64


def _surrogate(gate: bool, half_signal: float = 100.0, seed: int = 0):
    torch.manual_seed(seed)
    return GeometricProfileSurrogate(
        input_dim=IN, latent_dim=16, output_dim=441, profile_shape=SHAPE,
        beam_center=BEAM, lambda_min=0.95, lambda_max=1.2,
        local_shift=True, steer=True, gate_extent=gate,
        extent_half_signal=half_signal,
    )


def _meta(b=8):
    torch.manual_seed(1)
    return {
        "xyzcal.px.0": torch.rand(b) * 4000.0,
        "xyzcal.px.1": torch.rand(b) * 4000.0,
        "wavelength": 0.95 + torch.rand(b) * 0.25,
        "d": 1.5 + torch.rand(b) * 3.0,
    }


def test_gate_is_off_by_default_and_requires_steer():
    assert _surrogate(gate=False).gate_extent is False
    assert _surrogate(gate=False).extent_residual_head is None
    torch.manual_seed(0)
    nosteer = GeometricProfileSurrogate(
        input_dim=IN, latent_dim=16, output_dim=441, profile_shape=SHAPE,
        beam_center=BEAM, local_shift=True, steer=False, gate_extent=True,
    )
    # the gate acts on the warp, so without steering there is nothing to gate
    assert nosteer.gate_extent is False


def test_gated_off_is_identical_to_ungated():
    """A config without the gate must behave exactly as before it existed."""
    x = torch.randn(8, IN)
    meta = _meta()
    a = _surrogate(gate=False, seed=3)(x, mc_samples=2, metadata=meta)
    b = _surrogate(gate=True, seed=3)(x, mc_samples=2, metadata=meta,
                                      shape_signal=torch.zeros(8))
    # zero signal closes the gate; the residual head is zero-initialised anyway
    assert torch.allclose(a.mean_profile, b.mean_profile, atol=1e-6)


def test_zero_init_means_training_starts_from_pure_geometry():
    s = _surrogate(gate=True)
    x = torch.randn(8, IN)
    out_open = s(x, mc_samples=1, metadata=_meta(), shape_signal=torch.full((8,), 1e6))
    out_shut = s(x, mc_samples=1, metadata=_meta(), shape_signal=torch.zeros(8))
    assert torch.allclose(out_open.mean_profile, out_shut.mean_profile, atol=1e-6)


def test_gate_saturates_between_zero_and_one():
    s = _surrogate(gate=True, half_signal=100.0)
    like = torch.zeros(5, 16)
    g = s._extent_gate(torch.tensor([0.0, 50.0, 100.0, 1e3, 1e9]), like)
    assert g.shape == (5, 1)
    assert float(g[0]) == pytest.approx(0.0)
    assert float(g[2]) == pytest.approx(0.5, abs=1e-3)
    assert 0.0 <= float(g.min()) and float(g.max()) < 1.0
    assert torch.all(g[1:] > g[:-1])          # monotone in signal
    # negative signal (background over-subtracted) must not open the gate
    assert float(s._extent_gate(torch.tensor([-50.0]), torch.zeros(1, 16))[0]) == 0.0


def test_no_signal_reproduces_the_closed_gate():
    s = _surrogate(gate=True)
    x = torch.randn(4, IN)
    meta = _meta(4)
    a = s(x, mc_samples=1, metadata=meta)                       # no shape_signal at all
    b = s(x, mc_samples=1, metadata=meta, shape_signal=torch.zeros(4))
    assert torch.allclose(a.mean_profile, b.mean_profile, atol=1e-7)


def test_a_trained_residual_moves_bright_spots_and_not_weak_ones():
    """Once the residual head is non-zero the gate must decide who it reaches."""
    s = _surrogate(gate=True, half_signal=100.0)
    with torch.no_grad():
        s.extent_residual_head.weight.normal_(0.0, 0.05)
        s.extent_residual_head.bias.fill_(0.3)
    x = torch.randn(6, IN)
    meta = _meta(6)
    base = s(x, mc_samples=1, metadata=meta, shape_signal=torch.zeros(6)).mean_profile
    weak = s(x, mc_samples=1, metadata=meta, shape_signal=torch.full((6,), 1.0)).mean_profile
    bright = s(x, mc_samples=1, metadata=meta, shape_signal=torch.full((6,), 1e4)).mean_profile
    d_weak = float((weak - base).abs().max())
    d_bright = float((bright - base).abs().max())
    # the gate itself is the crisp property: 1/(1+100) against 1e4/(1e4+100)
    g = s._extent_gate(torch.tensor([1.0, 1e4]), torch.zeros(2, 16)).reshape(-1)
    assert float(g[0]) == pytest.approx(1.0 / 101.0, rel=1e-3)
    assert float(g[1]) == pytest.approx(1e4 / 10100.0, rel=1e-3)
    assert d_weak < 1e-2, "a 1-photon shoebox should barely move"
    assert d_bright > 10 * d_weak, "a bright shoebox should set its own extent"


def test_profiles_stay_on_the_simplex_under_the_gate():
    s = _surrogate(gate=True)
    with torch.no_grad():
        s.extent_residual_head.weight.normal_(0.0, 0.5)
        s.extent_residual_head.bias.normal_(0.0, 0.5)
    out = s(torch.randn(8, IN), mc_samples=3, metadata=_meta(),
            shape_signal=torch.rand(8) * 1e4)
    assert out.zp.shape == (3, 8, 441)
    assert torch.allclose(out.zp.sum(-1), torch.ones_like(out.zp.sum(-1)), atol=1e-5)
    assert torch.allclose(out.mean_profile.sum(-1), torch.ones(8), atol=1e-5)
    assert float(out.zp.min()) >= 0.0
