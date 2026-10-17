"""The Laue scale as physics, and the properties that make it worth having.

Every earlier Laue arm scaled with a generic MLP on lab-frame coordinates, or
with `LaueSpectralScale`, whose residual MLP on `[x, y, d]` destroys the
spectrum's identifiability: on a still `lambda = 2 d sin(theta(r))`, so a free
function of position and resolution absorbs 73% of a realistic `G(lambda)`.

These tests assert what the class is for -- that nothing is free in position or
resolution, that the polarisation is applied as known physics rather than
learned, that absorption is evaluated in the crystal frame, and that the basis
is the same one `scripts/fit_physical_scale.py` fits offline so the two are
comparable.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from integrator.model.scaling.chebyshev_scale import (
    LauePhysicalScale,
    _real_sph_harm,
    _rotate_about_axis,
)

ROOT = Path(__file__).resolve().parents[1]


def _offline():
    spec = importlib.util.spec_from_file_location(
        "fps", ROOT / "scripts" / "fit_physical_scale.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FPS = _offline()


def _inputs(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    s1 = torch.randn(n, 3, generator=g)
    return dict(
        wavelength=0.97 + 0.28 * torch.rand(n, generator=g),
        d=2.0 + 8.0 * torch.rand(n, generator=g),
        x=4000.0 * torch.rand(n, generator=g),
        y=4000.0 * torch.rand(n, generator=g),
        s1=s1 / s1.norm(dim=-1, keepdim=True),
        image=torch.randint(0, 250, (n,), generator=g).float(),
    )


def _scale(**over):
    kw = dict(lmax=4, n_images=250, beam_center=(2000.0, 2000.0))
    kw.update(over)
    return LauePhysicalScale(**kw)


@pytest.mark.parametrize("lmax", [1, 2, 3, 4])
def test_the_basis_matches_the_offline_fit_column_for_column(lmax):
    """Or an offline fit and this module are not measuring the same thing."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(300, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    want, want_l = FPS.real_sph_harm(v, lmax, True)
    got, got_l = _real_sph_harm(torch.from_numpy(v).double(), lmax)
    assert got.shape == want.shape
    assert list(want_l) == got_l
    assert np.allclose(got.numpy(), want, atol=1e-9)


def test_the_rotation_matches_the_offline_fit():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 3))
    ang = rng.uniform(-3.0, 3.0, size=200)
    want = FPS.rotate_about_axis(v, (0.0, 1.0, 0.0), ang)
    got = _rotate_about_axis(
        torch.from_numpy(v).double(),
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        torch.from_numpy(ang).double(),
    )
    assert np.allclose(got.numpy(), want, atol=1e-9)


def test_the_polarisation_is_physics_and_not_a_parameter():
    """Fitting it lands the plane 22.5 deg off a known truth and keeps it there.

    Its azimuthal signature `cos(2(phi - phi0))` lies inside the real harmonics
    at `l = 2, m = +-2`, which ARE `cos 2phi` and `sin 2phi`, so an absorption
    surface spans it and only the `sin^2(2theta)` modulation separates them.
    """
    m = _scale()
    names = {n for n, _ in m.named_parameters()}
    assert not any("tau" in n or "phi0" in n for n in names), names
    assert "tau" in dict(m.named_buffers())
    assert "phi0" in dict(m.named_buffers())

    # and it is the rate's multiplier P, not the tabulated correction 1/P: at
    # the azimuth where P is smallest the term must be NEGATIVE in log
    inp = _inputs()
    terms = m.log_terms(**inp)
    two_t = 2.0 * torch.asin(
        (inp["wavelength"] / (2.0 * inp["d"])).clamp(-1.0, 1.0)
    )
    phi = torch.atan2(inp["y"] - 2000.0, inp["x"] - 2000.0)
    want = FPS.polarisation_offset(
        two_t.numpy().astype(np.float64), phi.numpy().astype(np.float64), 0.95, 0.0
    )
    assert np.allclose(terms["polarisation"].numpy(), want, atol=1e-5)
    assert float(terms["polarisation"].max()) <= 1e-6


def test_nothing_is_free_in_detector_position_or_resolution():
    """The identifiability rule, as a property of the output.

    Moving an observation across the detector at fixed wavelength, resolution
    and image may only change the scale through the polarisation -- there is no
    free function of position. The same holds for resolution through decay.
    """
    m = _scale()
    with torch.no_grad():
        m.spectrum_c.normal_(0.0, 0.3)
        m.image_c.normal_(0.0, 0.1)
        m.decay_c.normal_(0.0, 0.1)
        m.absorption_c.normal_(0.0, 0.05)
    inp = _inputs(n=200, seed=4)
    base = m.log_terms(**inp)
    moved = dict(inp)
    moved["x"] = inp["x"] * 0.5 + 1000.0
    moved["y"] = inp["y"] * 0.5 + 1000.0
    after = m.log_terms(**moved)
    for term in ("spectrum", "image_gray", "decay", "absorption"):
        assert torch.allclose(base[term], after[term], atol=1e-6), term
    assert not torch.allclose(base["polarisation"], after["polarisation"])


def test_absorption_is_a_crystal_frame_term():
    """A lab-frame scale cannot represent it, which is the point of having it.

    The same diffracted direction on two different images is a DIFFERENT
    crystal-frame direction, so the term must change -- and that is the
    coordinate DIALS uses and no arm of ours has had.
    """
    m = _scale()
    with torch.no_grad():
        m.absorption_c.normal_(0.0, 0.1)
    inp = _inputs(n=128, seed=7)
    same_lab = dict(inp)
    same_lab["image"] = inp["image"] + 90.0 / m.deg_per_image
    a = m.log_terms(**inp)["absorption"]
    b = m.log_terms(**same_lab)["absorption"]
    assert not torch.allclose(a, b, atol=1e-4)


def test_it_starts_at_the_polarisation_and_nothing_else():
    """Zero coefficients, so the merge begins as an inverse-variance mean."""
    m = _scale()
    inp = _inputs()
    terms = m.log_terms(**inp)
    for name in ("spectrum", "image_gray", "decay", "absorption"):
        assert torch.allclose(terms[name], torch.zeros_like(terms[name])), name
    assert float(m.restraint_penalty()) == 0.0


def test_the_restraint_weights_are_the_ones_the_offline_fit_uses():
    """A restraint weight tuned offline has to mean the same thing here."""
    m = _scale()
    with torch.no_grad():
        m.absorption_c.fill_(0.0)
        m.absorption_c[0] = 1.0          # l = 1
        m.spectrum_c.fill_(0.0)
    assert float(m.restraint_penalty()) == pytest.approx(1.0 * 2.0)
    with torch.no_grad():
        m.absorption_c.fill_(0.0)
        m.absorption_c[-1] = 1.0         # l = 4
    assert float(m.restraint_penalty()) == pytest.approx(4.0 * 5.0)
    with torch.no_grad():
        m.absorption_c.fill_(0.0)
        m.spectrum_c.fill_(0.0)
        m.spectrum_c[2] = 1.0            # third order, roughness k^4
    assert float(m.restraint_penalty()) == pytest.approx(3.0 ** 4)


def test_the_keys_default_off():
    from integrator.configs.integrator import IntegratorCfg

    f = IntegratorCfg.__dataclass_fields__
    assert f["scale_laue_physical"].default is False
    # and the pre-existing key keeps its own default, or LaueSpectralScale
    # silently changes for every config that does not name it
    assert f["scale_spectrum_degree"].default == 40
    assert f["scale_laue_spectrum_degree"].default == 6


def test_it_reaches_the_merge_through_the_factory():
    """Per the repo rule: a new config is tested through `construct_integrator`."""
    import yaml

    from integrator.model.scaling.chebyshev_scale import LauePhysicalScale as L
    from integrator.utils.factory_utils import construct_integrator

    cfg = yaml.safe_load(
        (ROOT / "tests" / "configs" / "sim_laue_physical.yaml").read_text()
    )
    model = construct_integrator(cfg)
    assert isinstance(model.scale_fn, L)
    assert model.scale_restraint_weight > 0.0
