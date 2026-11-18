"""Tests for `HellingerBasisProfileSurrogate`.

The claims worth testing are the geometric ones, because they are what the
family is for: the profile stays on the simplex without a softmax, the basis is
orthonormal in the Fisher-Rao metric so the coefficients carry an angle, and the
directions that the counts cannot identify are absent from the span.
"""

import math

import pytest
import torch

from integrator.fit.direct import solve_profile_fit
from integrator.model.distributions import HellingerBasisProfileSurrogate
from integrator.registry import REGISTRY

SHAPE = [3, 21, 21]
P = SHAPE[0] * SHAPE[1] * SHAPE[2]
BEAM = [2039.525348886683, 2194.7467011214735]


def make(**kw):
    kw.setdefault("input_dim", 32)
    kw.setdefault("latent_dim", 8)
    kw.setdefault("output_dim", P)
    kw.setdefault("profile_shape", SHAPE)
    return HellingerBasisProfileSurrogate(**kw)


def physical(**kw):
    return make(
        base_mode="physical",
        base_args=dict(
            beam_center=BEAM,
            wavelength=1.03752,
            init_a=8.90,
            init_b=0.176,
            init_z=0.158,
            quad=2,
        ),
        **kw,
    )


def metadata(d_values):
    n = len(d_values)
    return {
        "d": torch.tensor(d_values),
        "xyzcal.px.0": torch.linspace(1500.0, 2500.0, n),
        "xyzcal.px.1": torch.linspace(1900.0, 2600.0, n),
    }


def n_eff(p):
    return 1.0 / p.pow(2).sum(-1)


def shannon(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def test_the_profile_is_a_density_without_a_softmax():
    torch.manual_seed(0)
    m = make()
    out = m(torch.randn(6, 32), mc_samples=4)
    p = out.zp
    assert p.shape == (4, 6, P)
    assert (p >= 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(4, 6), atol=1e-5)
    assert out.mean_profile.shape == (6, P)


def test_the_basis_is_orthonormal_and_tangent():
    torch.manual_seed(0)
    m = make()
    psi0 = m._base_point(None, None)
    e = m._frame(psi0)
    assert e.shape == (P, m.d)
    gram = e.T @ e - torch.eye(m.d)
    assert gram.abs().max() < 1e-5
    # tangency: a component along psi0 would change the norm, not the shape
    assert (e.T @ psi0).abs().max() < 1e-5


@pytest.mark.parametrize("fn", [n_eff, shannon])
def test_projecting_makes_the_width_stationary_in_the_coefficients(fn):
    """The whole point: `h` cannot move the width to first order.

    Both width functionals are tested because their gradients in `psi` are
    different directions -- `4 psi^3` against `-2 psi (log p + 1)` -- and
    removing only one leaves the other moving.
    """
    grads = {}
    for flag in (True, False):
        torch.manual_seed(0)
        m = make(project_breadth=flag)
        psi0 = m._base_point(None, None)
        e = m._frame(psi0)
        h = torch.zeros(1, 1, m.d, requires_grad=True)
        val = fn(m._decode(h, psi0, e, None)).sum()
        grads[flag] = torch.autograd.grad(val, h)[0].abs().max().item()
    assert grads[True] < 1e-4
    # and the unprojected family really does move, so the test has teeth
    assert grads[False] > 1e-2


def test_the_prior_radius_is_an_angle_independent_of_the_latent_width():
    """`|h|` over `d` coordinates grows as `sqrt(d)`, so the per-coordinate
    scale has to shrink for the geodesic radius to mean the same thing."""
    for d in (2, 8, 32):
        m = make(latent_dim=d, prior_radius=0.4)
        assert m.prior_scale == pytest.approx(0.4 / math.sqrt(d))
        torch.manual_seed(0)
        h = m.prior_scale * torch.randn(20000, 1, d)
        # E|h|^2 = d * prior_scale^2 = prior_radius^2
        assert h.norm(dim=-1).pow(2).mean().sqrt().item() == pytest.approx(
            0.4, rel=0.05
        )


def test_the_coefficient_norm_is_the_bhattacharyya_angle():
    """`cos |h| = sum_k sqrt(p_k p0_k)`, which is what gives the chart's axes a
    unit.

    Exact only while `psi` stays non-negative. Past that, `sqrt(p) = |psi|` and
    the folded pixels contribute `+2 sum |psi| psi0` instead of their signed
    value. The error is therefore bounded by the base profile's own mass in the
    folded region, which measured 8e-5 / 9e-4 / 2e-2 of `p0` at these three
    radii, matching the tolerances below. `p = psi^2` stays a valid density
    throughout -- a sign change is a node, not a failure.
    """
    torch.manual_seed(0)
    m = make()
    psi0 = m._base_point(None, None)
    e = m._frame(psi0)
    for radius, tol in ((0.1, 2e-4), (0.4, 2e-3), (1.0, 2e-2)):
        h = torch.zeros(1, 1, m.d)
        h[..., 0] = radius
        p = m._decode(h, psi0, e, None)
        bc = (p.clamp_min(0).sqrt() * psi0).sum(-1)
        assert bc.item() == pytest.approx(math.cos(radius), abs=tol)


def test_the_physical_base_point_follows_resolution_and_position():
    m = physical()
    md = metadata([12.9, 5.0, 3.0, 2.0, 1.2])
    mask = torch.ones(5, P, dtype=torch.bool)
    psi0 = m._base_point(md, mask)
    assert psi0.shape == (5, P)
    assert torch.allclose(psi0.pow(2).sum(-1), torch.ones(5), atol=1e-5)
    # a sin^2(theta) + b grows as d falls, so the profile covers more pixels
    widths = n_eff(psi0.pow(2))
    assert (widths.diff() > 0).all()


def test_the_physical_base_point_needs_metadata():
    m = physical()
    with pytest.raises(ValueError, match="metadata"):
        m._base_point(None, torch.ones(2, P, dtype=torch.bool))


def test_the_width_law_trains_while_the_coefficients_cannot_move_it():
    """Width is a pooled global quantity, shape is per-reflection."""
    m = physical()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert {"base.log_a", "base.log_b", "base.log_z"} <= trainable


def test_a_per_reflection_base_point_gets_its_own_orthonormal_span():
    m = physical()
    md = metadata([8.0, 3.0, 1.5])
    mask = torch.ones(3, P, dtype=torch.bool)
    psi0 = m._base_point(md, mask)
    e = m._frame(psi0)
    assert e.shape == (3, P, m.d)
    eye = torch.eye(m.d).expand(3, m.d, m.d)
    assert (e.transpose(-1, -2) @ e - eye).abs().max() < 1e-4
    assert torch.einsum("bpd,bp->bd", e, psi0).abs().max() < 1e-4
    # the spans differ, which is the point of conditioning on geometry
    assert (e[0] - e[2]).abs().max() > 1e-3


def test_freeze_base_holds_the_learned_base_point():
    assert not make(freeze_base=True).base_raw.requires_grad
    assert make(freeze_base=False).base_raw.requires_grad


def test_project_shift_removes_the_translation_directions():
    torch.manual_seed(0)
    m = make(project_shift=True)
    psi0 = m._base_point(None, None)
    e = m._frame(psi0)
    for col in m._shift_columns(psi0):
        assert (e.T @ col).abs().max() < 1e-4


def test_masked_pixels_carry_no_profile_mass():
    torch.manual_seed(0)
    m = make()
    mask = torch.ones(4, P, dtype=torch.bool)
    mask[:, : P // 2] = False
    p = m(torch.randn(4, 32), mc_samples=2, mask=mask).zp
    assert p[..., : P // 2].abs().max() == 0.0
    assert torch.allclose(p.sum(-1), torch.ones(2, 4), atol=1e-5)


def test_the_name_resolves_through_the_registry():
    assert REGISTRY["surrogates"]["hellinger_basis_profile"] is (
        HellingerBasisProfileSurrogate
    )


def test_the_intensity_variance_is_one_over_the_weighted_profile_variance():
    """`Var(I) = 1 / (sum_j w_j Var_w(p))`.

    This is why the profile is what constrains the intensity and the
    background: the determinant of the 2x2 information is `(sum w)^2 Var_w(p)`,
    which vanishes exactly when the profile is flat.
    """
    h = w = 21
    n = h * w
    yy, xx = torch.meshgrid(
        torch.arange(h) - 10.0, torch.arange(w) - 10.0, indexing="ij"
    )
    p = torch.exp(-0.5 * (yy**2 + xx**2) / 4.0).reshape(-1)
    p = (p / p.sum()).double()

    b_shape = 256
    torch.manual_seed(0)
    i_true = (400.0 + 200.0 * torch.rand(b_shape)).double()
    counts = torch.poisson(i_true[:, None] * p + 2.0).double()
    mask = torch.ones(b_shape, n, dtype=torch.bool)
    bg_cols = torch.ones(1, n, dtype=torch.float64)

    res = solve_profile_fit(
        counts, mask, p.expand(b_shape, n).contiguous(), bg_cols
    )
    weight = 1.0 / (
        res.intensity.clamp_min(0)[:, None] * p + res.background[:, None]
    ).clamp_min(1e-12)
    total = weight.sum(-1)
    mean_p = (weight * p).sum(-1) / total
    var_p = (weight * (p - mean_p[:, None]) ** 2).sum(-1) / total

    predicted = 1.0 / (total * var_p)
    assert torch.allclose(predicted, res.variance, rtol=1e-4)


def test_a_flat_profile_loses_the_intensity_information_entirely():
    """The gauge, as a limit: mixing toward uniform drives `Var_w(p)` to zero."""
    n = 441
    torch.manual_seed(0)
    p = torch.rand(n).double()
    p = p / p.sum()
    uniform = torch.full_like(p, 1.0 / n)
    weight = torch.ones(n, dtype=torch.float64)

    def info(prof):
        total = weight.sum()
        mean_p = (weight * prof).sum() / total
        return total * (weight * (prof - mean_p) ** 2).sum() / total

    previous = info(p)
    for alpha in (0.5, 0.2, 0.05, 0.0):
        current = info(alpha * p + (1 - alpha) * uniform)
        assert current < previous
        previous = current
    assert previous == pytest.approx(0.0, abs=1e-20)


def geometric(**kw):
    kw.setdefault("geom_dim", 6)
    kw.setdefault("geom_radius", 0.8)
    return make(base_mode="geometric", beam_center=BEAM, **kw)


def test_the_geometry_map_starts_at_the_global_base_point():
    """The head's last layer is zeroed, so before training every reflection
    gets `psi_bar`. A map that began somewhere arbitrary would move every
    profile before the shape coefficients had learned to correct it."""
    m = geometric()
    psi0 = m._base_point(metadata([12.0, 5.0, 2.0]), None)
    assert psi0.shape == (3, P)
    assert (psi0 - psi0[0]).abs().max() == 0.0


def test_the_geometry_map_stays_on_the_sphere_and_inside_its_radius():
    torch.manual_seed(0)
    m = geometric(geom_radius=0.5)
    with torch.no_grad():
        for prm in m.geom_head[-1].parameters():
            prm.normal_(0.0, 2.0)
    md = metadata([12.0, 8.0, 5.0, 3.0, 1.5])
    psi0 = m._base_point(md, None)
    assert torch.allclose(psi0.pow(2).sum(-1), torch.ones(5), atol=1e-5)
    psi_bar = m._base_point(md, None)  # same call, for the reference point
    bar = torch.nn.functional.softplus(m.base_raw)
    bar = bar / bar.norm()
    angle = torch.arccos((psi0 * bar).sum(-1).clamp(-1.0, 1.0))
    assert angle.max() <= 0.5 + 1e-5
    assert psi_bar.shape == psi0.shape


def test_geometry_moves_the_base_point_but_coefficients_cannot_move_the_width():
    """The division that makes the width identifiable without a formula: it is
    pooled through a map shared by every reflection at the same position, and
    absent from the per-reflection span."""
    torch.manual_seed(0)
    m = geometric()
    with torch.no_grad():
        for prm in m.geom_head[-1].parameters():
            prm.normal_(0.0, 0.6)
    md = metadata([12.0, 5.0, 1.5])
    psi0 = m._base_point(md, None)
    # geometry does change the profile across reflections
    assert (psi0[0] - psi0[2]).abs().max() > 1e-4
    # but h cannot change the width at any of those base points
    e = m._frame(psi0)
    h = torch.zeros(1, 3, m.d, requires_grad=True)
    val = n_eff(m._decode(h, psi0, e, None)).sum()
    assert torch.autograd.grad(val, h)[0].abs().max().item() < 1e-4


def test_geometric_needs_a_beam_centre_and_metadata():
    with pytest.raises(ValueError, match="beam_center"):
        make(base_mode="geometric")
    with pytest.raises(ValueError, match="metadata"):
        geometric()._base_point(None, None)


def test_an_unknown_base_mode_is_rejected():
    with pytest.raises(ValueError, match="base_mode"):
        make(base_mode="quadratic")
