"""Fitting the reflection's centre as well as its intensity.

`_profile` pins the Gaussian to the middle of the shoebox, so a mis-predicted
centroid puts the profile in the wrong place and biases the intensity low. These
tests pin the three properties that make the fix legitimate rather than merely
more flexible:

  - the two derivative columns sum to zero over the valid pixels, so they span
    the translation directions and cannot move total mass between the profile
    and the uniform direction. That is what keeps `I` identified, and it is the
    whole difference between this and letting an encoder reshape the profile per
    reflection;
  - a planted shift is recovered, and the intensity with it, where the
    centre-pinned solve is biased low on the same data;
  - the shrinkage holds weak reflections at their predicted centre instead of
    letting them chase a background fluctuation.

The last one is the failure mode to guard: an unshrunk per-reflection centre is
a two-parameter estimate from a handful of photons, and turning it loose would
scatter weak spots and bias exactly the reflections the recentring is meant to
help.
"""

import pytest
import torch

from integrator.fit import (
    shrink_shift,
    solve_profile_fit,
    solve_profile_fit_recentered,
)
from integrator.model.distributions.profile_surrogates import (
    PhysicalProfileSurrogate,
)

HW = 25
P = HW * HW


def surrogate(a=1e-6, b=4.0, c=5.0, hw=HW):
    """Widths pinned by construction: `a = 0` gives `v_tan = b`, `v_rad = b + c`.

    `beam_center=None` sets `phi = 0`, so radial is the fast (x) axis and
    tangential the slow (y) one, and the geometry needs no detector metadata.
    """
    return PhysicalProfileSurrogate(
        output_dim=hw * hw,
        profile_shape=[1, hw, hw],
        beam_center=None,
        init_a=a,
        init_b=b,
        init_c=c,
        init_z=1.0,
        quad=2,
        learn_globals=False,
    )


def meta(n):
    return {
        "d": torch.full((n,), 2.0),
        "wavelength": torch.full((n,), 1.1),
    }


def plant(n=256, mu=(1.5, -1.0), intensity=2000.0, background=20.0, seed=0):
    """Counts from `Poisson(I p(mu) + B)` with the surrogate's own shifted profile."""
    s = surrogate()
    md = meta(n)
    mask = torch.ones(n, P)
    true_mu = torch.zeros(n, 2)
    true_mu[:, 0], true_mu[:, 1] = mu[0], mu[1]
    with torch.no_grad():
        p_true, _ = s.profile_and_grad(md, mask, true_mu)
    g = torch.Generator().manual_seed(seed)
    counts = torch.poisson(intensity * p_true + background, generator=g)
    return s, md, mask, counts, true_mu, float(intensity), float(background)


def bound(s, md, mask):
    def fn(offset):
        with torch.no_grad():
            return s.profile_and_grad(md, mask, offset)

    return fn


# ------------------------------------------------------------- identification


def test_derivative_columns_sum_to_zero():
    """The property that keeps `I` identified; everything else rests on it."""
    s, md, mask = surrogate(), meta(8), torch.ones(8, P)
    for off in (None, torch.full((8, 2), 0.7), torch.full((8, 2), -2.0)):
        with torch.no_grad():
            p, g = s.profile_and_grad(md, mask, off)
        assert torch.allclose(p.sum(-1), torch.ones(8), atol=1e-5)
        assert g.sum(-1).abs().max() < 1e-5, "a column with mass trades against I"


def test_zero_sum_survives_a_partial_mask():
    """Normalization is over the VALID pixels, so the sum must be too."""
    s, md = surrogate(), meta(4)
    mask = torch.ones(4, P)
    mask[:, : 3 * HW] = 0.0
    mask[:, -2 * HW :] = 0.0
    with torch.no_grad():
        p, g = s.profile_and_grad(md, mask, torch.full((4, 2), 0.5))
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    assert g.sum(-1).abs().max() < 1e-5


def test_no_offset_reproduces_the_pinned_profile():
    """The trained models' path must be untouched."""
    s, md, mask = surrogate(), meta(6), torch.ones(6, P)
    with torch.no_grad():
        pinned = s._profile(md, mask)
        p, _ = s.profile_and_grad(md, mask, None)
    assert torch.allclose(pinned, p, atol=1e-7)


def test_analytic_gradient_matches_a_finite_difference():
    s, md, mask = surrogate(), meta(4), torch.ones(4, P)
    mu = torch.full((4, 2), 0.3)
    h = 0.05
    with torch.no_grad():
        _, g = s.profile_and_grad(md, mask, mu)
        for j in range(2):
            up, dn = mu.clone(), mu.clone()
            up[:, j] += h
            dn[:, j] -= h
            p_up, _ = s.profile_and_grad(md, mask, up)
            p_dn, _ = s.profile_and_grad(md, mask, dn)
            fd = (p_up - p_dn) / (2 * h)
            scale = fd.abs().max()
            assert (g[:, j] - fd).abs().max() < 2e-2 * scale, f"direction {j}"


# ----------------------------------------------------------------- recovery


def test_a_planted_shift_is_recovered():
    s, md, mask, counts, true_mu, _, _ = plant(mu=(1.5, -1.0))
    _res, mu = solve_profile_fit_recentered(
        counts, mask, bound(s, md, mask), n_outer=4, shrink=False
    )
    assert mu[:, 0].mean() == pytest.approx(1.5, abs=0.1)
    assert mu[:, 1].mean() == pytest.approx(-1.0, abs=0.1)


def test_a_shift_of_several_pixels_is_recovered():
    """The expansion is local, so this is the test that the iteration works."""
    s, md, mask, counts, _, _, _ = plant(mu=(2.5, 2.0), seed=3)
    _res, mu = solve_profile_fit_recentered(
        counts, mask, bound(s, md, mask), n_outer=6, shrink=False
    )
    assert mu[:, 0].mean() == pytest.approx(2.5, abs=0.15)
    assert mu[:, 1].mean() == pytest.approx(2.0, abs=0.15)


def test_recentring_removes_the_intensity_bias_the_pinned_solve_has():
    s, md, mask, counts, _, i_true, _ = plant(mu=(1.5, -1.0), seed=1)
    with torch.no_grad():
        p_pinned, _ = s.profile_and_grad(md, mask, None)
    pinned = solve_profile_fit(counts, mask, p_pinned)
    moved, _mu = solve_profile_fit_recentered(
        counts, mask, bound(s, md, mask), n_outer=4, shrink=False
    )

    # the defect: a profile in the wrong place collects less of the spot
    assert pinned.intensity.mean() < 0.9 * i_true
    assert moved.intensity.mean() == pytest.approx(i_true, rel=0.03)


def test_a_centred_spot_is_left_where_it_is():
    s, md, mask, counts, _, i_true, _ = plant(mu=(0.0, 0.0), seed=2)
    pinned = solve_profile_fit(counts, mask, s.profile_and_grad(md, mask, None)[0])
    moved, mu = solve_profile_fit_recentered(
        counts, mask, bound(s, md, mask), n_outer=3, shrink=False
    )
    assert mu.abs().mean() < 0.1
    assert moved.intensity.mean() == pytest.approx(
        pinned.intensity.mean(), rel=0.02
    )


def test_locating_the_spot_costs_almost_no_intensity_precision():
    """The shift directions are nearly orthogonal to the profile direction.

    `dp/dmu` is odd about the centre where `p` is even, so the weighted
    cross-information `sum_j w_j p_j dp_j/dmu` nearly cancels and the two extra
    columns take little information away from `I`. That is why the centre can be
    fitted per reflection almost for free.

    A stronger-looking claim -- that adding columns can only enlarge
    `[F^-1]_00` -- is NOT tested here, and does not hold for this solver: the
    IRLS weights are refit, so the two solves do not share an information
    matrix. Measured in float64 the pinned variance came out 4183.78 against
    4183.37 located, which is the weights moving, not a violated inequality.
    """
    s, md, mask, counts, _, _, b_true = plant(mu=(0.0, 0.0), seed=4)
    with torch.no_grad():
        p, g = s.profile_and_grad(md, mask, None)
    p, g, counts, mask = p.double(), g.double(), counts.double(), mask.double()

    pinned = solve_profile_fit(counts, mask, p)
    located = solve_profile_fit(counts, mask, p, shift_cols=g)
    assert torch.allclose(located.variance, pinned.variance, rtol=2e-3)

    # the orthogonality itself, at the fitted rate
    rate = pinned.intensity.unsqueeze(-1) * p + pinned.background.unsqueeze(-1)
    w = mask / rate.clamp_min(1e-6)
    f_ii = (w * p * p).sum(-1)
    for j in range(2):
        f_ia = (w * p * g[:, j]).sum(-1)
        f_aa = (w * g[:, j] * g[:, j]).sum(-1)
        rho = f_ia.abs() / (f_ii * f_aa).sqrt()
        assert rho.max() < 0.05, f"direction {j} correlates with I at {rho.max():.3f}"


# ----------------------------------------------------------------- shrinkage


def test_shrinkage_holds_weak_reflections_at_the_prediction():
    bright = plant(n=128, mu=(1.5, 0.0), intensity=4000.0, seed=5)
    weak = plant(n=128, mu=(1.5, 0.0), intensity=8.0, seed=6)

    out = {}
    for name, (s, md, mask, counts, _, _, _) in (("bright", bright), ("weak", weak)):
        res = solve_profile_fit(
            counts, mask, *s.profile_and_grad(md, mask, None)[:1],
            shift_cols=s.profile_and_grad(md, mask, None)[1],
        )
        delta, w = shrink_shift(res.extras["shift"], res.extras["shift_var"])
        out[name] = (delta, w, res.extras["shift"])

    # the planted shift is radial, so the two directions must be judged apart:
    # a bright spot keeps its radial shift, a weak one does not, and the
    # tangential direction -- where the truth is zero -- is shrunk away in both,
    # which is the behaviour that makes a per-reflection centre safe to fit
    assert out["bright"][1][:, 0].mean() > 0.9
    assert out["bright"][1][:, 1].mean() < 0.1
    assert out["weak"][1][:, 0].mean() < 0.2
    assert out["weak"][0][:, 0].abs().mean() < out["weak"][2][:, 0].abs().mean()


def test_shrinkage_does_not_stop_a_real_shift_being_found():
    s, md, mask, counts, _, _, _ = plant(mu=(1.5, -1.0), intensity=4000.0, seed=7)
    _res, mu = solve_profile_fit_recentered(
        counts, mask, bound(s, md, mask), n_outer=5, shrink=True
    )
    assert mu[:, 0].mean() == pytest.approx(1.5, abs=0.2)
    assert mu[:, 1].mean() == pytest.approx(-1.0, abs=0.2)


# --------------------------------------------------------------- no regression


def test_without_shift_columns_the_solve_is_unchanged():
    s, md, mask, counts, _, _, _ = plant(mu=(0.0, 0.0), seed=8)
    with torch.no_grad():
        p, _ = s.profile_and_grad(md, mask, None)
    a = solve_profile_fit(counts, mask, p)
    b = solve_profile_fit(counts, mask, p, shift_cols=None)
    assert torch.equal(a.intensity, b.intensity)
    assert torch.equal(a.variance, b.variance)
    assert torch.equal(a.background, b.background)
    assert "shift" not in a.extras


# ------------------------------------------------------------------- 3D data


def surrogate_3d(d=5, hw=11, a=1e-6, b=2.0, c=0.0, z=1.5):
    """A rotation-series shoebox: `d` frames of `hw x hw`.

    `c = 0` because the wavelength spread is a Laue term; a monochromatic
    rotation series has no radial excess. `z` is the rocking width along the
    frame axis, which is the direction the third shift lives in.
    """
    return PhysicalProfileSurrogate(
        output_dim=d * hw * hw,
        profile_shape=[d, hw, hw],
        beam_center=None,
        init_a=a,
        init_b=b,
        init_c=max(c, 1e-6),
        init_z=z,
        quad=2,
        learn_globals=False,
    )


def test_a_rotation_series_exposes_three_shift_directions():
    assert surrogate_3d().n_shift_directions == 3
    assert surrogate().n_shift_directions == 2


def test_the_frame_direction_column_also_sums_to_zero():
    s = surrogate_3d()
    n, p = 6, 5 * 11 * 11
    mask = torch.ones(n, p)
    md = {"d": torch.full((n,), 2.0), "wavelength": torch.full((n,), 1.0)}
    for off in (None, torch.full((n, 3), 0.4), torch.full((n, 3), -1.2)):
        with torch.no_grad():
            prof, g = s.profile_and_grad(md, mask, off)
        assert g.shape == (n, 3, p)
        assert torch.allclose(prof.sum(-1), torch.ones(n), atol=1e-5)
        assert g.sum(-1).abs().max() < 1e-5


def test_the_frame_gradient_matches_a_finite_difference():
    s = surrogate_3d()
    n, p = 4, 5 * 11 * 11
    mask = torch.ones(n, p)
    md = {"d": torch.full((n,), 2.0), "wavelength": torch.full((n,), 1.0)}
    mu = torch.full((n, 3), 0.2)
    h = 0.05
    with torch.no_grad():
        _, g = s.profile_and_grad(md, mask, mu)
        up, dn = mu.clone(), mu.clone()
        up[:, 2] += h
        dn[:, 2] -= h
        fd = (
            s.profile_and_grad(md, mask, up)[0]
            - s.profile_and_grad(md, mask, dn)[0]
        ) / (2 * h)
    assert (g[:, 2] - fd).abs().max() < 2e-2 * fd.abs().max()


def test_a_planted_shift_along_the_rocking_axis_is_recovered():
    s = surrogate_3d()
    n, p = 256, 5 * 11 * 11
    mask = torch.ones(n, p)
    md = {"d": torch.full((n,), 2.0), "wavelength": torch.full((n,), 1.0)}
    true_mu = torch.zeros(n, 3)
    true_mu[:, 0], true_mu[:, 2] = 0.8, 1.0
    with torch.no_grad():
        p_true, _ = s.profile_and_grad(md, mask, true_mu)
    counts = torch.poisson(
        3000.0 * p_true + 8.0, generator=torch.Generator().manual_seed(11)
    )

    def fn(off):
        with torch.no_grad():
            return s.profile_and_grad(md, mask, off)

    res, mu = solve_profile_fit_recentered(counts, mask, fn, n_outer=5, shrink=False)
    assert mu.shape == (n, 3)
    assert mu[:, 0].mean() == pytest.approx(0.8, abs=0.1)
    assert mu[:, 1].mean() == pytest.approx(0.0, abs=0.1)
    assert mu[:, 2].mean() == pytest.approx(1.0, abs=0.1)
    assert res.intensity.mean() == pytest.approx(3000.0, rel=0.04)


def test_the_pinned_3d_solve_is_biased_low_on_a_rocking_shift():
    """The defect the third direction exists to remove."""
    s = surrogate_3d()
    n, p = 256, 5 * 11 * 11
    mask = torch.ones(n, p)
    md = {"d": torch.full((n,), 2.0), "wavelength": torch.full((n,), 1.0)}
    true_mu = torch.zeros(n, 3)
    true_mu[:, 2] = 1.2
    with torch.no_grad():
        p_true, _ = s.profile_and_grad(md, mask, true_mu)
        p_pinned, _ = s.profile_and_grad(md, mask, None)
    counts = torch.poisson(
        3000.0 * p_true + 8.0, generator=torch.Generator().manual_seed(12)
    )
    pinned = solve_profile_fit(counts, mask, p_pinned)

    def fn(off):
        with torch.no_grad():
            return s.profile_and_grad(md, mask, off)

    moved, _ = solve_profile_fit_recentered(counts, mask, fn, n_outer=5, shrink=False)
    # a 1.2 px shift against a rocking width of sqrt(1.5) = 1.22 px over five
    # frames costs 7%: the profile is broad enough along this axis that the
    # overlap stays large, so the bias is smaller here than the 10%+ a
    # comparable in-plane shift produces against sigma_tan = 0.88 px
    assert pinned.intensity.mean() < 0.95 * 3000.0
    assert moved.intensity.mean() == pytest.approx(3000.0, rel=0.04)
    assert moved.intensity.mean() > pinned.intensity.mean() * 1.04


def test_the_plane_background_still_works_with_shift_columns():
    from integrator.fit import background_basis

    s, md, mask, counts, _, _, _ = plant(mu=(1.0, 0.5), seed=9)
    bg = background_basis((1, HW, HW), "plane", counts.device, counts.dtype)
    res, mu = solve_profile_fit_recentered(
        counts, mask, bound(s, md, mask), bg_cols=bg, n_outer=3, shrink=False
    )
    assert torch.isfinite(res.intensity).all()
    assert torch.isfinite(res.variance).all()
    assert mu[:, 0].mean() == pytest.approx(1.0, abs=0.15)
