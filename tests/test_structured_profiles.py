"""Tests for the two structured profile families.

They impose different priors at comparable cost -- Tucker says the basis is
separable and low rank, the control mesh says the profile is a smooth
deformation of one template -- so the properties worth testing are the ones
that make each prior bite, not that the output happens to be a density.
"""

import math

import pytest
import torch

from integrator.model.distributions import (
    ControlMeshProfileSurrogate,
    TuckerBasisProfileSurrogate,
)
from integrator.registry import REGISTRY

SHAPE_3D = [3, 21, 21]
SHAPE_2D = [1, 41, 41]
BEAM = [1987.604697, 1965.00825]


def meta(n):
    return {
        "xyzcal.px.0": torch.linspace(1500.0, 2500.0, n),
        "xyzcal.px.1": torch.linspace(1900.0, 2600.0, n),
        "d": torch.linspace(12.0, 1.5, n),
    }


def counts_for(n, p):
    torch.manual_seed(3)
    return torch.poisson(torch.rand(n, p) * 6.0)


def tucker(shape=SHAPE_3D, **kw):
    p = shape[0] * shape[1] * shape[2]
    kw.setdefault("ranks", (3, 6, 6))
    return TuckerBasisProfileSurrogate(
        input_dim=32, latent_dim=8, output_dim=p, profile_shape=shape,
        beam_center=BEAM, **kw
    )


def mesh(shape=SHAPE_2D, **kw):
    p = shape[0] * shape[1] * shape[2]
    kw.setdefault("lattice", (1, 4, 4))
    return ControlMeshProfileSurrogate(
        input_dim=32, output_dim=p, profile_shape=shape, beam_center=BEAM,
        template_sigma=3.0, **kw
    )


def run(m, shape, n=5, mc=3):
    p = shape[0] * shape[1] * shape[2]
    return m(
        torch.randn(n, 32),
        mc_samples=mc,
        metadata=meta(n),
        mask=torch.ones(n, p, dtype=torch.bool),
        counts=counts_for(n, p),
    )


def test_both_names_resolve():
    assert REGISTRY["surrogates"]["tucker_basis_profile"] is (
        TuckerBasisProfileSurrogate
    )
    assert REGISTRY["surrogates"]["control_mesh_profile"] is (
        ControlMeshProfileSurrogate
    )


@pytest.mark.parametrize(
    "build,shape",
    [(tucker, SHAPE_3D), (mesh, SHAPE_2D)],
)
def test_the_output_is_a_density(build, shape):
    torch.manual_seed(0)
    out = run(build(), shape)
    p = shape[0] * shape[1] * shape[2]
    assert out.zp.shape == (3, 5, p)
    assert (out.zp >= 0).all()
    assert torch.allclose(out.zp.sum(-1), torch.ones(3, 5), atol=1e-5)
    assert torch.allclose(out.mean_profile.sum(-1), torch.ones(5), atol=1e-5)


# ---------------------------------------------------------------- Tucker


def test_the_tucker_basis_really_is_low_rank():
    """The point of the structure. Each basis vector, reshaped to its grid and
    unfolded along an axis, must have rank at most that axis's Tucker rank --
    which is what forbids a speckle basis vector while leaving elongation and
    skew reachable, since those are separable."""
    torch.manual_seed(0)
    ranks = (3, 5, 5)
    m = tucker(ranks=ranks)
    basis = m.basis().reshape(m.d, *SHAPE_3D)
    for axis, r in zip((1, 2, 3), ranks):
        unfolded = basis.movedim(axis, 1).reshape(m.d, SHAPE_3D[axis - 1], -1)
        for vec in unfolded:
            assert torch.linalg.matrix_rank(vec, rtol=1e-4) <= r


def test_the_tucker_basis_costs_far_less_than_a_free_one():
    for ranks, expected_max in (((3, 4, 4), 0.1), ((3, 8, 8), 0.2)):
        m = tucker(ranks=ranks)
        structured = (
            m.core.numel() + m.u_d.numel() + m.u_h.numel() + m.u_w.numel()
        )
        free = m.d * (SHAPE_3D[0] * SHAPE_3D[1] * SHAPE_3D[2])
        assert structured < expected_max * free


def _wake_head(m):
    """Randomize the head's output layer.

    It is zero-initialized so a fresh model starts at one shared profile, which
    makes any test of what the features do vacuous: every input gives the same
    answer whether the wiring is right or not.
    """
    with torch.no_grad():
        m.head[-1].weight.normal_(0.0, 0.3)
    return m


def _profiles_for(m, x):
    p = SHAPE_3D[0] * SHAPE_3D[1] * SHAPE_3D[2]
    torch.manual_seed(1)
    return m(
        x,
        mc_samples=1,
        metadata=meta(x.shape[0]),
        mask=torch.ones(x.shape[0], p, dtype=torch.bool),
        counts=counts_for(x.shape[0], p),
    ).mean_profile


def test_tying_the_encoder_ignores_the_upstream_representation():
    """`softmax(Wh + b)` is an exponential family, so the ML `h` depends on the
    counts only through `W^T c`. A tied encoder therefore needs nothing from the
    upstream encoder, and must not silently use it."""
    torch.manual_seed(0)
    m = _wake_head(tucker(tie_encoder=True))
    torch.manual_seed(2)
    x = torch.randn(5, 32)
    assert torch.equal(_profiles_for(m, x), _profiles_for(m, x * 7.0))


def test_an_untied_encoder_does_use_it():
    torch.manual_seed(0)
    m = _wake_head(tucker(tie_encoder=False))
    torch.manual_seed(2)
    x = torch.randn(5, 32)
    a, b = _profiles_for(m, x), _profiles_for(m, x * 7.0)
    assert not torch.allclose(a, b)


def test_the_tucker_family_needs_the_shoebox():
    m = tucker()
    with pytest.raises(ValueError, match="counts"):
        m(torch.randn(4, 32), mc_samples=1, metadata=meta(4))


def test_the_conv_stub_is_small_and_optional():
    plain, with_conv = tucker(conv_channels=0), tucker(conv_channels=16)
    assert with_conv.n_feat == plain.n_feat + 16
    added = sum(p.numel() for p in with_conv.parameters()) - sum(
        p.numel() for p in plain.parameters()
    )
    # a two-layer stub, not a deep encoder: the measured alternative was
    # 758,368 parameters for a 12-dimensional output
    assert added < 10000


# ---------------------------------------------------------------- mesh


def test_a_pure_translation_costs_no_bending_energy():
    """The discrete Dirichlet energy of the lattice must see translation as
    free, because a constant displacement field is not a deformation. Anything
    else would penalize off-centring, which is the one thing the warp is
    supposed to handle for nothing."""
    torch.manual_seed(0)
    m = mesh()
    shift = torch.zeros(2, m.d)
    shift[:, : m.n_ctrl] = 1.7  # the same displacement at every control point
    assert m.bending_energy(shift).abs().max() < 1e-6
    rough = torch.randn(2, m.d)
    assert (m.bending_energy(rough) > 1e-3).all()


def test_pinning_the_log_determinant_removes_the_concentration_change():
    """`det J` is the local area change, so the mean of `log det J` is the
    profile's concentration -- the direction the counts cannot identify, since
    a 10x intensity error costs zero log-likelihood against a free flat
    background. Pinned, a warp may shear and translate but not concentrate."""
    torch.manual_seed(0)
    m = mesh(pin_log_det=True)
    disp = 0.6 * torch.randn(4, m.d)
    _, log_det = m._warp(disp)
    assert log_det.mean(dim=-1).abs().max() < 1e-5

    loose = mesh(pin_log_det=False)
    loose.load_state_dict(m.state_dict())
    _, log_det_loose = loose._warp(disp)
    assert log_det_loose.mean(dim=-1).abs().max() > 1e-4


def test_a_constant_displacement_translates_the_profile():
    torch.manual_seed(0)
    m = mesh(pin_log_det=True)
    n_px = SHAPE_2D[1]
    base, _ = m._warp(torch.zeros(1, m.d))
    shift = torch.zeros(1, m.d)
    # the x displacement block is the last of the three
    shift[:, 2 * m.n_ctrl :] = 3.0
    moved, _ = m._warp(shift)

    coord = torch.linspace(-1.0, 1.0, n_px)
    cx_base = (base.reshape(1, SHAPE_2D[1], n_px).sum(1) * coord).sum()
    cx_moved = (moved.reshape(1, SHAPE_2D[1], n_px).sum(1) * coord).sum()
    assert abs(float(cx_moved - cx_base)) > 0.05


def test_the_bending_energy_reaches_the_loss_through_kl():
    torch.manual_seed(0)
    stiff = mesh(bending_weight=10.0)
    free = mesh(bending_weight=0.0)
    free.load_state_dict(stiff.state_dict())
    with torch.no_grad():
        for mod in (stiff, free):
            mod.head[-1].weight.normal_(0.0, 0.5)
            mod.head[-1].bias[: mod.d].normal_(0.0, 0.5)
        free.head[-1].weight.copy_(stiff.head[-1].weight)
        free.head[-1].bias.copy_(stiff.head[-1].bias)
    torch.manual_seed(5)
    a = run(stiff, SHAPE_2D).kl
    torch.manual_seed(5)
    b = run(free, SHAPE_2D).kl
    assert (a > b).all()


def test_the_lattice_is_clipped_to_the_grid():
    """A depth-1 grid cannot carry a depth-4 control lattice."""
    m = mesh(shape=SHAPE_2D, lattice=(4, 4, 4))
    assert m.lattice[0] == 1
    assert m.n_ctrl == 16
    assert m.d == 48


def test_the_mesh_costs_about_what_a_tucker_basis_does():
    """They are alternatives, so the comparison has to be at matched cost."""
    n_mesh = sum(p.numel() for p in mesh().parameters())
    n_tucker = sum(p.numel() for p in tucker(shape=SHAPE_2D).parameters())
    assert 0.2 < n_mesh / n_tucker < 5.0
    assert math.isfinite(n_mesh / n_tucker)


# ---------------------------------------------------------------- steerable


def steerable(**kw):
    from integrator.model.distributions import SteerableProfileSurrogate

    kw.setdefault("max_order", 4)
    p = SHAPE_2D[0] * SHAPE_2D[1] * SHAPE_2D[2]
    return SteerableProfileSurrogate(
        input_dim=32, output_dim=p, profile_shape=SHAPE_2D, beam_center=BEAM, **kw
    )


def mixture(**kw):
    from integrator.model.distributions import MixtureGaussianProfileSurrogate

    kw.setdefault("n_components", 4)
    p = SHAPE_2D[0] * SHAPE_2D[1] * SHAPE_2D[2]
    return MixtureGaussianProfileSurrogate(
        input_dim=32, output_dim=p, profile_shape=SHAPE_2D, beam_center=BEAM, **kw
    )


@pytest.mark.parametrize("build", [steerable, mixture])
def test_the_gradient_is_finite_with_a_pixel_at_the_origin(build):
    """A centred odd grid puts a pixel at exactly (0, 0), and the pose starts
    unshifted, so the polar coordinate is evaluated at the origin. `hypot` and
    `atan2` are finite there but their gradients are 0/0: the forward pass
    looked healthy and training produced NaN after one step."""
    torch.manual_seed(0)
    m = build()
    p = SHAPE_2D[0] * SHAPE_2D[1] * SHAPE_2D[2]
    n = 4
    out = m(
        torch.randn(n, 32),
        mc_samples=2,
        metadata=meta(n),
        mask=torch.ones(n, p, dtype=torch.bool),
        counts=counts_for(n, p),
    )
    assert torch.isfinite(out.zp).all()
    out.zp.square().sum().backward()
    for name, prm in m.named_parameters():
        if prm.grad is not None:
            assert torch.isfinite(prm.grad).all(), f"non-finite grad in {name}"


def test_the_zernike_basis_is_steerable():
    """Rotating the coordinates by alpha must mix the cos/sin pair of angular
    order m by a plane rotation through m*alpha, exactly. That identity is why
    orientation costs this family nothing."""
    from integrator.model.distributions.profile_surrogates import zernike_basis

    torch.manual_seed(0)
    u = torch.rand(500) * 1.4 - 0.7
    v = torch.rand(500) * 1.4 - 0.7
    alpha = 0.7
    psi, names = zernike_basis(u, v, 4)
    ur = math.cos(alpha) * u + math.sin(alpha) * v
    vr = -math.sin(alpha) * u + math.cos(alpha) * v
    psir, _ = zernike_basis(ur, vr, 4)

    by_nm: dict[tuple[int, int], dict[str, int]] = {}
    for j, (n, mm, kind) in enumerate(names):
        by_nm.setdefault((n, mm), {})[kind] = j
    checked = 0
    for (n, mm), cols in by_nm.items():
        if mm == 0 or "s" not in cols:
            continue
        c, s_ = psi[:, cols["c"]], psi[:, cols["s"]]
        cr, sr = psir[:, cols["c"]], psir[:, cols["s"]]
        ca, sa = math.cos(mm * alpha), math.sin(mm * alpha)
        assert torch.allclose(cr, ca * c + sa * s_, atol=1e-5)
        assert torch.allclose(sr, -sa * c + ca * s_, atol=1e-5)
        checked += 1
    assert checked >= 3


def test_the_mixture_does_not_hold_every_component_at_once():
    """`(S, B, K, P)` intermediates, kept by autograd across the quadrature,
    asked the 20 GiB slice for 658 MiB at K=4 and 986 MiB at K=6. Peak memory
    must grow with K far more slowly than that."""
    import tracemalloc

    p = SHAPE_2D[0] * SHAPE_2D[1] * SHAPE_2D[2]
    peaks = {}
    for k in (2, 8):
        torch.manual_seed(0)
        m = mixture(n_components=k)
        tracemalloc.start()
        out = m(
            torch.randn(6, 32),
            mc_samples=3,
            metadata=meta(6),
            mask=torch.ones(6, p, dtype=torch.bool),
            counts=counts_for(6, p),
        )
        out.zp.sum().backward()
        peaks[k] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    # a per-component loop makes peak memory roughly flat in K; holding all of
    # them at once would make it grow about four-fold from K=2 to K=8
    assert peaks[8] < 2.5 * peaks[2]


@pytest.mark.parametrize(
    "build", [steerable, mixture, tucker, mesh,
              lambda: __import__("integrator.model.distributions", fromlist=["x"]).HellingerBasisProfileSurrogate(
                  input_dim=32, output_dim=SHAPE_2D[1] * SHAPE_2D[2],
                  profile_shape=SHAPE_2D)]
)
def test_the_state_dict_round_trips(build):
    """Save and reload every surrogate.

    `torch.meshgrid` returns expanded views, so a buffer registered as
    `zz.reshape(-1)` can be many elements over one storage element with stride
    zero. That trains perfectly and only fails when `load_state_dict` tries to
    copy into it -- which is at PREDICT, after the full training cost has been
    paid. A forward-pass test cannot see it; only a round trip can.
    """
    torch.manual_seed(0)
    a = build()
    torch.manual_seed(1)
    b = build()
    b.load_state_dict(a.state_dict())
    for (na, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(pa, pb), f"{na} did not round trip"


def test_the_steerable_profile_is_not_flat_at_initialization():
    """The flat profile is an attractor and `Z_0^0` sits exactly on it.

    `Z_0^0` is 1 over the whole disc, so a canonical coefficient vector of
    `[1, 0, ...]` IS the uniform-disc profile, and a projection of the RAW
    shoebox on 96.6%-background Laue data is nearly uniform too. The first
    trained arms collapsed to n_eff 928 of 1681 against a uniform profile's
    1681. Both are fixed here, and both are only visible on a realistic
    background-dominated box -- on a clean one the profile looks fine either
    way.
    """
    from integrator.model.distributions import SteerableProfileSurrogate

    n_px = 41
    p_tot = n_px * n_px
    yy, xx = torch.meshgrid(
        torch.arange(n_px) - 20.0, torch.arange(n_px) - 20.0, indexing="ij"
    )
    spot = torch.exp(-0.5 * ((yy / 1.5) ** 2 + (xx / 4.0) ** 2)).reshape(-1)
    spot = spot / spot.sum()
    n = 8
    # 900 signal photons against 22 per pixel over 1681 pixels: 97% background
    counts = torch.poisson(900.0 * spot.expand(n, p_tot) + 22.0)

    torch.manual_seed(0)
    m = SteerableProfileSurrogate(
        input_dim=32, max_order=4, output_dim=p_tot,
        profile_shape=[1, n_px, n_px], base_radius=8.0, beam_center=BEAM,
    )
    out = m(
        torch.randn(n, 32), mc_samples=1, metadata=meta(n),
        mask=torch.ones(n, p_tot, dtype=torch.bool), counts=counts,
    )
    n_eff = (1.0 / out.mean_profile.pow(2).sum(-1)).median()
    assert n_eff < 0.3 * p_tot, f"started near-uniform: n_eff {n_eff:.0f}"
    core = ((yy.abs() <= 2) & (xx.abs() <= 2)).reshape(-1)
    assert out.mean_profile[:, ~core].sum(-1).median() < 0.9


def test_the_depth_axis_gets_a_free_simplex_when_it_is_thin():
    """Three frames cannot identify a parametric rocking curve -- the measured
    width was 0.40 px, sub-frame -- so the depth axis is a free categorical,
    which costs the same D-1 numbers and cannot be wrong about the shape."""
    from integrator.model.distributions import SteerableProfileSurrogate

    p3 = 3 * 21 * 21
    thin = SteerableProfileSurrogate(
        input_dim=32, max_order=4, output_dim=p3, profile_shape=[3, 21, 21],
        base_radius=5.0, beam_center=BEAM, depth_mode="simplex",
    )
    flat = SteerableProfileSurrogate(
        input_dim=32, max_order=4, output_dim=1681,
        profile_shape=[1, 41, 41], beam_center=BEAM, depth_mode="simplex",
    )
    assert thin.depth_head is not None
    assert flat.depth_head is None, "a depth-1 grid has no rocking axis"
    with pytest.raises(ValueError, match="depth_mode"):
        SteerableProfileSurrogate(
            input_dim=32, output_dim=p3, profile_shape=[3, 21, 21],
            beam_center=BEAM, depth_mode="gaussian",
        )


def shifted(**kw):
    from integrator.model.distributions import ShiftedBasisProfileSurrogate

    p = SHAPE_2D[0] * SHAPE_2D[1] * SHAPE_2D[2]
    kw.setdefault("latent_dim", 24)
    return ShiftedBasisProfileSurrogate(
        input_dim=32, output_dim=p, profile_shape=SHAPE_2D, beam_center=BEAM, **kw
    )


def test_the_shift_leaves_the_basis_free():
    """The shift is added to the learned basis, not a constraint on it.

    The measured case for this: the plain learned basis beat every structured
    family on shape agreement, off-centring and asymmetry, so anything added
    must not reduce what it can represent. `softmax(W h + b)` is untouched;
    only a translation is applied afterwards.
    """
    torch.manual_seed(0)
    m = shifted()
    assert m.decoder.weight.shape == (SHAPE_2D[1] * SHAPE_2D[2], 24)
    p_px = SHAPE_2D[1] * SHAPE_2D[2]
    out = m(
        torch.randn(5, 32), mc_samples=2, metadata=meta(5),
        mask=torch.ones(5, p_px, dtype=torch.bool), counts=counts_for(5, p_px),
    )
    assert (out.zp >= 0).all()
    assert torch.allclose(out.zp.sum(-1), torch.ones(2, 5), atol=1e-5)


def test_the_multiscale_prior_prices_detail_without_forbidding_it():
    """Fine-scale groups get rougher columns AND a tighter prior, so detail
    costs more KL and appears only where the evidence pays for it. Nothing is
    forbidden -- every profile the free basis could reach is still reachable,
    at a price."""
    torch.manual_seed(0)
    groups = [6, 6, 12]
    m = shifted(scale_groups=groups, tau_decay=0.45)
    tau = m.tau.tolist()
    w = m.decoder.weight.detach().reshape(SHAPE_2D[1], SHAPE_2D[2], -1)
    rough = ((w[1:] - w[:-1]) ** 2).mean(dim=(0, 1))

    i, taus, roughs = 0, [], []
    for n in groups:
        taus.append(tau[i])
        roughs.append(float(rough[i : i + n].mean()))
        i += n
    # the prior tightens and the columns get rougher, in step
    assert taus[0] > taus[1] > taus[2]
    assert roughs[0] < roughs[1] < roughs[2]

    p_px = SHAPE_2D[1] * SHAPE_2D[2]
    out = m(
        torch.randn(5, 32), mc_samples=2, metadata=meta(5),
        mask=torch.ones(5, p_px, dtype=torch.bool), counts=counts_for(5, p_px),
    )
    # a per-dimension prior cannot be expressed by one scalar scale, so the
    # surrogate has to supply the KL itself
    assert out.kl is not None and out.kl.shape == (5,)
    assert torch.isfinite(out.kl).all()


def test_scale_groups_must_cover_the_latent():
    with pytest.raises(ValueError, match="scale_groups"):
        shifted(latent_dim=24, scale_groups=[4, 4])


@pytest.mark.parametrize("kind", ["rff", "rp"])
def test_the_fixed_embedding_replaces_the_encoder_and_trains_nothing(kind):
    """A training-free shoebox code in place of the learned encoder.

    `rff` is the kernel mean embedding: a shoebox is a distribution over pixel
    positions and for a characteristic kernel that embedding is injective, so
    distinct profiles give distinct codes. `rp` is a random projection, which
    carries the Johnson-Lindenstrauss distance guarantee. Neither has any
    trainable parameters of its own, and the surrogate must ignore the
    upstream encoder entirely once one is set.
    """
    p_px = SHAPE_2D[1] * SHAPE_2D[2]
    torch.manual_seed(0)
    m = shifted(latent_dim=12, fixed_embedding=32, fixed_embedding_kind=kind)
    assert m._embed is not None
    assert not m._embed.requires_grad
    kw = dict(
        mc_samples=1, metadata=meta(5),
        mask=torch.ones(5, p_px, dtype=torch.bool), counts=counts_for(5, p_px),
    )
    torch.manual_seed(1)
    a = m(torch.randn(5, 32), **kw).mean_profile
    torch.manual_seed(1)
    b = m(torch.randn(5, 32) * 50.0, **kw).mean_profile
    assert torch.equal(a, b), "the encoder output still reaches the profile"

    # and it does depend on the shoebox, which is the whole point
    kw2 = dict(kw)
    kw2["counts"] = counts_for(5, p_px) * 3.0 + 7.0
    torch.manual_seed(1)
    c = m(torch.randn(5, 32), **kw2).mean_profile
    assert not torch.allclose(a, c)


def test_the_fixed_embedding_needs_the_shoebox():
    m = shifted(latent_dim=12, fixed_embedding=32)
    with pytest.raises(ValueError, match="counts"):
        m(torch.randn(3, 32), mc_samples=1, metadata=meta(3))
