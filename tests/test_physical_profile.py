"""The geometry-computed profile, tested on the properties it exists to provide.

It carries no per-reflection latent and no basis matrix, so the two attractors of a
free softmax and the `h -> h/c, W -> cW` gauge should both be structurally absent,
and the widths should follow the measured law sigma^2 = a sin^2(theta) + b.
"""


import torch

from integrator.model.distributions.profile_surrogates import (
    PhysicalProfileSurrogate,
)

H = W = 21
P2 = H * W
BC = (2000.0, 2000.0)


def _meta(b=8, d=None, lam=1.05, x=None, y=None, poly=True):
    m = {
        "d": torch.full((b,), 3.0) if d is None else torch.as_tensor(d, dtype=torch.float32),
        "xyzcal.px.0": torch.full((b,), 2100.0) if x is None else torch.as_tensor(x, dtype=torch.float32),
        "xyzcal.px.1": torch.full((b,), 2000.0) if y is None else torch.as_tensor(y, dtype=torch.float32),
    }
    if poly:
        m["wavelength"] = torch.full((b,), lam)
    return m


def _surr(**kw):
    kw.setdefault("profile_shape", (1, H, W))
    kw.setdefault("output_dim", P2)
    kw.setdefault("beam_center", BC)
    return PhysicalProfileSurrogate(**kw)


def n_eff(p):
    p = p / p.sum(-1, keepdim=True)
    return 1.0 / p.pow(2).sum(-1)


def test_profile_is_a_normalized_density_with_no_latent():
    s = _surr()
    out = s(torch.randn(8, 64), mc_samples=3, metadata=_meta(8),
            mask=torch.ones(8, P2))
    assert out.zp.shape == (3, 8, P2)
    assert (out.zp >= 0).all()
    assert torch.allclose(out.zp.sum(-1), torch.ones(3, 8), atol=1e-5)
    # deterministic given geometry: every draw identical, so no profile entropy
    assert torch.allclose(out.zp[0], out.zp[1])
    assert torch.allclose(out.kl, torch.zeros(8))


def test_profile_ignores_the_encoder_entirely():
    """No per-reflection latent means the shoebox cannot move the profile. That is
    what removes the freedom the intensity was being absorbed into."""
    s = _surr()
    meta, mask = _meta(6), torch.ones(6, P2)
    a = s(torch.randn(6, 64), mc_samples=1, metadata=meta, mask=mask).mean_profile
    b = s(torch.randn(6, 64) * 100, mc_samples=1, metadata=meta, mask=mask).mean_profile
    assert torch.allclose(a, b)


def test_widths_follow_the_measured_law():
    """sigma_tan^2 = a sin^2(theta) + b, and the radial axis carries the extra
    wavelength term."""
    s = _surr(init_a=250.0, init_b=3.0, init_c=2.6)
    d = torch.tensor([2.0, 4.0, 8.0])
    meta = _meta(3, d=d, lam=1.05)
    v_rad, v_tan, _ = s.widths(meta, torch.device("cpu"), torch.float32)

    sin2 = (1.05 / (2 * d)) ** 2
    assert torch.allclose(v_tan, 250.0 * sin2 + 3.0, rtol=1e-5)
    assert torch.allclose(v_rad, v_tan + 2.6, rtol=1e-5)
    assert (v_tan[0] > v_tan[-1]).item()      # wider at high resolution


def test_monochromatic_drops_the_wavelength_term():
    """c acts along the radial direction only and has no meaning without a
    wavelength spread, so a mono batch must come out round."""
    s = _surr(init_c=50.0, wavelength=0.98)
    meta = _meta(4, poly=False)
    v_rad, v_tan, _ = s.widths(meta, torch.device("cpu"), torch.float32)
    assert torch.allclose(v_rad, v_tan)


def test_streak_points_along_the_radial_direction():
    """The long axis must follow the azimuth about the beam centre, which is what
    a learned round profile cannot do."""
    s = _surr(init_a=0.0, init_b=2.0, init_c=18.0, quad=1)
    # one reflection out along +x, one along +y
    meta = _meta(2, x=[2300.0, 2000.0], y=[2000.0, 2300.0])
    prof = s._profile(meta, torch.ones(2, P2)).reshape(2, H, W)

    for i, axis in enumerate(("x", "y")):
        img = prof[i]
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
        w = img / img.sum()
        cy, cx = (w * yy).sum(), (w * xx).sum()
        vxx = (w * (xx - cx) ** 2).sum()
        vyy = (w * (yy - cy) ** 2).sum()
        if axis == "x":
            assert vxx > 3 * vyy, f"+x reflection not elongated along x: {vxx:.2f} vs {vyy:.2f}"
        else:
            assert vyy > 3 * vxx, f"+y reflection not elongated along y: {vyy:.2f} vs {vxx:.2f}"


def test_never_delta_or_uniform_at_the_measured_parameters():
    """The property the surrogate exists for, checked at the globals actually
    measured on each dataset and over its real resolution range.

    HEWL Laue (25x25 box): a = 251, b = 3.15, c = 2.6 px^2.
    HEWL monochromatic (21x21 box): a = 9.4, b = 0.91, no wavelength term.
    """
    cases = [
        ("1118 Laue", (1, 25, 25), dict(init_a=251.0, init_b=3.15, init_c=2.6), 1.6, 1.05, True),
        ("9b7c mono", (1, 21, 21), dict(init_a=9.4, init_b=0.91, init_c=0.0), 1.3, 0.98, False),
    ]
    for name, shape, globals_, d_min, lam, poly in cases:
        npix = shape[1] * shape[2]
        s = PhysicalProfileSurrogate(
            output_dim=npix, profile_shape=shape, beam_center=BC,
            wavelength=None if poly else lam, quad=2, **globals_,
        )
        n = 40
        meta = {
            "d": torch.linspace(d_min, 20.0, n),
            "xyzcal.px.0": torch.linspace(1700.0, 2400.0, n),
            "xyzcal.px.1": torch.full((n,), 2050.0),
        }
        if poly:
            meta["wavelength"] = torch.full((n,), lam)
        ne = n_eff(s._profile(meta, torch.ones(n, npix)))
        assert ne.min() > 2.0, f"{name}: delta, n_eff {ne.min():.2f}"
        assert ne.max() < 0.75 * npix, f"{name}: flat, n_eff {ne.max():.1f} of {npix}"


def test_the_box_must_be_large_enough_for_the_widest_spot():
    """Not a property of the profile but of the extraction: a spot wider than its
    shoebox is truncated whatever models it. At a = 2000 px^2 -- eight times the
    largest measured value -- the 1.5 A spot has sigma 16 px, which no 21 px box
    can hold. Recorded so the guarantee above is not read as unconditional."""
    s = _surr(init_a=2000.0, init_b=10.0, quad=1)
    meta = _meta(1, d=torch.tensor([1.5]), lam=1.05)
    v_rad, v_tan, _ = s.widths(meta, torch.device("cpu"), torch.float32)
    assert v_tan.item() ** 0.5 > H / 2, "this case is only interesting if it overruns"
    ne = n_eff(s._profile(meta, torch.ones(1, P2))).item()
    assert ne > 0.6 * P2


def test_pixel_integration_matters_for_narrow_spots():
    """Monochromatic spots sit at sigma 0.76-1.27 px, where sampling pixel centres
    misses mass that quadrature recovers."""
    meta, mask = _meta(1, poly=False), torch.ones(1, P2)
    centre = _surr(init_a=0.0, init_b=0.6, wavelength=0.98, quad=1)
    quadr = _surr(init_a=0.0, init_b=0.6, wavelength=0.98, quad=3)
    p1 = centre._profile(meta, mask)
    p3 = quadr._profile(meta, mask)
    assert n_eff(p3).item() > n_eff(p1).item()
    assert abs(n_eff(p3).item() - n_eff(p1).item()) > 0.2


def test_masked_pixels_carry_no_profile_mass():
    s = _surr()
    mask = torch.ones(2, P2)
    mask[:, : P2 // 2] = 0.0
    prof = s._profile(_meta(2), mask)
    assert torch.allclose(prof[:, : P2 // 2], torch.zeros(2, P2 // 2))
    assert torch.allclose(prof.sum(-1), torch.ones(2), atol=1e-5)


def test_globals_are_learnable_and_few():
    s = _surr()
    trainable = [n for n, p in s.named_parameters() if p.requires_grad]
    assert set(trainable) == {"log_a", "log_b", "log_c", "log_z"}

    out = s(torch.randn(4, 64), mc_samples=2, metadata=_meta(4), mask=torch.ones(4, P2))
    out.zp.sum().backward()
    assert s.log_a.grad is not None and s.log_b.grad is not None


def test_globals_can_be_frozen_for_a_warm_start():
    s = _surr(learn_globals=False)
    assert not any(p.requires_grad for p in s.parameters())


def test_three_dimensional_shoebox():
    s = PhysicalProfileSurrogate(
        output_dim=3 * 11 * 11, profile_shape=(3, 11, 11),
        beam_center=BC, wavelength=0.98, quad=2,
    )
    out = s(torch.randn(5, 64), mc_samples=2, metadata=_meta(5, poly=False),
            mask=torch.ones(5, 363))
    assert out.zp.shape == (2, 5, 363)
    assert torch.allclose(out.zp.sum(-1), torch.ones(2, 5), atol=1e-5)


def test_extreme_globals_broaden_but_cannot_go_uniform():
    """Honest limit of the guarantee: a large enough `b` does make the profile fill
    its box. The difference from a free softmax is that uniform is not an ATTRACTOR
    here -- reaching it needs a global width an order of magnitude past the measured
    value, and that single number is fitted against every bright reflection at once,
    which is exactly the leverage a per-reflection latent does not have."""
    s = _surr(init_a=0.0, init_b=30.0, quad=1)
    prof = s._profile(_meta(1), torch.ones(1, P2))
    ne = n_eff(prof).item()
    assert ne > 0.7 * P2, "a sigma of 5.5 px in a 21 px box should be broad"

    # even then it is a centred Gaussian, not the uniform profile: the peak still
    # exceeds the corner by a clear margin, which `softmax` collapse does not
    img = prof.reshape(H, W)
    assert (img[H // 2, W // 2] / img[0, 0]).item() > 2.0


def test_full_factory_path_and_backward():
    """Through resolve_config/construct_integrator: the resolver must supply
    output_dim, profile_shape and the beam centre itself."""
    from integrator.utils.factory_utils import construct_integrator, resolve_config

    cfg = {
        "integrator": {"name": "hierarchical", "args": {
            "data_dim": "2d", "d": 1, "h": 21, "w": 21,
            "encoder_out": 64, "mc_samples": 2}},
        "loss": {"name": "polychromatic_wilson", "args": {
            "degree": 4, "lambda_min": 0.95, "lambda_max": 1.2,
            "init_B": 30.0, "n_bins": 1}},
        "surrogates": {"qp": {"name": "physical_profile", "args": {
            "beam_center": [2000.0, 2000.0], "init_a": 251.0,
            "init_b": 3.15, "init_c": 2.6}}},
        "data_loader": {"name": "polychromatic_data", "args": {"data_dir": None}},
        "optimizer": {"lr": 1e-3},
    }
    model = construct_integrator(resolve_config(cfg))
    assert type(model.surrogates["qp"]).__name__ == "PhysicalProfileSurrogate"

    torch.manual_seed(0)
    b = 6
    counts = torch.poisson(torch.full((b, P2), 12.0))
    meta = {
        "d": torch.full((b,), 2.5), "group_label": torch.zeros(b),
        "wavelength": torch.full((b,), 1.05),
        "xyzcal.px.0": torch.linspace(1900.0, 2200.0, b),
        "xyzcal.px.1": torch.full((b,), 2050.0),
    }
    batch = (counts, counts.clone(), torch.ones(b, P2), meta)

    out = model(*batch)
    rates = out["forward_out"]["rates"]
    assert torch.isfinite(rates).all() and (rates > 0).all()
    assert torch.allclose(out["forward_out"]["qp_mean"].sum(-1), torch.ones(b), atol=1e-4)

    step = model._step(batch, step="train")
    assert torch.isfinite(step["loss"])
    step["loss"].backward()
    qp = model.surrogates["qp"]
    assert qp.log_a.grad is not None, "the globals must receive gradient from the ELBO"
