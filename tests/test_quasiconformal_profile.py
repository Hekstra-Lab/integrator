import math

import torch

from integrator.model.distributions.quasiconformal_profile import (
    QuasiconformalProfileSurrogate,
)

BC = (100.0, 100.0)


def _meta(phi_deg, n=1, side=21):
    """Reflections at azimuth `phi_deg`, 60 px from the beam centre, centred in the box."""
    phi = math.radians(phi_deg)
    x, y = BC[0] + 60 * math.cos(phi), BC[1] + 60 * math.sin(phi)
    # box chosen so the predicted centroid sits on the central pixel's centre
    c = (side - 1) / 2
    return {
        "xyzcal.px.0": torch.full((n,), x),
        "xyzcal.px.1": torch.full((n,), y),
        "bbox.0": torch.full((n,), x - c - 0.5),
        "bbox.2": torch.full((n,), y - c - 0.5),
        "wavelength": torch.full((n,), 1.1),
    }


def _surrogate(**kw):
    return QuasiconformalProfileSurrogate(
        8, beam_center=BC, profile_shape=(1, 21, 21), lambda_min=1.0, lambda_max=1.2, **kw
    )


def _second_moments(p, side=21):
    p = p.reshape(side, side)
    yy, xx = torch.meshgrid(torch.arange(side).float(), torch.arange(side).float(), indexing="ij")
    mx, my = (p * xx).sum(), (p * yy).sum()
    cxx = (p * (xx - mx) ** 2).sum()
    cyy = (p * (yy - my) ** 2).sum()
    cxy = (p * (xx - mx) * (yy - my)).sum()
    return mx, my, torch.tensor([[cxx, cxy], [cxy, cyy]])


def test_profiles_normalize_over_the_mask():
    q = _surrogate()
    mask = torch.ones(3, 441)
    mask[1, :100] = 0
    out = q(torch.randn(3, 8), mc_samples=4, metadata=_meta(30, 3), mask=mask)
    assert out.zp.shape == (4, 3, 441)
    assert torch.allclose(out.mean_profile.sum(-1), torch.ones(3), atol=1e-5)
    assert out.mean_profile[1, :100].abs().max() == 0
    assert out.kl.shape == (3,) and torch.all(out.kl >= 0)


def test_positive_mu_elongates_along_the_azimuth():
    q = _surrogate(init_mu=0.6)
    for phi_deg in (0.0, 45.0, 90.0, 135.0):
        p = q(torch.zeros(1, 8), metadata=_meta(phi_deg)).mean_profile[0]
        mx, my, cov = _second_moments(p)
        evals, evecs = torch.linalg.eigh(cov)
        major = evecs[:, -1]
        angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180
        assert abs((angle - phi_deg + 90) % 180 - 90) < 2.0
        # K = 4 for |mu| = 0.6, and the variance ratio is K^2
        assert 12 < float(evals[-1] / evals[0]) < 20
        assert abs(float(mx) - 10) < 0.05 and abs(float(my) - 10) < 0.05


def test_mu_is_bounded_and_the_map_never_folds():
    q = _surrogate()
    theta = torch.randn(64, 11) * 20
    p = q.physical(theta)
    assert torch.all(p["mu"].abs() < q.mu_max + 1e-6)
    tot = p["c"].abs() + p["d"].abs() + p["e"].abs()
    assert torch.all(tot <= 0.45 + 1e-5)


def test_gradient_reaches_the_encoder_heads_and_the_global_offset():
    q = _surrogate()
    out = q(torch.randn(2, 8), mc_samples=3, metadata=_meta(10, 2))
    (out.zp[..., 220].sum() - out.kl.sum()).backward()
    for name in ("loc_head.weight", "scale_head.weight", "theta0", "beta_raw", "shift_head.2.weight"):
        assert dict(q.named_parameters())[name].grad.abs().sum() > 0, name
