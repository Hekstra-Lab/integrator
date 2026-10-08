"""The two non-degenerate profile families, tested on the properties they exist for.

A free `softmax(Wh + b)` has two attractors, one-hot and uniform, and a gauge
`h -> h/c, W -> cW` that makes its KL weight meaningless. These check that the
replacements genuinely close both doors rather than merely looking different.
"""

import math

import pytest
import torch

from integrator.model.distributions.profile_surrogates import (
    FixedBasisProfileSurrogate,
    ProfileSurrogate,
    ShapeletProfileSurrogate,
    _monomial_basis,
)


def n_eff(p: torch.Tensor) -> torch.Tensor:
    p = p / p.sum(-1, keepdim=True)
    return 1.0 / p.pow(2).sum(-1)


# ---------------------------------------------------------------- fixed basis


def test_monomial_basis_shape_and_gauge_fixing():
    """Quadratic monomials give 9 columns in 3D and 5 in 2D, each unit norm."""
    basis3, exp3 = _monomial_basis((3, 7, 7), max_order=2)
    assert basis3.shape == (3 * 7 * 7, 9)
    assert len(exp3) == 9

    basis2, exp2 = _monomial_basis((1, 7, 7), max_order=2)
    assert basis2.shape == (49, 5)
    assert all(i == 0 for i, _, _ in exp2)

    for b in (basis2, basis3):
        assert torch.allclose(b.norm(dim=0), torch.ones(b.shape[1]), atol=1e-5)


def test_fixed_basis_decoder_is_frozen():
    """The basis must not train, or the gauge comes back."""
    s = FixedBasisProfileSurrogate(input_dim=8, output_dim=49, profile_shape=(1, 7, 7))
    assert not s.decoder.weight.requires_grad
    assert s.decoder.bias.requires_grad
    assert s.d == 5

    before = s.decoder.weight.clone()
    out = s(torch.randn(4, 8), mc_samples=2)
    out.zp.sum().backward()
    assert torch.equal(s.decoder.weight, before)
    assert s.decoder.weight.grad is None


def test_fixed_basis_recovers_a_gaussian():
    """A quadratic log-profile IS an anisotropic Gaussian, so a hand-set latent
    must reproduce one to high accuracy."""
    H = W = 21
    s = FixedBasisProfileSurrogate(input_dim=4, output_dim=H * W, profile_shape=(1, H, W))
    # columns are (y, x, y^2, yx, x^2) up to ordering; drive only the squares
    idx = {e: i for i, e in enumerate(s.exponents)}
    h = torch.zeros(1, s.d)
    h[0, idx[(0, 2, 0)]] = -40.0
    h[0, idx[(0, 0, 2)]] = -40.0

    logits = h @ s.decoder.weight.T + s.decoder.bias
    prof = torch.softmax(logits, -1).reshape(H, W)

    assert prof.argmax().item() == (H // 2) * W + W // 2
    row = prof[H // 2]
    assert (row[: W // 2].diff() > 0).all()
    assert (row[W // 2 :].diff() < 0).all()


def test_fixed_basis_cannot_reach_a_delta_at_moderate_logits():
    """A free softmax hits n_eff ~ 1 with modest random logits; a quadratic basis
    at the same latent scale stays an order of magnitude broader."""
    torch.manual_seed(0)
    H = W = 21
    P = H * W
    free = ProfileSurrogate(input_dim=8, latent_dim=5, output_dim=P)
    fixed = FixedBasisProfileSurrogate(
        input_dim=8, output_dim=P, profile_shape=(1, H, W)
    )

    h = torch.randn(256, 5) * 3.0
    free_prof = torch.softmax(h @ free.decoder.weight.T + free.decoder.bias, -1)
    fixed_prof = torch.softmax(h @ fixed.decoder.weight.T + fixed.decoder.bias, -1)

    assert n_eff(free_prof).median() < 0.25 * n_eff(fixed_prof).median()


# ------------------------------------------------------------------- shapelet


def test_shapelet_profile_is_a_normalized_density():
    s = ShapeletProfileSurrogate(
        input_dim=8, output_dim=21 * 21, profile_shape=(1, 21, 21), quad=2
    )
    out = s(torch.randn(5, 8), mc_samples=3)

    assert out.zp.shape == (3, 5, 441)
    assert out.mean_profile.shape == (5, 441)
    assert (out.zp >= 0).all()
    assert torch.allclose(out.zp.sum(-1), torch.ones(3, 5), atol=1e-5)
    assert torch.allclose(out.mean_profile.sum(-1), torch.ones(5), atol=1e-5)


def test_shapelet_at_zero_latent_is_the_envelope_gaussian():
    """eps = 0 leaves c = (1, 0, ...), i.e. the order-0 Hermite squared, which is a
    Gaussian of width beta/sqrt(2). That is the no-information fallback, and it must
    be a spot rather than a flat box."""
    H = W = 21
    s = ShapeletProfileSurrogate(
        input_dim=4, output_dim=H * W, profile_shape=(1, H, W),
        beta_par=3.0, beta_perp=3.0, quad=3,
    )
    prof = s._profile(torch.zeros(1, 1, s.d), torch.zeros(1), s._frame(None, 1, "cpu", torch.float32))
    grid = prof.reshape(H, W)

    assert grid.argmax().item() == (H // 2) * W + W // 2
    # n_eff of a 2-D Gaussian of width sigma on a fine grid is 4*pi*sigma^2
    sigma = 3.0 / math.sqrt(2.0)
    assert n_eff(prof.reshape(1, -1)).item() == pytest.approx(
        4 * math.pi * sigma**2, rel=0.15
    )


def test_shapelet_never_reaches_delta_or_uniform():
    """Across a wide sweep of coefficients the family stays bounded away from both
    attractors. This is the property the whole surrogate exists to provide."""
    torch.manual_seed(0)
    H = W = 21
    P = H * W
    s = ShapeletProfileSurrogate(
        input_dim=4, output_dim=P, profile_shape=(1, H, W),
        beta_par=2.0, beta_perp=1.2, quad=2,
    )
    frame = s._frame(None, 512, "cpu", torch.float32)
    for scale in (0.1, 0.5, 2.0, 10.0):
        eps = torch.randn(1, 512, s.d) * scale
        ne = n_eff(s._profile(eps, torch.zeros(512), frame).reshape(512, P))
        assert ne.min() > 2.0, f"delta at eps scale {scale}: n_eff {ne.min():.2f}"
        assert ne.max() < 0.5 * P, f"flat at eps scale {scale}: n_eff {ne.max():.1f}"


def test_shapelet_order_prior_decays():
    """tau_n = tau0 (1+n)^-decay. The decay is load-bearing: a flat prior across
    orders lets degeneracy back in as the order rises."""
    s = ShapeletProfileSurrogate(
        input_dim=4, output_dim=49, profile_shape=(1, 7, 7),
        n_par=3, n_perp=3, tau0=0.5, tau_decay=1.5,
    )
    orders = [
        i + j
        for i in range(4)
        for j in range(4)
    ][1:]
    expected = torch.tensor([0.5 * (1 + n) ** -1.5 for n in orders])
    assert torch.allclose(s.tau, expected, atol=1e-6)
    assert s.tau.max() <= 0.5


def test_shapelet_reports_its_own_kl():
    """The prior differs per order, so a scalar prior_scale cannot express it and
    the surrogate must hand the loss a finished KL."""
    s = ShapeletProfileSurrogate(
        input_dim=8, output_dim=49, profile_shape=(1, 7, 7), quad=1
    )
    out = s(torch.randn(6, 8), mc_samples=2)
    assert out.kl is not None and out.kl.shape == (6,)
    assert (out.kl >= 0).all()

    from integrator.model.loss.kl_helpers import compute_profile_kl

    kl = compute_profile_kl(out, prior_scale=3.0, profile_kl_weight=2.0, device="cpu")
    assert torch.allclose(kl, out.kl * 2.0)


def test_shapelet_3d_uses_the_frame_axis():
    s = ShapeletProfileSurrogate(
        input_dim=8, output_dim=3 * 11 * 11, profile_shape=(3, 11, 11),
        n_par=3, n_perp=2, n_z=1, quad=1,
    )
    assert s.n_z == 1
    assert s.d == (3 + 1) * (2 + 1) * (1 + 1) - 1
    out = s(torch.randn(4, 8), mc_samples=2)
    assert out.zp.shape == (2, 4, 363)
    assert torch.allclose(out.zp.sum(-1), torch.ones(2, 4), atol=1e-5)


def test_shapelet_2d_shape_forces_nz_zero():
    s = ShapeletProfileSurrogate(
        input_dim=8, output_dim=49, profile_shape=(1, 7, 7), n_z=3
    )
    assert s.n_z == 0


def test_shapelet_gradients_flow_to_the_encoder_heads():
    s = ShapeletProfileSurrogate(
        input_dim=8, output_dim=49, profile_shape=(1, 7, 7), quad=1
    )
    out = s(torch.randn(4, 8), mc_samples=2)
    (out.zp.sum() + out.kl.sum()).backward()
    assert s.loc_head.weight.grad is not None
    assert s.loc_head.weight.grad.abs().sum() > 0
    assert s.scale_head.weight.grad is not None


# ------------------------------------------------------- full factory path


def _cfg(qp_name: str, qp_args: dict, dim: str = "2d") -> dict:
    geom = {"data_dim": "2d", "d": 1, "h": 21, "w": 21} if dim == "2d" else {
        "data_dim": "3d", "d": 3, "h": 11, "w": 11
    }
    return {
        "integrator": {
            "name": "hierarchical",
            "args": {**geom, "encoder_out": 64, "mc_samples": 2},
        },
        "loss": {
            "name": "polychromatic_wilson",
            "args": {
                "degree": 4, "lambda_min": 0.95, "lambda_max": 1.2,
                "init_B": 30.0, "n_bins": 1,
            },
        },
        "surrogates": {"qp": {"name": qp_name, "args": dict(qp_args)}},
        "data_loader": {"name": "polychromatic_data", "args": {"data_dir": None}},
        "optimizer": {"lr": 1e-3},
    }


def _batch(npix: int, b: int = 6):
    torch.manual_seed(1)
    counts = torch.poisson(torch.full((b, npix), 12.0))
    meta = {
        "d": torch.full((b,), 2.5),
        "group_label": torch.zeros(b),
        "wavelength": torch.full((b,), 1.05),
    }
    return counts, counts.clone(), torch.ones(b, npix), meta


@pytest.mark.parametrize(
    "qp_name,qp_args,dim,npix",
    [
        ("fixed_basis_profile", {"max_order": 2}, "2d", 441),
        ("fixed_basis_profile", {"max_order": 2}, "3d", 363),
        ("shapelet_profile", {"n_par": 4, "n_perp": 2, "quad": 2}, "2d", 441),
        ("shapelet_profile", {"n_par": 3, "n_perp": 2, "n_z": 1, "quad": 1}, "3d", 363),
    ],
)
def test_full_factory_path_and_backward(qp_name, qp_args, dim, npix):
    """Through resolve_config/construct_integrator, not the class directly: the
    resolver has to supply output_dim and profile_shape itself."""
    from integrator.utils.factory_utils import construct_integrator, resolve_config

    model = construct_integrator(resolve_config(_cfg(qp_name, qp_args, dim)))
    assert type(model.surrogates["qp"]).__name__.startswith(
        "FixedBasis" if qp_name == "fixed_basis_profile" else "Shapelet"
    )

    out = model(*_batch(npix))
    rates = out["forward_out"]["rates"]
    assert torch.isfinite(rates).all()
    assert (rates > 0).all()

    profile = out["forward_out"]["qp_mean"]
    assert torch.allclose(profile.sum(-1), torch.ones(6), atol=1e-4)

    step = model._step(_batch(npix), step="train")
    assert torch.isfinite(step["loss"])
    step["loss"].backward()
