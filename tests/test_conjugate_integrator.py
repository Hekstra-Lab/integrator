"""Regression tests for ConjugateIntegrator.

Covers the per-observation Poisson-Gamma conjugate update:
  - construction / forward / backward through the factory
  - EM convergence of the responsibility fixed point
  - exactness of the implicit-function-theorem gradient (vs finite diff)
  - the loss-consistency guard (pi_weight, wilson_alpha, prior shape)
"""

import pytest
import torch

from integrator.model.scaling.conjugate_integrator import ConjugateIntegrator
from integrator.utils.factory_utils import construct_integrator

P = 441  # 21 x 21, 2d


def _cfg():
    return {
        "integrator": {
            "name": "conjugate",
            "args": {
                "data_dim": "2d", "d": 1, "h": 21, "w": 21,
                "mc_samples": 3, "lr": 1e-3,
                "wilson_alpha": 1.0, "n_em_iters": 40, "em_tol": 1e-3,
                "scale_frame_max": 1000.0,
            },
        },
        "encoders": [
            {"name": "profile_encoder", "args": {"data_dim": "2d"}},
            {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
            {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
        ],
        "surrogates": {
            "qp": {"name": "learned_basis_profile",
                   "args": {"input_dim": 64, "output_dim": 441}},
            "qbg": {"name": "gammaA", "args": {"in_features": 64}},
        },
        "loss": {"name": "monochromatic_wilson",
                 "args": {"mc_samples": 3, "eps": 1e-6, "pi_weight": 1.0}},
        "data_loader": {"name": "rotation_data", "args": {"data_dir": "/tmp"}},
    }


def _mock_metadata(B):
    return {
        "asu_id": torch.randint(0, 50, (B,)),
        "d": torch.rand(B) + 0.5,
        "group_label": torch.zeros(B),
        "profile_group_label": torch.zeros(B),
        "is_coset": torch.zeros(B, dtype=torch.bool),
        "xyzcal.px.0": torch.rand(B) * 2000,
        "xyzcal.px.1": torch.rand(B) * 2000,
        "xyzcal.px.2": torch.rand(B) * 1000,
        "lp": torch.rand(B) * 0.5 + 0.5,
    }


@pytest.fixture
def integ():
    torch.manual_seed(0)
    return construct_integrator(_cfg())


def test_construct(integ):
    assert isinstance(integ, ConjugateIntegrator)
    assert integ.n_em_iters == 40
    assert integ.em_tol == pytest.approx(1e-3)
    assert integ.alpha_W == 1.0


def test_forward_and_backward(integ):
    B = 16
    meta = _mock_metadata(B)
    out = integ(
        torch.randint(0, 30, (B, P)).float(),
        torch.randn(B, P),
        torch.ones(B, P),
        meta,
    )
    fwd = out["forward_out"]
    assert fwd["rates"].shape == (B, 3, P)
    ld = integ.loss(
        rate=fwd["rates"], counts=fwd["counts"], qp=out["qp"], qi=out["qi"],
        qbg=out["qbg"], mask=fwd["mask"],
        group_labels=meta["group_label"].long(), metadata=meta,
    )
    ld["loss"].backward()
    grads = [p.grad for p in integ.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any((g != 0).any() for g in grads)


def _synthetic(B=64, seed=1):
    torch.manual_seed(seed)
    profile = torch.softmax(torch.randn(B, P), dim=-1)
    bg = torch.rand(B) * 0.4 + 0.05
    tau = torch.rand(B) * 0.5 + 0.2
    mask = torch.ones(B, P)
    I_true = torch.rand(B) * 30
    scale = torch.rand(B) * 0.5 + 0.5
    counts = torch.poisson(scale[:, None] * I_true[:, None] * profile + bg[:, None])
    return counts, profile, bg, tau, mask, scale


def test_em_converges_to_fixed_point(integ):
    counts, profile, bg, tau, mask, scale = _synthetic()
    with torch.no_grad():
        a, b, _ = integ._conjugate_em(counts, profile, bg, scale, tau, mask)
        log_beta = b.clamp(min=1e-12).log()
        # CAVI fixed point is in the geometric-mean intensity exp(E[log I]).
        I_tilde = torch.exp(torch.digamma(a.clamp(min=1e-6)) - log_beta)
        sr = scale[:, None] * I_tilde[:, None] * profile
        pi = sr / (sr + bg[:, None]).clamp(min=1e-12)
        a2 = integ.alpha_W + (pi * counts * mask).sum(-1)
        I_tilde2 = torch.exp(torch.digamma(a2.clamp(min=1e-6)) - log_beta)
        resid = ((I_tilde2 - I_tilde).abs() / I_tilde.clamp(min=1e-12)).max()
    assert resid < integ.em_tol, f"not converged: {resid}"


def test_implicit_gradient_matches_finite_difference(integ):
    """The EM gradient must be the exact implicit-function derivative of the
    converged fixed point, not the (biased) naive one-step gradient."""
    counts, profile, bg, tau, mask, scale = _synthetic(B=128)

    scale_leaf = scale.clone().requires_grad_(True)
    a, b, _ = integ._conjugate_em(counts, profile, bg, scale_leaf, tau, mask)
    g_auto = torch.autograd.grad((a / b).sum(), scale_leaf)[0]

    def converged_I(sv):
        with torch.no_grad():
            aa, bb, _ = integ._conjugate_em(counts, profile, bg, sv, tau, mask)
            return aa / bb

    eps = 1e-3
    g_fd = (converged_I(scale + eps) - converged_I(scale - eps)) / (2 * eps)

    rel = (g_auto - g_fd).abs() / g_fd.abs().clamp(min=1e-6)
    assert rel.median() < 0.02
    assert rel.mean() < 0.02

    # Sanity: the CAVI fixed-point Jacobian K is genuinely non-negligible here,
    # so a naive one-step gradient (which omits the 1/(1-K) factor) would be wrong.
    with torch.no_grad():
        a, b, _ = integ._conjugate_em(counts, profile, bg, scale, tau, mask)
        log_beta = b.clamp(min=1e-12).log()
        I_tilde = torch.exp(torch.digamma(a.clamp(min=1e-6)) - log_beta)
        sr = scale[:, None] * I_tilde[:, None] * profile
        pic = sr / (sr + bg[:, None]).clamp(min=1e-12)
        K = torch.special.polygamma(1, a.clamp(min=1e-6)) * (counts * pic * (1 - pic) * mask).sum(-1)
    assert K.mean() > 0.2


@pytest.mark.parametrize(
    "where,key,val",
    [
        ("loss", "pi_weight", 0.0),
        ("integrator", "wilson_alpha", 0.5),
        ("loss", "learn_concentration", True),
    ],
)
def test_consistency_guard_raises(where, key, val):
    cfg = _cfg()
    cfg[where]["args"][key] = val
    with pytest.raises(ValueError):
        construct_integrator(cfg)


# ---- calibrated exact-posterior export (Fix A + Fix B) ----------------------

def _ref_quad_moments(counts, e, bg, tau, grid, aW=1.0):
    """Independent dense reference for the collapsed-posterior moments."""
    I = grid.unsqueeze(-1)  # (B, G, 1)
    rate = e.unsqueeze(1) * I + bg[:, None, None]  # (B, G, P)
    dterm = (counts.unsqueeze(1) * torch.log(rate.clamp(min=1e-30))).sum(-1)
    lin = tau + e.sum(-1)
    logp = (aW - 1) * torch.log(grid.clamp(min=1e-30)) - lin[:, None] * grid + dterm
    dw = torch.diff(grid, dim=1, prepend=grid[:, :1]).clamp(min=1e-30).log()
    w = torch.softmax(logp + dw, 1)
    m1 = (w * grid).sum(-1)
    m2 = (w * grid**2).sum(-1)
    return m1, (m2 - m1**2).clamp(min=0)


def test_quad_moments_matches_reference(integ):
    counts, profile, bg, tau, mask, scale = _synthetic(B=32)
    e = scale[:, None] * profile
    lo = torch.zeros(counts.shape[0]) + 1e-8
    hi = (e.sum(-1) * 0 + 60.0)  # wide fixed grid
    grid = lo[:, None] + torch.linspace(0, 1, 4000)[None, :] * (hi - lo)[:, None]
    m_chunked, v_chunked = integ._quad_moments(counts, mask, e, bg, tau, grid, chunk=37)
    m_ref, v_ref = _ref_quad_moments(counts, e, bg, tau, grid, aW=integ.alpha_W)
    assert torch.allclose(m_chunked, m_ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(v_chunked, v_ref, rtol=1e-5, atol=1e-6)


def test_exact_posterior_widens_variance(integ):
    """On a high-background case the exact quadrature variance must exceed the
    (too-narrow) mean-field Gamma variance."""
    torch.manual_seed(3)
    B, P = 48, 441
    profile = torch.softmax(torch.randn(B, P), dim=-1)
    bg = torch.full((B,), 2.0)          # high background -> large allocation gap
    tau = torch.rand(B) * 0.3 + 0.2
    mask = torch.ones(B, P)
    scale = torch.ones(B)
    I_true = torch.rand(B) * 8 + 1
    counts = torch.poisson(scale[:, None] * I_true[:, None] * profile + bg[:, None])

    alpha, beta, _ = integ._conjugate_em(counts, profile, bg, scale, tau, mask)
    mf_var = alpha / beta.pow(2)

    e = scale[:, None] * profile
    mean_mf = alpha / beta
    std_eff = 3.0 * alpha.sqrt() / beta
    lo = (mean_mf - 8 * std_eff).clamp(min=1e-8)
    hi = torch.maximum(mean_mf + 12 * std_eff, lo + 1e-3)
    grid = lo[:, None] + torch.linspace(0, 1, 2048)[None, :] * (hi - lo)[:, None]
    _, exact_var = integ._quad_moments(counts, mask, e, bg, tau, grid, chunk=128)

    # exact variance is strictly wider, and substantially so at high bg
    assert (exact_var > mf_var).all()
    assert exact_var.median() > 1.3 * mf_var.median()


def test_exact_intensity_posterior_end_to_end(integ):
    B, P = 16, 441
    meta = _mock_metadata(B)
    counts = torch.randint(0, 30, (B, P)).float()
    shoebox = torch.randn(B, P)
    mask = torch.ones(B, P)
    for n_nuis in (1, 6):
        out = integ.exact_intensity_posterior(
            counts, shoebox, mask, meta, n_nuisance=n_nuis, n_grid=512, grid_chunk=64
        )
        for k in ("mean", "var", "std", "alpha", "beta"):
            assert out[k].shape == (B,)
            assert torch.isfinite(out[k]).all()
            assert (out[k] > 0).all()
        # moment-matched Gamma is consistent with the reported moments
        assert torch.allclose(out["mean"], out["alpha"] / out["beta"], rtol=1e-4)
        assert torch.allclose(
            out["var"], out["alpha"] / out["beta"].pow(2), rtol=1e-4
        )


def test_predict_step_surfaces_exact_keys(integ):
    integ.predict_keys = list(integ.predict_keys) + [
        "qi_exact_mean", "qi_exact_var", "qi_exact_std",
    ]
    integ.exact_posterior_n_nuisance = 4
    integ.exact_posterior_n_grid = 256
    B, P = 12, 441
    meta = _mock_metadata(B)
    batch = (
        torch.randint(0, 30, (B, P)).float(),
        torch.randn(B, P),
        torch.ones(B, P),
        meta,
    )
    out = integ.predict_step(batch, 0)
    for k in ("qi_exact_mean", "qi_exact_var", "qi_exact_std"):
        assert k in out
        assert out[k].shape == (B,)
        assert torch.isfinite(out[k]).all() and (out[k] > 0).all()
