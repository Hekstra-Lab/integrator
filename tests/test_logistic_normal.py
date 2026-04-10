"""Tests for the logistic-normal profile surrogate.

Covers:
  1. ProfilePosterior output sums to 1 (simplex constraint)
  2. KL is non-negative
  3. KL == 0 when q matches p (mu=0, std=sigma_prior)
  4. Gradients flow through rsample to encoder parameters
  5. W and b do NOT receive gradients
  6. Loss.forward handles ProfilePosterior correctly
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from integrator.model.distributions.logistic_normal import (
    LogisticNormalSurrogate,
    ProfilePosterior,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

K = 441   # 21×21 pixels
D = 14    # latent dimension
B = 8     # batch size
INPUT_DIM = 64


def make_basis_file(tmp_path: Path, d: int = D, k: int = K, sigma_prior: float = 3.0) -> str:
    """Write a random profile_basis.pt and return its path."""
    W = torch.randn(k, d)
    b = torch.zeros(k)
    basis = {
        "W": W,
        "b": b,
        "d": d,
        "sigma_prior": sigma_prior,
        "orders": [(i, j) for i in range(5) for j in range(5 - i) if not (i == 0 and j == 0)][:d],
        "max_order": 4,
        "sigma_ref": 3.0,
        "basis_type": "hermite",
    }
    path = str(tmp_path / "profile_basis.pt")
    torch.save(basis, path)
    return path


@pytest.fixture
def tmp_basis(tmp_path):
    return make_basis_file(tmp_path)


@pytest.fixture
def surrogate(tmp_basis):
    return LogisticNormalSurrogate(input_dim=INPUT_DIM, basis_path=tmp_basis)


@pytest.fixture
def profile_posterior(surrogate):
    x = torch.randn(B, INPUT_DIM)
    return surrogate(x)


# ---------------------------------------------------------------------------
# ProfilePosterior tests
# ---------------------------------------------------------------------------


class TestProfilePosterior:
    def test_rsample_sums_to_one(self, profile_posterior):
        """Sampled profiles must lie on the probability simplex."""
        S = 5
        prf = profile_posterior.rsample([S])  # (S, B, K)
        assert prf.shape == (S, B, K)
        sums = prf.sum(dim=-1)  # (S, B)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"profiles do not sum to 1: min={sums.min():.6f}, max={sums.max():.6f}"

    def test_rsample_no_sample_shape(self, profile_posterior):
        """rsample() with empty shape returns (B, K) on the simplex."""
        prf = profile_posterior.rsample()  # (B, K)
        assert prf.shape == (B, K)
        sums = prf.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_kl_non_negative(self, profile_posterior):
        """KL divergence must be >= 0 for all batch elements."""
        kl = profile_posterior.kl_divergence()  # (B,)
        assert kl.shape == (B,)
        assert (kl >= 0).all(), f"negative KL found: {kl}"

    def test_kl_zero_at_prior(self, tmp_basis):
        """KL should be (approximately) 0 when q matches the prior."""
        sigma_p = 3.0  # matches basis file
        # mu=0, std=sigma_p → q = p
        mu_h = torch.zeros(B, D)
        std_h = torch.full((B, D), sigma_p)

        # We need W and b from the file
        basis = torch.load(tmp_basis, weights_only=False)
        pp = ProfilePosterior(
            mu_h=mu_h,
            std_h=std_h,
            W=basis["W"],
            b=basis["b"],
            sigma_prior=sigma_p,
        )
        kl = pp.kl_divergence()
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5), \
            f"KL should be 0 at prior, got: {kl}"

    def test_mean_sums_to_one(self, profile_posterior):
        """The mean profile (at posterior mean h) should sum to 1."""
        mean_prf = profile_posterior.mean  # (B, K)
        assert mean_prf.shape == (B, K)
        sums = mean_prf.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_concentration_is_none(self, profile_posterior):
        """concentration attribute exists and is None (Dirichlet compat shim)."""
        assert profile_posterior.concentration is None

    def test_mean_h_shape(self, profile_posterior):
        assert profile_posterior.mean_h.shape == (B, D)


# ---------------------------------------------------------------------------
# LogisticNormalSurrogate tests
# ---------------------------------------------------------------------------


class TestLogisticNormalSurrogate:
    def test_forward_returns_profile_posterior(self, surrogate):
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x)
        assert isinstance(out, ProfilePosterior)

    def test_buffers_no_grad(self, surrogate):
        """W and b must be registered buffers — they should NOT receive gradients."""
        x = torch.randn(B, INPUT_DIM, requires_grad=True)
        out = surrogate(x)
        prf = out.rsample([4])  # (4, B, K)
        loss = prf.sum()
        loss.backward()

        # Buffers have no grad (they are not nn.Parameters)
        assert surrogate.W.grad is None, "W should not have gradient"
        assert surrogate.b.grad is None, "b should not have gradient"

    def test_encoder_params_get_grad(self, surrogate):
        """Gradients must flow through rsample to mu_head and std_head."""
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x)
        prf = out.rsample([4])  # (4, B, K)
        # Use profile in a downstream scalar loss
        loss = (prf.sum(-1) - 1.0).pow(2).sum()
        loss.backward()

        mu_grad = surrogate.mu_head.weight.grad
        std_grad = surrogate.std_head.weight.grad
        assert mu_grad is not None, "mu_head.weight has no gradient"
        assert std_grad is not None, "std_head.weight has no gradient"
        assert mu_grad.abs().sum() > 0, "mu_head.weight gradient is all zeros"

    def test_std_positive(self, surrogate):
        """std_h must be positive (softplus output)."""
        # Force extreme output by setting large bias
        with torch.no_grad():
            surrogate.std_head.bias.fill_(100.0)
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x)
        assert (out.std_h > 0).all()

        # Also test with very negative bias
        with torch.no_grad():
            surrogate.std_head.bias.fill_(-100.0)
        out = surrogate(x)
        assert (out.std_h > 0).all()

        # Restore
        with torch.no_grad():
            surrogate.std_head.bias.fill_(-0.81)

    def test_device_buffers_follow_model(self, surrogate):
        """Buffers should be accessible as tensors on the same device as the model."""
        assert isinstance(surrogate.W, torch.Tensor)
        assert isinstance(surrogate.b, torch.Tensor)
        assert surrogate.W.shape == (K, D)
        assert surrogate.b.shape == (K,)

    def test_d_attribute(self, surrogate):
        assert surrogate.d == D

    def test_sigma_prior_attribute(self, surrogate):
        assert surrogate.sigma_prior == 3.0

    def test_kl_gradient_flows(self, surrogate):
        """KL must be differentiable w.r.t. mu_head and std_head parameters."""
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x)
        kl = out.kl_divergence().mean()
        kl.backward()

        assert surrogate.mu_head.weight.grad is not None
        assert surrogate.std_head.weight.grad is not None


# ---------------------------------------------------------------------------
# Integration with Loss
# ---------------------------------------------------------------------------


class TestLossWithProfilePosterior:
    def test_loss_forward_no_pprf_cfg(self, surrogate):
        """Loss.forward with pprf_cfg=None and a ProfilePosterior computes valid loss."""
        from integrator.model.loss.loss import Loss

        loss_fn = Loss(pprf_cfg=None, pi_cfg=None, pbg_cfg=None, mc_samples=4)

        B_local = 4
        K_local = 441
        S = 4

        x = torch.randn(B_local, INPUT_DIM)
        qp = surrogate(x)

        # Fake rate, counts, mask consistent with K_local pixels
        rate = torch.rand(B_local, S, K_local) + 0.1
        counts = torch.poisson(rate[:, 0, :])  # (B, K)
        mask = torch.ones(B_local, K_local, 1)

        out = loss_fn(rate=rate, counts=counts, qp=qp, mask=mask)

        assert "loss" in out
        assert torch.isfinite(out["loss"])
        assert out["kl_prf_mean"] > 0, "KL should be positive with non-prior params"

    def test_loss_backward(self, surrogate):
        """Gradient must flow from loss to surrogate parameters."""
        from integrator.model.loss.loss import Loss

        loss_fn = Loss(pprf_cfg=None, pi_cfg=None, pbg_cfg=None, mc_samples=4)

        B_local = 4
        K_local = 441
        S = 4

        x = torch.randn(B_local, INPUT_DIM)
        qp = surrogate(x)

        # Build a fake rate that depends on surrogate (gradient path)
        prf_samples = qp.rsample([S])               # (S, B, K)
        rate = prf_samples.permute(1, 0, 2) + 0.1   # (B, S, K)
        counts = torch.ones(B_local, K_local)
        mask = torch.ones(B_local, K_local, 1)

        out = loss_fn(rate=rate, counts=counts, qp=qp, mask=mask)
        out["loss"].backward()

        assert surrogate.mu_head.weight.grad is not None
        assert surrogate.std_head.weight.grad is not None
