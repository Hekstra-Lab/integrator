"""Tests for the profile surrogates and KL helpers.

Covers:
  1. ProfileSurrogateOutput samples sum to 1 (simplex constraint)
  2. KL helpers produce correct values
  3. Gradients flow through samples to encoder parameters
  4. W and b do NOT receive gradients (for buffer-based surrogates)
  5. Loss.forward handles ProfileSurrogateOutput correctly
"""

from pathlib import Path

import pytest
import torch

from integrator.model.distributions.profile_surrogates import (
    FixedBasisProfileSurrogate,
    ProfileSurrogateOutput,
)
from integrator.model.loss.kl_helpers import (
    compute_profile_kl_global,
    compute_profile_kl_per_bin,
)

K = 441  # 21x21 pixels
D = 14  # latent dimension
B = 8  # batch size
S = 5  # MC samples
INPUT_DIM = 64


def make_basis_file(
    tmp_path: Path, d: int = D, k: int = K, sigma_prior: float = 3.0
) -> str:
    """Write a random profile_basis.pt and return its path."""
    W = torch.randn(k, d)
    b = torch.zeros(k)
    basis = {
        "W": W,
        "b": b,
        "d": d,
        "sigma_prior": sigma_prior,
        "orders": [
            (i, j)
            for i in range(5)
            for j in range(5 - i)
            if not (i == 0 and j == 0)
        ][:d],
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
    return FixedBasisProfileSurrogate(
        input_dim=INPUT_DIM, basis_path=tmp_basis
    )


@pytest.fixture
def profile_output(surrogate):
    x = torch.randn(B, INPUT_DIM)
    return surrogate(x, mc_samples=S)


class TestProfileSurrogateOutput:
    def test_zp_sums_to_one(self, profile_output):
        """Sampled profiles must lie on the probability simplex."""
        assert profile_output.zp.shape == (S, B, K)
        sums = profile_output.zp.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_single_sample(self, surrogate):
        """mc_samples=1 returns (1, B, K) on the simplex."""
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x, mc_samples=1)
        assert out.zp.shape == (1, B, K)
        sums = out.zp.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_mean_profile_sums_to_one(self, profile_output):
        """The mean profile (at posterior mean h) should sum to 1."""
        mean_prf = profile_output.mean_profile
        assert mean_prf.shape == (B, K)
        sums = mean_prf.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_mu_h_shape(self, profile_output):
        assert profile_output.mu_h.shape == (B, D)

    def test_std_h_shape(self, profile_output):
        assert profile_output.std_h.shape == (B, D)

    def test_no_kl_field(self, profile_output):
        """ProfileSurrogateOutput should not have a kl field."""
        assert not hasattr(profile_output, "kl")

    def test_no_shim_methods(self, profile_output):
        """No backward-compat shims should exist."""
        assert not hasattr(profile_output, "rsample")
        assert not hasattr(profile_output, "kl_divergence")
        assert not hasattr(profile_output, "concentration")


class TestKLHelpers:
    def test_kl_global_non_negative(self, profile_output):
        """KL divergence must be >= 0."""
        kl = compute_profile_kl_global(
            profile_output.mu_h, profile_output.std_h, 3.0
        )
        assert kl.shape == (B,)
        assert (kl >= 0).all(), f"negative KL found: {kl}"

    def test_kl_global_zero_at_prior(self):
        """KL should be ~0 when q matches the prior."""
        sigma_p = 3.0
        mu_h = torch.zeros(B, D)
        std_h = torch.full((B, D), sigma_p)
        kl = compute_profile_kl_global(mu_h, std_h, sigma_p)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_per_bin_non_negative(self, profile_output):
        """Per-bin KL must be >= 0."""
        n_bins = 4
        mu_prior = torch.randn(n_bins, D)
        std_prior = torch.rand(n_bins, D) + 0.1
        group_labels = torch.randint(0, n_bins, (B,))
        kl = compute_profile_kl_per_bin(
            profile_output.mu_h,
            profile_output.std_h,
            mu_prior,
            std_prior,
            group_labels,
        )
        assert kl.shape == (B,)
        assert (kl >= 0).all()

    def test_kl_per_bin_zero_at_prior(self):
        """Per-bin KL should be ~0 when q matches the per-bin prior."""
        n_bins = 3
        mu_prior = torch.randn(n_bins, D)
        std_prior = torch.rand(n_bins, D) + 0.5
        group_labels = torch.randint(0, n_bins, (B,))
        mu_h = mu_prior[group_labels]
        std_h = std_prior[group_labels]
        kl = compute_profile_kl_per_bin(
            mu_h, std_h, mu_prior, std_prior, group_labels
        )
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_gradient_flows(self, profile_output):
        """KL must be differentiable w.r.t. mu_h and std_h."""
        mu_h = profile_output.mu_h.detach().requires_grad_(True)
        std_h = profile_output.std_h.detach().requires_grad_(True)
        kl = compute_profile_kl_global(mu_h, std_h, 3.0).mean()
        kl.backward()
        assert mu_h.grad is not None
        assert std_h.grad is not None


class TestFixedBasisProfileSurrogate:
    def test_forward_returns_profile_surrogate_output(self, surrogate):
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x, mc_samples=S)
        assert isinstance(out, ProfileSurrogateOutput)

    def test_buffers_no_grad(self, surrogate):
        """W and b must be registered buffers -- they should NOT receive gradients."""
        x = torch.randn(B, INPUT_DIM, requires_grad=True)
        out = surrogate(x, mc_samples=S)
        loss = out.zp.sum()
        loss.backward()
        assert surrogate.W.grad is None
        assert surrogate.b.grad is None

    def test_encoder_params_get_grad(self, surrogate):
        """Gradients must flow through samples to mu_head and std_head."""
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x, mc_samples=S)
        loss = (out.zp.sum(-1) - 1.0).pow(2).sum()
        loss.backward()
        assert surrogate.mu_head.weight.grad is not None
        assert surrogate.std_head.weight.grad is not None
        assert surrogate.mu_head.weight.grad.abs().sum() > 0

    def test_std_positive(self, surrogate):
        """std_h must be positive (softplus output)."""
        with torch.no_grad():
            surrogate.std_head.bias.fill_(100.0)
        x = torch.randn(B, INPUT_DIM)
        out = surrogate(x, mc_samples=1)
        assert (out.std_h > 0).all()

        with torch.no_grad():
            surrogate.std_head.bias.fill_(-100.0)
        out = surrogate(x, mc_samples=1)
        assert (out.std_h > 0).all()

        with torch.no_grad():
            surrogate.std_head.bias.fill_(-0.81)

    def test_device_buffers_follow_model(self, surrogate):
        assert isinstance(surrogate.W, torch.Tensor)
        assert isinstance(surrogate.b, torch.Tensor)
        assert surrogate.W.shape == (K, D)
        assert surrogate.b.shape == (K,)

    def test_d_attribute(self, surrogate):
        assert surrogate.d == D


class TestLossWithProfileSurrogateOutput:
    def test_loss_forward_no_pprf_cfg(self, surrogate):
        """Loss.forward with pprf_cfg=None and ProfileSurrogateOutput computes valid loss."""
        from integrator.model.loss.loss import Loss

        loss_fn = Loss(pprf_cfg=None, pi_cfg=None, pbg_cfg=None, mc_samples=4)

        B_local = 4
        K_local = 441
        S_local = 4

        x = torch.randn(B_local, INPUT_DIM)
        qp = surrogate(x, mc_samples=S_local)

        rate = torch.rand(B_local, S_local, K_local) + 0.1
        counts = torch.poisson(rate[:, 0, :])
        mask = torch.ones(B_local, K_local, 1)

        out = loss_fn(rate=rate, counts=counts, qp=qp, mask=mask)

        assert "loss" in out
        assert torch.isfinite(out["loss"])
        assert out["kl_prf_mean"] > 0, (
            "KL should be positive with non-prior params"
        )

    def test_loss_backward(self, surrogate):
        """Gradient must flow from loss to surrogate parameters."""
        from integrator.model.loss.loss import Loss

        loss_fn = Loss(pprf_cfg=None, pi_cfg=None, pbg_cfg=None, mc_samples=4)

        B_local = 4
        K_local = 441
        S_local = 4

        x = torch.randn(B_local, INPUT_DIM)
        qp = surrogate(x, mc_samples=S_local)

        rate = qp.zp.permute(1, 0, 2) + 0.1
        counts = torch.ones(B_local, K_local)
        mask = torch.ones(B_local, K_local, 1)

        out = loss_fn(rate=rate, counts=counts, qp=qp, mask=mask)
        out["loss"].backward()

        assert surrogate.mu_head.weight.grad is not None
        assert surrogate.std_head.weight.grad is not None
