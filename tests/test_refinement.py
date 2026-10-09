import pytest
import torch

from integrator.model.scaling.refinement_integrator import DeterministicIntensity


class TestDeterministicIntensity:
    def test_mean_returns_F_sq(self):
        F_sq = torch.tensor([1.0, 2.0, 3.0])
        qi = DeterministicIntensity(F_sq)
        assert torch.allclose(qi.mean, F_sq)

    def test_variance_is_zero(self):
        F_sq = torch.tensor([1.0, 2.0, 3.0])
        qi = DeterministicIntensity(F_sq)
        assert (qi.variance == 0).all()

    def test_arg_constraints_empty(self):
        qi = DeterministicIntensity(torch.tensor([1.0]))
        assert qi.arg_constraints == {}


class TestRefinementLoss:
    def test_no_intensity_kl(self):
        from integrator.model.loss.refinement_loss import RefinementLoss

        loss_fn = RefinementLoss(mc_samples=2, eps=1e-6)

        B, S, K = 4, 2, 441
        rate = torch.rand(B, S, K) + 0.1
        counts = torch.randint(0, 10, (B, K)).float()
        mask = torch.ones(B, K)
        qi = DeterministicIntensity(torch.rand(B))

        from torch.distributions import Gamma

        qbg = Gamma(torch.ones(B), torch.ones(B))

        from integrator.model.distributions.profile_surrogates import (
            ProfileSurrogateOutput,
        )

        qp = ProfileSurrogateOutput(
            zp=torch.rand(S, B, K),
            mean_profile=torch.rand(B, K),
            loc=torch.randn(B, 8),
            scale=torch.rand(B, 8) + 0.1,
        )

        result = loss_fn(
            rate=rate,
            counts=counts,
            qp=qp,
            qi=qi,
            qbg=qbg,
            mask=mask,
            group_labels=torch.zeros(B, dtype=torch.long),
            metadata={
                "d": torch.rand(B) + 0.5,
                "xyzcal.px.0": torch.rand(B) * 2000,
                "xyzcal.px.1": torch.rand(B) * 2000,
            },
        )

        assert result["kl_i_mean"].item() == 0.0
        assert result["loss"].isfinite()
        assert result["neg_ll_mean"].isfinite()
        assert result["kl_prf_mean"].isfinite()
        assert result["kl_bg_mean"].isfinite()

    def test_loss_backward(self):
        from integrator.model.loss.refinement_loss import RefinementLoss
        from torch.distributions import Gamma

        loss_fn = RefinementLoss(mc_samples=2, eps=1e-6)
        B, S, K = 4, 2, 9

        F_sq = torch.rand(B, requires_grad=True)
        profile = torch.softmax(torch.randn(B, K), dim=-1)
        bg = torch.rand(B, 1) + 0.1
        rate = (F_sq.unsqueeze(1).unsqueeze(-1) * profile.unsqueeze(1) + bg.unsqueeze(1))
        counts = torch.randint(0, 10, (B, K)).float()
        mask = torch.ones(B, K)

        from integrator.model.distributions.profile_surrogates import (
            ProfileSurrogateOutput,
        )

        qp = ProfileSurrogateOutput(
            zp=profile.unsqueeze(0),
            mean_profile=profile,
            loc=torch.randn(B, 4),
            scale=torch.rand(B, 4) + 0.1,
        )
        qbg = Gamma(torch.ones(B), torch.ones(B))

        result = loss_fn(
            rate=rate,
            counts=counts,
            qp=qp,
            qi=DeterministicIntensity(F_sq),
            qbg=qbg,
            mask=mask,
            group_labels=torch.zeros(B, dtype=torch.long),
            metadata={
                "d": torch.rand(B) + 0.5,
                "xyzcal.px.0": torch.rand(B) * 2000,
                "xyzcal.px.1": torch.rand(B) * 2000,
            },
        )
        result["loss"].backward()
        assert F_sq.grad is not None
        assert (F_sq.grad != 0).any()
