import pytest
import torch
from torch.distributions import Gamma

from integrator.model.scaling.chebyshev_scale import ChebyshevScale
from integrator.model.scaling.hkl_table import HKLLookupTable
from integrator.utils.factory_utils import construct_integrator


def _mock_metadata(B, n_hkl=50):
    return {
        "asu_id": torch.randint(0, n_hkl, (B,)),
        "H": torch.randint(-5, 6, (B,)).float(),
        "K": torch.randint(-5, 6, (B,)).float(),
        "L": torch.randint(0, 10, (B,)).float(),
        "d": torch.rand(B) + 0.5,
        "group_label": torch.zeros(B),
        "profile_group_label": torch.zeros(B),
        "is_coset": torch.zeros(B, dtype=torch.bool),
        "xyzcal.px.0": torch.rand(B) * 2000,
        "xyzcal.px.1": torch.rand(B) * 2000,
        "xyzcal.px.2": torch.rand(B) * 1000,
        "lp": torch.rand(B) * 0.5 + 0.5,
    }


class TestHKLLookupTable:
    def test_forward_shapes(self):
        table = HKLLookupTable(n_hkl=10)
        asu_ids = torch.tensor([0, 3, 7])
        qi, F_sq = table(asu_ids, mc_samples=5)
        assert F_sq.shape == (5, 3)
        assert isinstance(qi, Gamma)
        assert qi.concentration.shape == (3,)
        assert qi.rate.shape == (3,)

    def test_F_sq_positive(self):
        table = HKLLookupTable(n_hkl=10)
        _, F_sq = table(torch.tensor([0, 5, 9]), mc_samples=100)
        assert (F_sq > 0).all()

    def test_concentration_above_k_min(self):
        table = HKLLookupTable(n_hkl=10, k_min=0.1)
        qi, _ = table(torch.arange(10), mc_samples=1)
        assert (qi.concentration >= 0.1).all()

    def test_shared_params_for_same_id(self):
        table = HKLLookupTable(n_hkl=5)
        ids = torch.tensor([2, 2, 2])
        qi, _ = table(ids, mc_samples=1)
        assert torch.allclose(qi.concentration[0], qi.concentration[1])
        assert torch.allclose(qi.rate[0], qi.rate[1])

    def test_different_ids_independent(self):
        torch.manual_seed(42)
        table = HKLLookupTable(n_hkl=100)
        with torch.no_grad():
            table.raw_mu.weight.normal_()
        qi, _ = table(torch.tensor([0, 50]), mc_samples=1)
        assert not torch.allclose(qi.concentration[0], qi.concentration[1])

    def test_gradients_flow(self):
        table = HKLLookupTable(n_hkl=10)
        ids = torch.tensor([0, 3])
        qi, F_sq = table(ids, mc_samples=4)
        loss = F_sq.mean()
        loss.backward()
        assert table.raw_mu.weight.grad is not None
        grad_dense = table.raw_mu.weight.grad.to_dense()
        nonzero_rows = grad_dense.abs().sum(dim=1) > 0
        assert nonzero_rows[0].item()
        assert nonzero_rows[3].item()
        assert not nonzero_rows[1].item()


class TestChebyshevScale:
    def test_output_positive(self):
        scale = ChebyshevScale(degree=3, frame_min=0.0, frame_max=100.0)
        frames = torch.linspace(0, 100, 20)
        s = scale(frames)
        assert (s > 0).all()
        assert s.shape == (20,)

    def test_init_near_one(self):
        scale = ChebyshevScale(degree=5, init_scale=1.0)
        s = scale(torch.tensor([500.0]))
        assert abs(s.item() - 1.0) < 0.01

    def test_smooth(self):
        scale = ChebyshevScale(degree=3, frame_min=0.0, frame_max=100.0)
        frames = torch.linspace(0, 100, 100)
        s = scale(frames)
        diffs = (s[1:] - s[:-1]).abs()
        assert diffs.max() < 0.5

    def test_gradients(self):
        scale = ChebyshevScale(degree=3)
        frames = torch.tensor([100.0, 500.0, 900.0])
        s = scale(frames)
        s.sum().backward()
        assert scale.c.grad is not None
        assert (scale.c.grad != 0).any()


class TestScalingFactory:
    @pytest.fixture
    def scaling_cfg(self, tmp_path):
        return {
            "integrator": {
                "name": "scaling",
                "args": {
                    "data_dim": "2d",
                    "d": 1,
                    "h": 21,
                    "w": 21,
                    "n_hkl": 50,
                    "mc_samples": 2,
                    "lr": 1e-3,
                    "scale_frame_max": 1000.0,
                },
            },
            "encoders": [
                {"name": "profile_encoder", "args": {"data_dim": "2d"}},
                {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
                {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
            ],
            "surrogates": {
                "qp": {
                    "name": "learned_basis_profile",
                    "args": {"input_dim": 64, "output_dim": 441},
                },
                "qbg": {
                    "name": "gammaA",
                    "args": {"in_features": 64},
                },
            },
            "loss": {
                "name": "monochromatic_wilson",
                "args": {
                    "mc_samples": 2,
                    "eps": 1e-6,
                },
            },
            "data_loader": {
                "name": "rotation_data",
                "args": {"data_dir": str(tmp_path)},
            },
        }

    def test_construct(self, scaling_cfg):
        from integrator.model.scaling import ScalingIntegrator

        integrator = construct_integrator(scaling_cfg)
        assert isinstance(integrator, ScalingIntegrator)
        assert integrator.hkl_table.n_hkl == 50
        assert integrator.scale_fn.degree == 5

    def test_forward(self, scaling_cfg):
        integrator = construct_integrator(scaling_cfg)
        B = 4
        metadata = _mock_metadata(B)
        outputs = integrator(
            torch.rand(B, 441),
            torch.randn(B, 441),
            torch.ones(B, 441),
            metadata,
        )
        assert outputs["forward_out"]["rates"].shape == (B, 2, 441)
        assert outputs["forward_out"]["asu_id"].shape == (B,)
        assert isinstance(outputs["qi"], Gamma)

    def test_loss_backward(self, scaling_cfg):
        integrator = construct_integrator(scaling_cfg)
        B = 4
        metadata = _mock_metadata(B)
        outputs = integrator(
            torch.randint(0, 10, (B, 441)).float(),
            torch.randn(B, 441),
            torch.ones(B, 441),
            metadata,
        )
        fwd = outputs["forward_out"]

        loss_dict = integrator.loss(
            rate=fwd["rates"],
            counts=fwd["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=fwd["mask"],
            group_labels=metadata["group_label"].long(),
            metadata=metadata,
        )
        loss_dict["loss"].backward()

        assert integrator.hkl_table.raw_mu.weight.grad is not None
        assert integrator.hkl_table.raw_fano.weight.grad is not None
        assert integrator.scale_fn.c.grad is not None

    def test_sparse_embeddings(self, scaling_cfg):
        integrator = construct_integrator(scaling_cfg)
        assert integrator.hkl_table.raw_mu.sparse
        assert integrator.hkl_table.raw_fano.sparse
        assert integrator.automatic_optimization is False
