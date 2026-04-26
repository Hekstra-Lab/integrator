"""Core test suite for the integrator package.

Self-contained tests that run on CPU without W&B or external data.
All synthetic data is generated via conftest fixtures.
"""

import pytest
import torch

from integrator.configs.integrator import IntegratorCfg
from integrator.data_loaders.data_module import (
    SIMULATED_COLS,
    SimulatedShoeboxLoader,
)
from integrator.model.distributions.dirichlet import DirichletDistribution
from integrator.model.distributions.folded_normal import (
    FoldedNormalDistribution,
)
from integrator.model.distributions.gamma import (
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamC,
    GammaDistributionRepamD,
)
from integrator.model.distributions.log_normal import LogNormalDistribution
from integrator.model.encoders.encoders import IntensityEncoder, ShoeboxEncoder
from integrator.model.integrators.base_integrator import BaseIntegrator
from integrator.model.loss.per_bin_loss import PerBinLoss
from integrator.registry import REGISTRY
from integrator.utils.factory_utils import (
    construct_data_loader,
    construct_integrator,
    construct_trainer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B = 8  # batch size for unit tests
IN_FEATURES = 64
D, H, W = 3, 21, 21
N_PIXELS = D * H * W

# ---------------------------------------------------------------------------
# A. Distribution tests
# ---------------------------------------------------------------------------

# Two-param distributions that support separate_inputs mode
TWO_PARAM_DISTS = [
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamC,
    GammaDistributionRepamD,
    FoldedNormalDistribution,
    LogNormalDistribution,
]

ALL_DISTS = TWO_PARAM_DISTS


def _make_dist(cls, separate_inputs=False):
    """Instantiate a distribution surrogate with default test params."""
    if cls is DirichletDistribution:
        return cls(in_features=IN_FEATURES, sbox_shape=(D, H, W))
    if cls in TWO_PARAM_DISTS:
        return cls(in_features=IN_FEATURES, separate_inputs=separate_inputs)
    return cls(in_features=IN_FEATURES)


def _call_dist(module, x, x_=None):
    """Call distribution forward."""
    return module(x, x_) if x_ is not None else module(x)


@pytest.fixture(params=ALL_DISTS, ids=lambda c: c.__name__)
def dist_module(request):
    return _make_dist(request.param)


@pytest.fixture()
def dist_inputs():
    torch.manual_seed(42)
    x = torch.randn(B, IN_FEATURES)
    x_ = torch.randn(B, IN_FEATURES)
    return x, x_


class TestDistributions:
    def test_distribution_forward_shape(self, dist_module, dist_inputs):
        x, _ = dist_inputs
        q = _call_dist(dist_module, x)
        assert q.batch_shape[0] == B

    def test_distribution_sample_shape(self, dist_module, dist_inputs):
        x, _ = dist_inputs
        q = _call_dist(dist_module, x)
        samples = q.rsample([5])
        assert samples.shape[0] == 5

    def test_distribution_no_nan(self, dist_module, dist_inputs):
        x, _ = dist_inputs
        q = _call_dist(dist_module, x)
        assert torch.isfinite(q.mean).all()
        assert torch.isfinite(q.variance).all()

    def test_distribution_gradient_flow(self, dist_module, dist_inputs):
        x, _ = dist_inputs
        x.requires_grad_(True)
        q = _call_dist(dist_module, x)
        loss = q.rsample([3]).sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "cls",
    TWO_PARAM_DISTS,
    ids=lambda c: c.__name__,
)
def test_distribution_separate_inputs(cls, dist_inputs):
    """Two-param distributions with separate_inputs=True accept (x, x_)."""
    x, x_ = dist_inputs
    module = _make_dist(cls, separate_inputs=True)
    q = module(x, x_)
    assert q.rsample().shape[0] == B


@pytest.mark.parametrize(
    "cls",
    TWO_PARAM_DISTS,
    ids=lambda c: c.__name__,
)
def test_distribution_separate_inputs_fallback(cls, dist_inputs):
    """separate_inputs=True should still work with x_ omitted."""
    x, _ = dist_inputs
    module = _make_dist(cls, separate_inputs=True)
    q = module(x, None)
    assert q.rsample().shape[0] == B


@pytest.mark.parametrize(
    "cls",
    [
        GammaDistributionRepamA,
        GammaDistributionRepamB,
        GammaDistributionRepamC,
        GammaDistributionRepamD,
    ],
    ids=lambda c: c.__name__,
)
def test_distribution_positive_params(cls, dist_inputs):
    x, x_ = dist_inputs
    module = _make_dist(cls)
    q = _call_dist(module, x, x_)
    assert (q.concentration > 0).all()
    assert (q.rate > 0).all()


def test_dirichlet_concentration_shape(dist_inputs):
    x, _ = dist_inputs
    module = DirichletDistribution(
        in_features=IN_FEATURES, sbox_shape=(D, H, W)
    )
    q = module(x)
    assert q.concentration.shape == (B, N_PIXELS)


# ---------------------------------------------------------------------------
# B. Encoder tests
# ---------------------------------------------------------------------------


class TestEncoders:
    def test_shoebox_encoder_3d(self):
        enc = ShoeboxEncoder(
            data_dim="3d", input_shape=(D, H, W), encoder_out=IN_FEATURES
        )
        x = torch.randn(B, 1, D, H, W)
        out = enc(x)
        assert out.shape == (B, IN_FEATURES)

    def test_shoebox_encoder_2d(self):
        enc = ShoeboxEncoder(
            data_dim="2d",
            input_shape=(H, W),
            encoder_out=IN_FEATURES,
            conv1_kernel_size=(3, 3),
            conv1_padding=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2),
            conv2_kernel_size=(3, 3),
            conv2_padding=(0, 0),
        )
        x = torch.randn(B, 1, H, W)
        out = enc(x)
        assert out.shape == (B, IN_FEATURES)

    def test_intensity_encoder_3d(self):
        enc = IntensityEncoder(
            data_dim="3d",
            encoder_out=IN_FEATURES,
            conv1_kernel_size=(3, 3, 3),
            conv1_padding=(1, 1, 1),
            pool_kernel_size=(1, 2, 2),
            pool_stride=(1, 2, 2),
            conv2_kernel_size=(3, 3, 3),
            conv2_padding=(0, 0, 0),
            conv3_kernel_size=(3, 3, 3),
            conv3_padding=(1, 1, 1),
        )
        x = torch.randn(B, 1, D, H, W)
        out = enc(x)
        assert out.shape == (B, IN_FEATURES)

    def test_intensity_encoder_2d(self):
        enc = IntensityEncoder(data_dim="2d", encoder_out=IN_FEATURES)
        x = torch.randn(B, 1, H, W)
        out = enc(x)
        assert out.shape == (B, IN_FEATURES)


# ---------------------------------------------------------------------------
# C. Factory & config tests
# ---------------------------------------------------------------------------


class TestFactoryConfig:
    def test_construct_integrator(self, sim_config):
        model = construct_integrator(sim_config)
        assert isinstance(model, BaseIntegrator)

    def test_construct_data_loader(self, sim_config):
        dl = construct_data_loader(sim_config)
        dl.setup()
        batch = next(iter(dl.train_dataloader()))
        assert len(batch) == 4  # counts, standardized, masks, metadata

    def test_registry_keys(self):
        for category, entries in REGISTRY.items():
            for name, cls in entries.items():
                assert isinstance(name, str)
                assert cls is not None

    def test_config_rejects_bad_lr(self):
        with pytest.raises(ValueError, match="lr must be positive"):
            IntegratorCfg(data_dim="3d", d=3, h=21, w=21, lr=-1)

    def test_config_rejects_bad_data_dim(self):
        with pytest.raises(ValueError, match="data_dim must be"):
            IntegratorCfg(data_dim="4d", d=3, h=21, w=21)


# ---------------------------------------------------------------------------
# D. Model forward-pass tests
# ---------------------------------------------------------------------------


def _make_batch(batch_size=B):
    """Create a synthetic batch matching SimulatedShoeboxLoader format."""
    torch.manual_seed(0)
    counts = torch.poisson(torch.full((batch_size, N_PIXELS), 5.0))
    standardized = torch.randn(batch_size, N_PIXELS)
    masks = torch.ones(batch_size, N_PIXELS)
    metadata = {col: torch.rand(batch_size) for col in SIMULATED_COLS}
    metadata["refl_id"] = torch.arange(batch_size, dtype=torch.float32)
    metadata["refl_ids"] = metadata["refl_id"]
    metadata["is_test"] = torch.zeros(batch_size, dtype=torch.bool)
    return counts, standardized, masks, metadata


class TestModelForward:
    def test_modela_forward(self, sim_config):
        model = construct_integrator(sim_config)
        model.eval()
        counts, shoebox, mask, metadata = _make_batch()
        with torch.no_grad():
            out = model(counts, shoebox, mask, metadata)
        for key in ("forward_out", "qp", "qi", "qbg"):
            assert key in out

    def test_modelb_forward(self, sim_config):
        cfg = _modelb_config(sim_config)
        model = construct_integrator(cfg)
        model.eval()
        counts, shoebox, mask, metadata = _make_batch()
        with torch.no_grad():
            out = model(counts, shoebox, mask, metadata)
        for key in ("forward_out", "qp", "qi", "qbg"):
            assert key in out

    def test_training_step_returns_loss(self, sim_config):
        model = construct_integrator(sim_config)
        batch = _make_batch()
        result = model.training_step(batch, 0)
        assert "loss" in result
        assert torch.isfinite(result["loss"])

    def test_predict_step_filters_keys(self, sim_config):
        model = construct_integrator(sim_config)
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            preds = model.predict_step(batch, 0)
        # Only keys in predict_keys should be present
        for k in preds:
            assert k in model.predict_keys


def _modelb_config(base_config):
    """Derive a ModelB config from the base ModelA config."""
    import copy

    cfg = copy.deepcopy(base_config)
    cfg["integrator"]["name"] = "modelb"
    encoder_out = cfg["integrator"]["args"]["encoder_out"]

    # ModelB needs 3 encoders: profile, k, r
    intensity_enc = cfg["encoders"][1]
    cfg["encoders"] = [
        cfg["encoders"][0],  # profile (shoebox)
        copy.deepcopy(intensity_enc),  # k
        copy.deepcopy(intensity_enc),  # r
    ]

    # ModelB uses two-input surrogates for qi and qbg
    cfg["surrogates"]["qi"]["name"] = "gammaC"
    cfg["surrogates"]["qbg"]["name"] = "gammaC"

    return cfg


# ---------------------------------------------------------------------------
# E. Loss tests
# ---------------------------------------------------------------------------


class TestLoss:
    @pytest.fixture()
    def loss_inputs(self, sim_config):
        model = construct_integrator(sim_config)
        batch = _make_batch()
        counts, shoebox, mask, metadata = batch
        with torch.no_grad():
            out = model(counts, shoebox, mask, metadata)
        return model.loss, out, counts, mask

    def test_loss_forward(self, loss_inputs):
        loss_fn, out, counts, mask = loss_inputs
        result = loss_fn(
            rate=out["forward_out"]["rates"],
            counts=out["forward_out"]["counts"],
            qp=out["qp"],
            qi=out["qi"],
            qbg=out["qbg"],
            mask=out["forward_out"]["mask"],
        )
        for key in ("loss", "neg_ll_mean", "kl_mean"):
            assert key in result

    def test_loss_finite(self, loss_inputs):
        loss_fn, out, counts, mask = loss_inputs
        result = loss_fn(
            rate=out["forward_out"]["rates"],
            counts=out["forward_out"]["counts"],
            qp=out["qp"],
            qi=out["qi"],
            qbg=out["qbg"],
            mask=out["forward_out"]["mask"],
        )
        for key, val in result.items():
            assert torch.isfinite(val), f"{key} is not finite"

    def test_loss_backward(self, sim_config):
        model = construct_integrator(sim_config)
        batch = _make_batch()
        result = model.training_step(batch, 0)
        result["loss"].backward()
        # At least some parameters should have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ---------------------------------------------------------------------------
# F. Data loader tests
# ---------------------------------------------------------------------------

SIM_FILE_NAMES = {
    "counts": "counts.pt",
    "masks": "masks.pt",
    "stats": "stats_anscombe.pt",
    "reference": "reference.pt",
    "standardized_counts": None,
}


class TestDataLoader:
    def test_simulated_loader_setup(self, sim_data_dir):
        loader = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir),
            batch_size=10,
            num_workers=0,
            subset_size=100,
            shoebox_file_names=SIM_FILE_NAMES,
        )
        loader.setup()
        assert loader.train_dataset is not None
        assert loader.val_dataset is not None

    def test_simulated_loader_batch_shape(self, sim_data_dir):
        loader = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir),
            batch_size=10,
            num_workers=0,
            subset_size=100,
            shoebox_file_names=SIM_FILE_NAMES,
        )
        loader.setup()
        batch = next(iter(loader.train_dataloader()))
        counts, standardized, masks, meta = batch
        assert counts.dim() == 2
        assert standardized.dim() == 2
        assert masks.dim() == 2
        assert isinstance(meta, dict)

    def test_simulated_loader_metadata_keys(self, sim_data_dir):
        loader = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir),
            batch_size=10,
            num_workers=0,
            subset_size=100,
            shoebox_file_names=SIM_FILE_NAMES,
        )
        loader.setup()
        batch = next(iter(loader.train_dataloader()))
        _, _, _, meta = batch
        for col in SIMULATED_COLS:
            assert col in meta, f"Missing metadata key: {col}"

    def test_simulated_loader_refl_ids_alias(
        self, sim_data_dir, sim_reference
    ):
        # Remove refl_ids and add refl_id to test the alias logic
        # (data_module aliases refl_id -> refl_ids when refl_ids is missing)
        ref = dict(sim_reference)
        del ref["refl_ids"]
        ref["refl_id"] = torch.arange(len(ref["is_test"]), dtype=torch.float32)
        torch.save(ref, sim_data_dir / "reference.pt")

        loader = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir),
            batch_size=10,
            num_workers=0,
            subset_size=100,
            shoebox_file_names=SIM_FILE_NAMES,
        )
        loader.setup()
        batch = next(iter(loader.train_dataloader()))
        _, _, _, meta = batch
        assert "refl_ids" in meta

    # ------------------------------------------------------------------
    # Anscombe transformation tests
    # ------------------------------------------------------------------

    def test_anscombe_false_formula(self, sim_data_dir):
        """anscombe=False: standardized = (counts * masks - stats[0]) / stats[1].sqrt()"""
        loader = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir),
            batch_size=500,
            num_workers=0,
            shoebox_file_names=SIM_FILE_NAMES,
            anscombe=False,
        )
        loader.setup()

        counts = torch.load(
            sim_data_dir / "counts.pt", weights_only=False
        ).squeeze(-1)
        masks = torch.load(
            sim_data_dir / "masks.pt", weights_only=False
        ).squeeze(-1)
        stats = torch.load(
            sim_data_dir / "stats_anscombe.pt", weights_only=False
        )

        expected = ((counts * masks) - stats[0]) / stats[1].sqrt()
        actual = loader.full_dataset.standardized_counts

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_anscombe_true_formula(self, sim_data_dir_anscombe):
        """anscombe=True: standardized = ((2*sqrt(c+0.375) - stats[0]) / sqrt(stats[1])) * masks"""
        loader = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir_anscombe),
            batch_size=500,
            num_workers=0,
            shoebox_file_names=SIM_FILE_NAMES,
            anscombe=True,
        )
        loader.setup()

        counts = torch.load(
            sim_data_dir_anscombe / "counts.pt", weights_only=False
        ).squeeze(-1)
        masks = torch.load(
            sim_data_dir_anscombe / "masks.pt", weights_only=False
        ).squeeze(-1)
        stats = torch.load(
            sim_data_dir_anscombe / "stats_anscombe.pt", weights_only=False
        )

        anscombe_transformed = 2 * (counts + 0.375).sqrt()
        expected = (
            (anscombe_transformed - stats[0]) / stats[1].sqrt()
        ) * masks
        actual = loader.full_dataset.standardized_counts

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_anscombe_true_differs_from_false(self, sim_data_dir_anscombe):
        """anscombe=True and anscombe=False produce different standardized counts."""
        loader_true = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir_anscombe),
            batch_size=500,
            num_workers=0,
            shoebox_file_names=SIM_FILE_NAMES,
            anscombe=True,
        )
        loader_false = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir_anscombe),
            batch_size=500,
            num_workers=0,
            shoebox_file_names=SIM_FILE_NAMES,
            anscombe=False,
        )
        loader_true.setup()
        loader_false.setup()

        std_true = loader_true.full_dataset.standardized_counts
        std_false = loader_false.full_dataset.standardized_counts

        assert not torch.allclose(std_true, std_false)

    def test_anscombe_true_masked_pixels_are_zero(self, sim_data_dir_anscombe):
        """anscombe=True: pixels where mask=0 should have standardized_counts=0."""
        loader = SimulatedShoeboxLoader(
            data_dir=str(sim_data_dir_anscombe),
            batch_size=500,
            num_workers=0,
            shoebox_file_names=SIM_FILE_NAMES,
            anscombe=True,
        )
        loader.setup()

        masks = torch.load(
            sim_data_dir_anscombe / "masks.pt", weights_only=False
        ).squeeze(-1)
        std = loader.full_dataset.standardized_counts

        # The fixture masks out pixel 0 for all samples
        assert (masks[:, 0] == 0.0).all(), (
            "fixture should have mask=0 at pixel 0"
        )
        assert (std[:, 0] == 0.0).all(), (
            "masked pixels must be 0 in standardized counts"
        )


# ---------------------------------------------------------------------------
# G. Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_one_epoch_train_val(self, sim_config):
        model = construct_integrator(sim_config)
        data = construct_data_loader(sim_config)
        trainer = construct_trainer(sim_config, logger=False)

        data.setup()
        trainer.fit(model, datamodule=data)
        assert trainer.current_epoch == 1


# ---------------------------------------------------------------------------
# H. PerBinLoss tests
# ---------------------------------------------------------------------------


class TestPerBinLoss:
    def test_construct_per_bin_integrator(self, per_bin_sim_config):
        model = construct_integrator(per_bin_sim_config)
        assert isinstance(model, BaseIntegrator)
        assert isinstance(model.loss, PerBinLoss)

    def test_per_bin_loss_has_buffers(self, per_bin_sim_config):
        model = construct_integrator(per_bin_sim_config)
        loss = model.loss
        assert hasattr(loss, "tau_per_group")
        assert hasattr(loss, "bg_rate_per_group")
        assert loss.tau_per_group.dim() == 1
        assert loss.bg_rate_per_group.dim() == 1

    def test_per_bin_loss_forward(self, per_bin_sim_config):
        model = construct_integrator(per_bin_sim_config)
        model.eval()
        counts, shoebox, mask, metadata = _make_batch()
        metadata["group_label"] = torch.randint(0, 5, (B,)).float()
        with torch.no_grad():
            out = model(counts, shoebox, mask, metadata)
        assert "forward_out" in out

    def test_per_bin_loss_training_step(self, per_bin_sim_config):
        model = construct_integrator(per_bin_sim_config)
        counts, shoebox, mask, metadata = _make_batch()
        metadata["group_label"] = torch.randint(0, 5, (B,)).float()
        batch = (counts, shoebox, mask, metadata)
        result = model.training_step(batch, 0)
        assert "loss" in result
        assert torch.isfinite(result["loss"])

    def test_per_bin_loss_backward(self, per_bin_sim_config):
        model = construct_integrator(per_bin_sim_config)
        counts, shoebox, mask, metadata = _make_batch()
        metadata["group_label"] = torch.randint(0, 5, (B,)).float()
        batch = (counts, shoebox, mask, metadata)
        result = model.training_step(batch, 0)
        result["loss"].backward()
        # Check gradients flow through encoders
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite grad in {name}"
                )

    def test_one_epoch_per_bin(self, per_bin_sim_config):
        model = construct_integrator(per_bin_sim_config)
        data = construct_data_loader(per_bin_sim_config)
        trainer = construct_trainer(per_bin_sim_config, logger=False)

        data.setup()
        trainer.fit(model, datamodule=data)
        assert trainer.current_epoch == 1
