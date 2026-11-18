"""Core test suite for the integrator package.

Self-contained tests that run on CPU without W&B or external data.
All synthetic data is generated via conftest fixtures.
"""

from copy import deepcopy

import pytest
import torch

from integrator.configs.integrator import IntegratorCfg
from integrator.configs.optimizer import OptimizerConfig
SIMULATED_COLS = [
    "shoebox_median",
    "shoebox_var",
    "shoebox_mean",
    "shoebox_min",
    "shoebox_max",
    "intensity",
    "background",
    "refl_ids",
    "is_test",
    "group_label",
    "profile_group_label",
    "d",
]
from integrator.model.distributions.dirichlet import DirichletDistribution
from integrator.model.distributions.gamma import (
    GammaDistributionRepamA,
    GammaDistributionRepamB,
    GammaDistributionRepamE,
    build_gamma,
)
from integrator.model.distributions.profile_surrogates import ProfileSurrogate
from integrator.model.encoders.encoders import IntensityEncoder, ProfileEncoder
from integrator.model.integrators.base_integrator import BaseIntegrator
from integrator.registry import REGISTRY
from integrator.utils.factory_utils import construct_integrator, resolve_config

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
]

ALL_DISTS = TWO_PARAM_DISTS


def _make_dist(cls):
    """Instantiate a distribution surrogate with default test params."""
    if cls is DirichletDistribution:
        return cls(in_features=IN_FEATURES, sbox_shape=(D, H, W))
    return cls(in_features=IN_FEATURES)


def _call_dist(module, x, x_=None):
    """Call distribution forward - two-param dists always get both args."""
    if isinstance(module, tuple(TWO_PARAM_DISTS)):
        return module(x, x_ if x_ is not None else x)
    return module(x)


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
def test_distribution_dual_inputs(cls, dist_inputs):
    """Two-param distributions accept (x, x_) for separate encoder paths."""
    x, x_ = dist_inputs
    module = _make_dist(cls)
    q = module(x, x_)
    assert q.rsample().shape[0] == B


@pytest.mark.parametrize(
    "cls",
    [
        GammaDistributionRepamA,
        GammaDistributionRepamB,
    ],
    ids=lambda c: c.__name__,
)
def test_distribution_positive_params(cls, dist_inputs):
    x, x_ = dist_inputs
    module = _make_dist(cls)
    q = _call_dist(module, x, x_)
    assert (q.concentration > 0).all()
    assert (q.rate > 0).all()


@pytest.mark.parametrize(
    ("reparameterization", "expected_cls"),
    [
        ("shape_rate", GammaDistributionRepamA),
        ("mean_fano", GammaDistributionRepamB),
        ("nakagami", GammaDistributionRepamE),
        ("a", GammaDistributionRepamA),
        ("b", GammaDistributionRepamB),
        ("e", GammaDistributionRepamE),
    ],
)
def test_build_gamma_dispatch(reparameterization, expected_cls):
    """The single `gamma` builder selects the right reparameterization module."""
    module = build_gamma(
        reparameterization=reparameterization, in_features=IN_FEATURES
    )
    assert isinstance(module, expected_cls)


def test_build_gamma_default_is_shape_rate():
    assert isinstance(
        build_gamma(in_features=IN_FEATURES), GammaDistributionRepamA
    )


def test_build_gamma_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown gamma reparameterization"):
        build_gamma(reparameterization="nope", in_features=IN_FEATURES)


@pytest.mark.parametrize(
    "reparameterization", ["shape_rate", "mean_fano", "nakagami"]
)
def test_positive_constraint_unified_across_reparameterizations(
    reparameterization,
):
    """A single `positive_constraint` arg works for every reparameterization."""
    scalar = build_gamma(
        reparameterization=reparameterization,
        in_features=IN_FEATURES,
        positive_constraint="exp",
    )
    pair = build_gamma(
        reparameterization=reparameterization,
        in_features=IN_FEATURES,
        positive_constraint=["exp", "softplus"],
    )
    head1, head2 = sorted(
        n for n in vars(scalar) if n.endswith("_constraint_name")
    )
    # A scalar applies to both heads; a pair sets the two heads independently.
    assert getattr(scalar, head1) == getattr(scalar, head2) == "exp"
    assert {getattr(pair, head1), getattr(pair, head2)} == {"exp", "softplus"}


def test_positive_constraint_rejects_bad_sequence():
    with pytest.raises(ValueError, match="2-element sequence"):
        build_gamma(
            in_features=IN_FEATURES,
            positive_constraint=["exp", "softplus", "exp"],
        )


def test_gamma_registry_key_builds_module(dist_inputs):
    """`gamma` is the registered surrogate key and forwards extra args."""
    builder = REGISTRY["surrogates"]["gamma"]
    module = builder(
        reparameterization="mean_fano", in_features=IN_FEATURES, k_min=0.05
    )
    assert isinstance(module, GammaDistributionRepamB)
    x, x_ = dist_inputs
    q = module(x, x_)
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
    def test_profile_encoder_3d(self):
        enc = ProfileEncoder(
            data_dim="3d", input_shape=(D, H, W), encoder_out=IN_FEATURES
        )
        x = torch.randn(B, 1, D, H, W)
        out = enc(x)
        assert out.shape == (B, IN_FEATURES)

    def test_profile_encoder_2d(self):
        enc = ProfileEncoder(
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

    def test_registry_keys(self):
        for category, entries in REGISTRY.items():
            for name, cls in entries.items():
                assert isinstance(name, str)
                assert cls is not None

    def test_config_rejects_bad_lr(self):
        with pytest.raises(ValueError, match="lr must be positive"):
            OptimizerConfig(lr=-1)

    def test_config_rejects_bad_data_dim(self):
        with pytest.raises(ValueError, match="data_dim must be"):
            IntegratorCfg(data_dim="4d", d=3, h=21, w=21)


class TestResolveConfig:
    def test_full_config_resolves_to_itself(self, sim_config):
        """Resolving a fully-specified config is idempotent."""
        once = resolve_config(deepcopy(sim_config))
        twice = resolve_config(deepcopy(once))
        assert once == twice

    def test_synthesizes_omitted_encoders_and_surrogates(self, sim_config):
        """A config without `encoders:`/`surrogates:` is filled from the integrator."""
        minimal = deepcopy(sim_config)
        del minimal["encoders"]
        del minimal["surrogates"]

        resolved = resolve_config(deepcopy(minimal))
        # hierarchical integrator declares 5 encoders (1 profile + 4 intensity).
        assert [e["name"] for e in resolved["encoders"]] == [
            "profile_encoder",
            "intensity_encoder",
            "intensity_encoder",
            "intensity_encoder",
            "intensity_encoder",
        ]
        assert set(resolved["surrogates"]) == {"qp", "qbg", "qi"}
        # encoder_out is read from the integrator section and injected.
        eo = sim_config["integrator"]["args"]["encoder_out"]
        assert resolved["surrogates"]["qi"]["args"]["in_features"] == eo
        assert resolved["encoders"][0]["args"]["encoder_out"] == eo
        assert (
            resolved["surrogates"]["qp"]["args"]["output_dim"] == D * H * W
        )

        # It builds through the full factory path.
        model = construct_integrator(minimal)
        assert isinstance(model, BaseIntegrator)
        assert set(model.encoders.keys()) == set(
            REGISTRY["integrator"]["hierarchical"].REQUIRED_ENCODERS
        )

    def test_partial_surrogate_override_merges_over_defaults(self):
        """A partial `surrogates:` overrides named args; the rest comes from defaults."""
        cfg = {
            "integrator": {
                "name": "hierarchical",
                "args": {
                    "encoder_out": 64,
                    "data_dim": "3d",
                    "d": 3,
                    "h": 21,
                    "w": 21,
                },
            },
            "surrogates": {
                "qi": {"args": {"k_min": 0.1}},
                "qbg": {"args": {"k_min": 0.1}},
            },
            "data_loader": {"args": {"data_dir": "/nonexistent"}},
        }
        resolved = resolve_config(cfg)
        # qp is supplied entirely by the integrator default.
        assert resolved["surrogates"]["qp"]["name"] == "learned_basis_profile"
        # qi/qbg keep the overridden k_min but inherit name/reparameterization.
        qi = resolved["surrogates"]["qi"]
        assert qi["name"] == "gamma"
        assert qi["args"]["k_min"] == 0.1
        assert qi["args"]["reparameterization"] == "mean_fano"
        assert qi["args"]["in_features"] == 64

    def test_profile_prior_scale_defaults_onto_qp(self):
        """profile_prior_scale lives on the qp surrogate, defaulting to 3.0."""
        base = {
            "integrator": {
                "name": "hierarchical",
                "args": {
                    "encoder_out": 64,
                    "data_dim": "3d",
                    "d": 3,
                    "h": 21,
                    "w": 21,
                },
            },
            "data_loader": {"args": {"data_dir": "/nonexistent"}},
        }
        # Default: synthesized qp carries prior_scale 3.0.
        resolved = resolve_config(deepcopy(base))
        assert resolved["surrogates"]["qp"]["args"]["prior_scale"] == 3.0
        # Override flows through the partial-merge.
        override = deepcopy(base)
        override["surrogates"] = {"qp": {"args": {"prior_scale": 7.0}}}
        resolved = resolve_config(override)
        assert resolved["surrogates"]["qp"]["args"]["prior_scale"] == 7.0

    def test_dirichlet_qp_gets_no_prior_scale(self):
        """prior_scale is a learned-basis concept; a Dirichlet qp must not receive it."""
        cfg = {
            "integrator": {
                "name": "hierarchical",
                "args": {
                    "encoder_out": 64,
                    "data_dim": "3d",
                    "d": 3,
                    "h": 21,
                    "w": 21,
                },
            },
            "surrogates": {"qp": {"name": "dirichlet", "args": {}}},
            "data_loader": {"args": {"data_dir": "/nonexistent"}},
        }
        resolved = resolve_config(cfg)
        assert resolved["surrogates"]["qp"]["name"] == "dirichlet"
        assert "prior_scale" not in resolved["surrogates"]["qp"]["args"]
        # And it builds (DirichletDistribution takes neither prior_scale nor **kwargs).
        from integrator.utils.factory_utils import _get_surrogate_modules

        mods = _get_surrogate_modules(resolved)
        assert type(mods["qp"]).__name__ == "DirichletDistribution"

    def test_mode_fills_loss_and_data_loader(self):
        cfg = {
            "mode": "polychromatic",
            "integrator": {"name": "hierarchical", "args": {}},
            "data_loader": {"args": {}},
        }
        resolved = resolve_config(cfg)
        assert resolved["loss"]["name"] == "polychromatic_wilson"
        assert resolved["data_loader"]["name"] == "polychromatic_data"

    def test_mode_does_not_override_explicit_name(self):
        cfg = {
            "mode": "monochromatic",
            "loss": {"name": "polychromatic_wilson", "args": {}},
            "integrator": {"name": "hierarchical", "args": {}},
        }
        resolved = resolve_config(cfg)
        assert resolved["loss"]["name"] == "polychromatic_wilson"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            resolve_config({"mode": "xray", "integrator": {"name": "hierarchical"}})

    def test_misplaced_or_unknown_args_rejected(self):
        """No backward compat: args in the wrong section / unknown loss args raise."""
        base = {
            "integrator": {
                "name": "hierarchical",
                "args": {
                    "encoder_out": 64,
                    "data_dim": "3d",
                    "d": 3,
                    "h": 21,
                    "w": 21,
                },
            },
            "optimizer": {"lr": 0.001},
            "loss": {"name": "monochromatic_wilson", "args": {"n_bins": 3}},
            "data_loader": {
                "name": "rotation_data",
                "args": {"data_dir": "/nonexistent"},
            },
        }
        # Optimizer key left under the integrator is no longer relocated.
        bad_opt = deepcopy(base)
        bad_opt["integrator"]["args"]["lr"] = 0.002
        with pytest.raises(TypeError):
            construct_integrator(bad_opt)
        # Unknown loss arg (e.g. the removed mc_samples) is rejected.
        bad_loss = deepcopy(base)
        bad_loss["loss"]["args"]["mc_samples"] = 4
        with pytest.raises(ValueError):
            construct_integrator(bad_loss)


def test_profile_surrogate_carries_prior_scale():
    """The profile surrogate owns prior_scale and exposes it on its output."""
    sur = ProfileSurrogate(input_dim=IN_FEATURES, latent_dim=8, prior_scale=2.5)
    out = sur(torch.randn(B, IN_FEATURES))
    assert out.prior_scale == 2.5


def test_wilson_background_prior_scalar_or_per_bin():
    """Background prior is a shared Gamma by default, or per-resolution-bin when arrays are given."""
    loss_cls = REGISTRY["loss"]["monochromatic_wilson"]
    # scalar -> 0-d buffer shared across reflections
    shared = loss_cls()
    assert shared.bg_concentration.ndim == 0
    # per-bin arrays -> (n_bins,) buffer indexed by group_label
    per_bin = loss_cls(
        n_bins=3,
        bg_concentration=[1.0, 2.0, 3.0],
        bg_rate=[0.5, 0.5, 0.5],
    )
    assert per_bin.bg_concentration.ndim == 1
    assert per_bin.bg_concentration.numel() == 3


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
    metadata["group_label"] = torch.randint(0, 5, (batch_size,))
    metadata["d"] = torch.rand(batch_size) * 5.0 + 1.0
    return counts, standardized, masks, metadata


class TestModelForward:
    def test_modelc_forward(self, sim_config):
        model = construct_integrator(sim_config)
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
        return model.loss, out, metadata

    def test_loss_forward(self, loss_inputs):
        loss_fn, out, metadata = loss_inputs
        result = loss_fn(
            rate=out["forward_out"]["rates"],
            counts=out["forward_out"]["counts"],
            qp=out["qp"],
            qi=out["qi"],
            qbg=out["qbg"],
            mask=out["forward_out"]["mask"],
            group_labels=metadata["group_label"],
            metadata=metadata,
        )
        for key in ("loss", "neg_ll_mean", "kl_mean"):
            assert key in result

    def test_loss_finite(self, loss_inputs):
        loss_fn, out, metadata = loss_inputs
        result = loss_fn(
            rate=out["forward_out"]["rates"],
            counts=out["forward_out"]["counts"],
            qp=out["qp"],
            qi=out["qi"],
            qbg=out["qbg"],
            mask=out["forward_out"]["mask"],
            group_labels=metadata["group_label"],
            metadata=metadata,
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





