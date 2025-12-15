import torch

from integrator.data_loaders import (
    ShoeboxDataModule,
    ShoeboxDataModule2,
    ShoeboxDataModule2D,
)
from integrator.model.distributions import (
    DirichletDistribution,
    FoldedNormalDistribution,
    GammaDistribution,
    GammaDistributionRepamA,
    HalfNormalDistribution,
    LogNormalDistribution,
)
from integrator.model.encoders import (
    IntensityEncoder,
    IntensityEncoder2DMinimal,
    MLPMetadataEncoder,
    ProfileEncoder2DMinimal,
    ShoeboxEncoder,
)
from integrator.model.integrators import Integrator, IntegratorModelB
from integrator.model.loss import Loss

REGISTRY = {
    "encoders": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
        "mlp_metadata_encoder": MLPMetadataEncoder,
        "shoebox_minimal": ProfileEncoder2DMinimal,
        "intensity_encoder_minimal": IntensityEncoder2DMinimal,
    },
    "loss": {
        "elbo": Loss,
        "loss": Loss,
    },
    "integrator": {
        "integrator": Integrator,
        "modelb": IntegratorModelB,
    },
    "qi": {
        "gamma": GammaDistribution,
        "gammaA": GammaDistributionRepamA,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
    },
    "qp": {
        "dirichlet": DirichletDistribution,
    },
    "qbg": {
        "gamma": GammaDistribution,
        "gammaA": GammaDistributionRepamA,
        "half_normal": HalfNormalDistribution,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module": ShoeboxDataModule2,
        "shoebox_data_module_2d": ShoeboxDataModule2D,
    },
}

ARGUMENT_RESOLVER = {
    "trainer": {
        "accelerator": {
            "auto": lambda: "gpu" if torch.cuda.is_available() else "cpu",
            "gpu": "gpu",
            "cpu": "cpu",
        },
    },
    "loss": {
        "p_bg": {
            "exponential": torch.distributions.Exponential,
            "gamma": torch.distributions.Gamma,
            "half_cauchy": torch.distributions.half_cauchy.HalfCauchy,
            "half_normal": torch.distributions.half_normal.HalfNormal,
            "log_normal": torch.distributions.LogNormal,
        },
        "p_I": {
            "exponential": torch.distributions.Exponential,
            "gamma": torch.distributions.Gamma,
            "half_cauchy": torch.distributions.half_cauchy.HalfCauchy,
            "half_normal": torch.distributions.half_normal.HalfNormal,
            "log_normal": torch.distributions.LogNormal,
        },
        "p_prf": {"dirichlet": torch.distributions.Dirichlet},
    },
}
