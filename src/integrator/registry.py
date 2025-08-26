import torch

from .data_loaders import (
    ShoeboxDataModule,
    ShoeboxDataModule2,
    ShoeboxDataModule2D,
)
from .model.distributions import (
    DirichletDistribution,
    FoldedNormalDistribution,
    GammaDistribution,
    HalfNormalDistribution,
    LogNormalDistribution,
)
from .model.encoders import (
    IntensityEncoder,
    IntensityEncoder2D,
    MLPMetadataEncoder,
    ShoeboxEncoder,
    ShoeboxEncoder2D,
)
from .model.integrators import Integrator
from .model.loss import Loss

REGISTRY = {
    "encoders": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
        "shoebox_encoder_2d": ShoeboxEncoder2D,
        "intensity_encoder_2d": IntensityEncoder2D,
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "loss": {
        "elbo": Loss,
        "loss": Loss,
    },
    "integrator": {"integrator": Integrator},
    "qi": {
        "gamma": GammaDistribution,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
    },
    "qp": {
        "dirichlet": DirichletDistribution,
    },
    "qbg": {
        "gamma": GammaDistribution,
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
