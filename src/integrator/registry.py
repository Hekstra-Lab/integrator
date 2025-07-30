import torch

from .data_loaders import ShoeboxDataModule, ShoeboxDataModule2, ShoeboxDataModule2D
from .model.distributions import (
    DirichletDistribution,
    FoldedNormalDistribution,
    GammaDistribution,
    HalfNormalDistribution,
    LogNormalDistribution,
    MVNDistribution,
    NormalDistribution,
)
from .model.encoders import (
    IntensityEncoder,
    IntensityEncoder2D,
    MLPMetadataEncoder,
    ShoeboxEncoder,
    ShoeboxEncoder2D,
)
from .model.integrators import Integrator, Integrator2D, LRMVNIntegrator, Model2
from .model.loss import Loss, Loss2, LRMVNLoss, MVNLoss

REGISTRY = {
    "metadata_encoders": {
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "shoebox_encoders": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
        "shoebox_encoder_2d": ShoeboxEncoder2D,
        "intensity_encoder_2d": IntensityEncoder2D,
    },
    "loss": {
        "elbo": Loss,
        "mvn_loss": MVNLoss,
        "lrmvn_loss": LRMVNLoss,
        "loss2": Loss2,
    },
    "integrator": {
        "lrmvn_integrator": LRMVNIntegrator,
        "integrator": Integrator,
        "model2": Model2,
        "integrator_2d": Integrator2D,
    },
    "qi": {
        "gamma": GammaDistribution,
        "log_normal": LogNormalDistribution,
        "normal": NormalDistribution,
        "folded_normal": FoldedNormalDistribution,
    },
    "qp": {
        "dirichlet": DirichletDistribution,
        "mvn": MVNDistribution,
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
            "gamma": torch.distributions.gamma.Gamma,
            "half_normal": torch.distributions.half_normal.HalfNormal,
            "half_cauchy": torch.distributions.half_cauchy.HalfCauchy,
            "exponential": torch.distributions.exponential.Exponential,
        },
        "p_I": {
            "gamma": torch.distributions.gamma.Gamma,
            "log_normal": torch.distributions.log_normal.LogNormal,
            "exponential": torch.distributions.exponential.Exponential,
            "half_normal": torch.distributions.half_normal.HalfNormal,
            "half_cauchy": torch.distributions.half_cauchy.HalfCauchy,
        },
        "p_prf": {
            "laplace": torch.distributions.laplace.Laplace,
        },
    },
}
