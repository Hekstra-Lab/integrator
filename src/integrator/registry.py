import torch

from .data_loaders import ShoeboxDataModule, ShoeboxDataModule2
from .model.decoders import Decoder, MVNDecoder
from .model.distribution import (
    FoldedNormalDistribution,
    GammaDistribution,
    HalfNormalDistribution,
    LogNormalDistribution,
    NormalDistribution,
)
from .model.encoders import (
    IntensityEncoder,
    MLPImageEncoder,
    MLPMetadataEncoder,
    ShoeboxEncoder,
)
from .model.integrators import (
    DefaultIntegrator,
    Integrator,
    LRMVNIntegrator,
)
from .model.loss import Loss, Loss2, LRMVNLoss, MVNLoss
from .model.profiles import DirichletProfile, MVNProfile

REGISTRY = {
    "metadata_encoder": {
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "encoder": {
        "mlp_image_encoder": MLPImageEncoder,
        "shoebox_encoder": ShoeboxEncoder,
    },
    "image_encoder": {
        "mlp_image_encoder": MLPImageEncoder,
        "shoebox_encoder": ShoeboxEncoder,
        "mlp_metadata_encoder": MLPMetadataEncoder,
    },
    "profile_encoder": {
        "shoebox_encoder": ShoeboxEncoder,
    },
    "intensity_encoder": {
        "shoebox_encoder": ShoeboxEncoder,
        "intensity_encoder": IntensityEncoder,
    },
    "decoder": {
        "default_decoder": Decoder,
        "mvn_decoder": MVNDecoder,
    },
    "profile": {
        "dirichlet": DirichletProfile,
        "mvn": MVNProfile,
    },
    "loss": {
        "elbo": Loss,
        "mvn_loss": MVNLoss,
        "lrmvn_loss": LRMVNLoss,
        "loss2": Loss2,
    },
    "integrator": {
        "default_integrator": DefaultIntegrator,
        "lrmvn_integrator": LRMVNIntegrator,
        "integrator": Integrator,
    },
    "q_I": {
        "gamma": GammaDistribution,
        "log_normal": LogNormalDistribution,
        "normal": NormalDistribution,
        "folded_normal": FoldedNormalDistribution,
    },
    "q_bg": {
        "gamma": GammaDistribution,
        "half_normal": HalfNormalDistribution,
        "log_normal": LogNormalDistribution,
        "folded_normal": FoldedNormalDistribution,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
        "shoebox_data_module": ShoeboxDataModule2,
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
        "p_p": {
            "laplace": torch.distributions.laplace.Laplace,
        },
    },
}
