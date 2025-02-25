from .model.encoders import (
    CNNResNet,
    FcEncoder,
    CNNResNet2,
    DevEncoder,
    CNN_3d,
    FcResNet,
)
from .model.decoders import Decoder
from .model.profiles import (
    DirichletProfile,
    BetaProfile,
)
from .model.loss import Loss
from .model.integrators import DefaultIntegrator, DevIntegrator
from .model.distribution import GammaDistribution, LogNormalDistribution
from .data_loaders import ShoeboxDataModule
import torch

REGISTRY = {
    "encoder": {
        "encoder1": CNNResNet,  # done
        "fc_encoder": FcEncoder,  # for metadata
        "dev_encoder": DevEncoder,
        "fc_resnet": FcResNet,  # done
        "3d_cnn": CNN_3d,  # done
    },
    "decoder": {
        "decoder1": Decoder,
    },
    "profile": {
        "dirichlet": DirichletProfile,
        "beta": BetaProfile,
    },
    "loss": {
        "elbo": Loss,
    },
    "integrator": {
        "integrator1": DefaultIntegrator,
        "test_integrator": DevIntegrator,
    },
    "q_I": {
        "gamma": GammaDistribution,
        "log_normal": LogNormalDistribution,
    },
    "q_bg": {
        "gamma": GammaDistribution,
    },
    "data_loader": {
        "default": ShoeboxDataModule,
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
        },
        "p_I": {
            "gamma": torch.distributions.gamma.Gamma,
        },
        "p_p": {
            "beta": torch.distributions.beta.Beta,
            "laplace": torch.distributions.laplace.Laplace,
        },
    },
}
