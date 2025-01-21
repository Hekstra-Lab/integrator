from .model.encoders import CNNResNet, FcEncoder
from .model.decoders import Decoder
from .model.profiles import DirichletProfile
from .model.loss import Loss
from .model.integrators import DefaultIntegrator
from .model.distribution import GammaDistribution
from .data_loaders import ShoeboxDataModule
import torch


REGISTRY = {
    "encoder": {
        "encoder1": CNNResNet,
        "fc_encoder": FcEncoder,
    },
    "decoder": {
        "decoder1": Decoder,
    },
    "profile": {
        "dirichlet": DirichletProfile,
    },
    "loss": {
        "loss1": Loss,
    },
    "integrator": {
        "integrator1": DefaultIntegrator,
    },
    "q_I": {
        "gamma": GammaDistribution,
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
        }
    }
}
