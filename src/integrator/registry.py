from .model.encoders import *
from .model.decoders import *
from .model.profiles import *
from .model.loss import *
from .model.integrators import *
from .model.distribution import *
from .data_loaders import ShoeboxDataModule
import torch

REGISTRY = {
    "metadata_encoder": {
        "mlp_metadata_encoder": MLPMetadataEncoder,  # for metadata
    },
    "encoder": {
        "mlp_image_encoder": MLPImageEncoder,  # done
    },
    "image_encoder": {
        "cnn_3d": CNNResNet2,  # done
        "mlp_image_encoder": MLPImageEncoder,  # done
        "3d_cnn": CNN_3d,  # shoebox encoder
        "dev_encoder": DevEncoder,
        "dirchlet_concentration": DirichletConcentration,
    },
    "decoder": {
        "default_decoder": Decoder,
        "mvn_decoder": MVNDecoder,
        "unet_decoder": UnetDecoder,
        "unet_decoder2": UnetDecoder2,
    },
    "profile": {
        "dirichlet": DirichletProfile,
        "beta": BetaProfile,
        "mvn": MVNProfile,
    },
    "loss": {
        "elbo": Loss,
        "mvn_loss": MVNLoss,
        "unet_loss": UnetLoss,
    },
    "integrator": {
        "default_integrator": DefaultIntegrator,
        "test_integrator": DevIntegrator,
        "mvn_integrator": MVNIntegrator,
        "mlp_integrator": MLPIntegrator,
    },
    "q_I": {
        "gamma": GammaDistribution,
        "log_normal": LogNormalDistribution,
    },
    "q_bg": {
        "gamma": GammaDistribution,
        "half_normal": HalfNormalDistribution,
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
            "half_normal": torch.distributions.half_normal.HalfNormal,
        },
        "p_I": {
            "gamma": torch.distributions.gamma.Gamma,
            "log_normal": torch.distributions.log_normal.LogNormal,
        },
        "p_p": {
            "beta": torch.distributions.beta.Beta,
            "laplace": torch.distributions.laplace.Laplace,
        },
    },
}
# %%
